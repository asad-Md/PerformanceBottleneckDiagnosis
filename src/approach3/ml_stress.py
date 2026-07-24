#!/usr/bin/env python3
"""
ml_stress.py — Real ML training workloads used purely as a CPU/RAM stress
generator on a Ryzen 7 7840HS / 16GB RAM laptop (HP Omen 16).

Forces everything onto CPU (no CUDA) so the load lands where you want it:
CPU cores + RAM + memory bandwidth. Batch sizes / model sizes are picked
deliberately large so they exceed 16GB RAM and force swapping / thrashing.

Modes:
  llm       - finetune GPT-2 (medium/large) on WikiText-103
  cnn       - train ResNet-50 on CIFAR-100 with heavy augmentation + workers
  combined  - runs llm + cnn simultaneously in separate processes
  bigdata   - heavy pandas/numpy transforms over Amazon Polarity (4M rows)

Usage examples:
  python3 ml_stress.py --mode llm --model gpt2-medium --batch-size 32 --seq-len 512 --workers 16
  python3 ml_stress.py --mode cnn --batch-size 512 --workers 16
  python3 ml_stress.py --mode combined --workers 16
  python3 ml_stress.py --mode bigdata --workers 16
"""

import os
# Force CPU-only BEFORE torch import, so nothing sneaks onto the GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
import time
import argparse
import multiprocessing as mp

def log(msg):
    print(f"[ml_stress] {msg}", flush=True)


# ----------------------------------------------------------------------
# LLM mode: finetune GPT-2 on WikiText-103 (CPU only, big batch/seq)
# ----------------------------------------------------------------------
def run_llm(model_name="gpt2-medium", batch_size=32, seq_len=512, workers=16, epochs=1000):
    import torch
    from torch.utils.data import DataLoader
    from transformers import GPT2LMHeadModel, GPT2TokenizerFast
    from datasets import load_dataset

    torch.set_num_threads(os.cpu_count())
    log(f"Loading tokenizer/model: {model_name} (CPU)")
    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_name)
    model.train()

    log("Loading WikiText-103 (raw) — first run will download ~500MB")
    ds = load_dataset("Salesforce/wikitext", "wikitext-103-raw-v1", split="train")
    ds = ds.filter(lambda x: len(x["text"].strip()) > 200)

    def tokenize(batch):
        return tokenizer(batch["text"], truncation=True, padding="max_length",
                          max_length=seq_len, return_tensors=None)

    ds = ds.map(tokenize, batched=True, remove_columns=["text"], num_proc=max(1, workers // 2))
    ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    loader = DataLoader(ds, batch_size=batch_size, shuffle=True,
                         num_workers=workers, pin_memory=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    # Deliberately hold onto extra activation copies to inflate RAM usage
    memory_hog = []

    step = 0
    log(f"Starting training loop: batch={batch_size} seq_len={seq_len} workers={workers}")
    for epoch in range(epochs):
        for batch in loader:
            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
            outputs = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            # Intentional memory pressure: keep a rolling buffer of detached
            # copies of hidden states so RSS keeps climbing past 16GB.
            memory_hog.append(outputs.logits.detach().clone())
            if len(memory_hog) > 40:
                memory_hog.pop(0)

            step += 1
            if step % 5 == 0:
                log(f"epoch={epoch} step={step} loss={loss.item():.4f}")
    log("llm mode: completed all epochs (unexpected — should be killed by timeout)")


# ----------------------------------------------------------------------
# CNN mode: ResNet-50 on CIFAR-100, big batch + many dataloader workers
# ----------------------------------------------------------------------
def run_cnn(batch_size=512, workers=16, epochs=1000):
    import torch
    import torch.nn as nn
    import torchvision
    from torchvision import transforms
    from torch.utils.data import DataLoader

    torch.set_num_threads(os.cpu_count())

    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),   # upscaling 32x32 -> 224x224 on purpose: CPU heavy
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
    ])

    log("Loading CIFAR-100 (torchvision, auto-downloads ~170MB)")
    dataset = torchvision.datasets.CIFAR100(root="./data", train=True, download=True, transform=transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                         num_workers=workers, pin_memory=False, persistent_workers=True,
                         prefetch_factor=4)

    log("Building ResNet-50 (randomly initialized, CPU)")
    model = torchvision.models.resnet50(weights=None, num_classes=100)
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    memory_hog = []
    step = 0
    log(f"Starting training loop: batch={batch_size} workers={workers}")
    for epoch in range(epochs):
        for images, labels in loader:
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            memory_hog.append(images.clone())
            if len(memory_hog) > 30:
                memory_hog.pop(0)

            step += 1
            if step % 5 == 0:
                log(f"epoch={epoch} step={step} loss={loss.item():.4f}")
    log("cnn mode: completed all epochs (unexpected — should be killed by timeout)")


# ----------------------------------------------------------------------
# Combined mode: LLM + CNN training as separate processes, same time
# ----------------------------------------------------------------------
def run_combined(workers=16):
    llm_workers = max(4, workers // 2)
    cnn_workers = max(4, workers // 2)

    p1 = mp.Process(target=run_llm, kwargs=dict(model_name="gpt2-medium",
                                                 batch_size=16, seq_len=512,
                                                 workers=llm_workers))
    p2 = mp.Process(target=run_cnn, kwargs=dict(batch_size=256, workers=cnn_workers))
    p1.start()
    p2.start()
    log(f"combined mode: llm pid={p1.pid}, cnn pid={p2.pid}")
    p1.join()
    p2.join()


# ----------------------------------------------------------------------
# Bigdata mode: heavy pandas/numpy transforms over a large real dataset
# ----------------------------------------------------------------------
def run_bigdata(workers=16, rounds=100000):
    import pandas as pd
    import numpy as np
    from datasets import load_dataset

    log("Loading Amazon Polarity (4M rows) — first run downloads several GB")
    ds = load_dataset("amazon_polarity", split="train")

    log("Converting to pandas (this alone will use multiple GB of RAM)")
    df = ds.to_pandas()

    memory_hog = []
    for r in range(rounds):
        # CPU/memory heavy string + numeric transforms, repeated
        df["title_len"] = df["title"].str.len()
        df["content_upper"] = df["content"].str.upper()
        df["content_words"] = df["content"].str.split().apply(len)
        df["score"] = np.log1p(df["title_len"] * df["content_words"] + 1)
        grouped = df.groupby("label")[["score", "content_words"]].agg(["mean", "std", "max"])
        merged = df.merge(df.sample(frac=0.3), on="label", suffixes=("", "_dup"))

        memory_hog.append(merged.copy())
        if len(memory_hog) > 5:
            memory_hog.pop(0)

        if r % 2 == 0:
            log(f"round={r} rows={len(df)} grouped_shape={grouped.shape} merged_shape={merged.shape}")


# ----------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["llm", "cnn", "combined", "bigdata"])
    parser.add_argument("--model", default="gpt2-medium")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=512)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()

    log(f"Mode={args.mode} | CPU cores available={os.cpu_count()} | CUDA disabled")

    if args.mode == "llm":
        run_llm(model_name=args.model, batch_size=args.batch_size,
                seq_len=args.seq_len, workers=args.workers)
    elif args.mode == "cnn":
        run_cnn(batch_size=args.batch_size, workers=args.workers)
    elif args.mode == "combined":
        run_combined(workers=args.workers)
    elif args.mode == "bigdata":
        run_bigdata(workers=args.workers)


if __name__ == "__main__":
    main()