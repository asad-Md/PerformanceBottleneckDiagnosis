#!/usr/bin/env bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "======================================="
echo "[SESSION] Starting Sequential ML Stress Tests"
echo "======================================="

echo "[+] Running LLM Workload (GPT-2 Medium)..."
python3 ml_stress.py --mode llm --model gpt2-medium --batch-size 1 --seq-len 1 --workers 1

echo "[+] Running CNN Workload (ResNet-50 / CIFAR-100)..."
python3 ml_stress.py --mode cnn --batch-size 1 --workers 1

echo "[+] Running Combined Workload (LLM + CNN)..."
python3 ml_stress.py --mode combined --model gpt2-medium --batch-size 1 --seq-len 1 --workers 1

# echo "[+] Running Big Data Workload (Amazon Polarity)..."
# python3 ml_stress.py --mode bigdata --workers 1

echo "======================================="
echo "[SESSION] All stress tests completed successfully!"
echo "======================================="