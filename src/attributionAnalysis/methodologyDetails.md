# Bottleneck Attribution Model — Complete Research Notes and Methodology

## 1. Research Context

The project aims to diagnose and attribute system-performance bottlenecks using kernel-level telemetry collected through eBPF.

The broader architecture is:

```text
Application / Workload
        ↓
Linux Kernel
        ↓
eBPF telemetry collection
        ↓
Pinned BPF maps
        ↓
Userspace reader / feature extraction
        ↓
Bottleneck attribution model
        ↓
Bottleneck type / attribution
```

The realtime implementation does **not** continuously poll the kernel from userspace. Instead:

* eBPF probes collect kernel-level events.
* Metrics are aggregated into pinned BPF maps.
* The Python userspace component implements its own reader logic.
* The reader periodically extracts/flushes the map contents.
* The extracted telemetry is converted into model features.
* The trained ML model performs bottleneck prediction.

The model is therefore intended to operate on **snapshots/windows of aggregated kernel telemetry**, rather than raw continuous event streams.

---

# 2. Research Motivation for Bottleneck Attribution

The objective is not merely to detect whether the system is experiencing abnormal behaviour.

The more important problem is:

> **Given observed system telemetry, which underlying resource/stressor is responsible for the performance bottleneck?**

This is an attribution problem.

For example:

```text
Observed telemetry
       ↓
High scheduling activity
High runtime
High stalls
       ↓
Possible causes
       ↓
CPU bottleneck
```

The project considers multiple possible bottleneck dimensions:

1. CPU
2. Memory
3. Cache
4. I/O
5. Lock
6. Context Switch

A workload may activate more than one of these simultaneously.

Therefore, the attribution problem is naturally represented as a **multi-label classification problem**.

---

# 3. Important Reference Paper

The main methodological reference is:

**Fida et al., "Bottleneck Identification in Cloudified Mobile Networks Based on Distributed Telemetry," IEEE Transactions on Mobile Computing, 2024.**

The paper separates bottleneck identification into detection/attribution stages and uses an MLP for bottleneck classification.

The paper's attribution model is a **single multi-output MLP**, rather than training one independent model for each bottleneck profile.

Its architecture is approximately:

```text
76 input features
       ↓
64 neurons
       ↓
24 neurons
       ↓
16 neurons
       ↓
10 output neurons
```

The hidden layers use ReLU activations, while the output layer produces sigmoid probabilities.

The multiple sigmoid outputs allow the same sample to represent multiple simultaneous bottlenecks.

The paper explicitly evaluates both:

* single bottlenecks
* composite bottlenecks

and reports that composite attribution is more difficult because bottleneck effects can interact.

The paper also uses F1-based evaluation and evaluates composite predictions using metrics such as:

* Hamming Loss
* Exact Match Ratio

The paper reports that its MLP can perform inference in under approximately 2 ms once trained, while training is performed offline.

---

# 4. Initial Question: Is One RF per Bottleneck/Condition Correct?

The original implementation direction was based on Random Forest models.

The concern was that the implementation appeared to be moving toward a separate RF for each attribution/stressor, for example:

```text
Telemetry
   ├── RF_CPU
   ├── RF_Memory
   ├── RF_Cache
   ├── RF_IO
   ├── RF_Lock
   └── RF_ContextSwitch
```

or potentially even separate models for individual workload conditions such as:

```text
RF_cpu_extreme
RF_cpu_ultra
RF_mem_extreme
RF_mem_thrashing
RF_cpu_mem_extreme
...
```

This was determined to be less appropriate for the intended methodology.

The main issue is that the problem is not fundamentally:

> "Which independent binary classifier fires?"

It is:

> "Which bottleneck dimensions are active for this telemetry sample?"

Therefore, the model should learn a unified mapping:

```text
Telemetry
    ↓
ONE attribution model
    ↓
CPU       probability
Memory    probability
Cache     probability
IO        probability
Lock      probability
ContextSwitch probability
```

A sample can consequently be:

```text
CPU = 1
Memory = 1
Cache = 0
IO = 0
Lock = 0
ContextSwitch = 0
```

representing a composite:

```text
CPU + Memory
```

without requiring a separate `CPU+Memory` model.

---

# 5. Decision: Use MLP for the Main Attribution Model

The final direction was changed to an **MLP**, because this is the model used by the reference paper for bottleneck attribution.

For the project's data, the paper's 10 output neurons are adapted to 6 attribution dimensions.

The resulting architecture is:

```text
33 telemetry features
        ↓
      Dense
       64
        ↓
      ReLU
        ↓
      Dense
       24
        ↓
      ReLU
        ↓
      Dense
       16
        ↓
      ReLU
        ↓
      Dense
        6
        ↓
  CPU / Memory / Cache /
  IO / Lock / ContextSwitch
```

The model has **4,238 trainable parameters**.

The six outputs represent independent bottleneck dimensions.

---

# 6. Multi-Label Target Representation

The final target columns are:

```text
TARGET_COLS = [
    "CPU",
    "Memory",
    "Cache",
    "IO",
    "Lock",
    "ContextSwitch"
]
```

Each row therefore has a six-dimensional binary target:

```text
[CPU, Memory, Cache, IO, Lock, ContextSwitch]
```

Example:

```text
[1, 0, 0, 0, 0, 0]
```

means CPU bottleneck.

Example:

```text
[0, 1, 1, 0, 0, 0]
```

means:

```text
Memory + Cache
```

Example:

```text
[1, 1, 0, 1, 0, 0]
```

means:

```text
CPU + Memory + IO
```

This representation is important because composite bottlenecks are represented directly rather than being converted into unrelated multiclass categories.

---

# 7. Dataset and Session-Label Audit

The original dataset contains:

```text
777,578 total rows
50 unique session labels
```

A complete session-label attribution audit was performed.

The audit found:

```text
50 actual session labels
35 mapped/usable labels
15 unmapped labels
```

The 15 unmapped labels were:

```text
cpu_high
cpu_l3_cache_misses
cpu_low
cpu_medium
cpu_overloaded
ctx_flood
idle
io_async_saturation
io_low
lock_low
mem_high
mem_low
mem_swap_thrash
mem_swap_thrashing
mem_tlb_cache_miss
```

These labels were not silently assigned to an arbitrary bottleneck class.

Instead, they were treated as `UNKNOWN`/unmapped and removed from the supervised attribution dataset.

This was an important methodological decision because incorrectly assigning an unknown session to a bottleneck class would introduce label noise.

---

# 8. Stressor Distribution Before Filtering

The attribution audit showed that the six stressor dimensions occur with substantially different frequencies.

The approximate row-level distribution was:

| Stressor      |    Rows | Percentage |
| ------------- | ------: | ---------: |
| CPU           | 647,475 |     83.27% |
| Memory        | 624,638 |     80.33% |
| Cache         | 133,428 |     17.16% |
| IO            | 115,393 |     14.84% |
| Lock          |  90,622 |     11.65% |
| ContextSwitch | 121,529 |     15.63% |

This revealed a substantial **multi-label class imbalance**.

CPU and Memory are very common, while Cache, IO, Lock and ContextSwitch are substantially less frequent.

This imbalance later became an important part of the ML experiments.

---

# 9. Removal of UNKNOWN/Unmapped Labels

The 15 unmapped session labels were removed before model training.

Dataset size changed from:

```text
777,578 rows
```

to:

```text
694,819 rows
```

Therefore:

```text
82,759 rows
```

were removed.

The retained dataset contains:

```text
694,819 rows
35 valid session labels
6 attribution dimensions
```

Target validation showed:

```text
Rows with missing targets: 0
```

Therefore every retained row has a complete six-dimensional attribution vector.

A session-label consistency check also passed:

> Every retained session label has one consistent target vector.

This is important because the same session label should not map to different causal target vectors.

---

# 10. Final Attribution Feature Set

The original attribution dataset contained:

```text
41 candidate numeric features
```

An audit was performed to identify:

* identifiers
* metadata
* timestamps
* constant features
* unsuitable fields

The following features were removed:

```text
avg_runq_ratio
avg_rwsem_write_wait_ns
cpu
large_page_allocs
max_rwsem_write_wait_ns
pid
rwsem_write_contentions
timestamp_sec
total_free_bytes
```

The final attribution model uses:

```text
33 features
```

The important methodological principle is that identifiers and direct metadata should not be used as predictive features.

In particular:

```text
pid
cpu
timestamp_sec
```

were excluded.

The final feature set consists of kernel/system telemetry such as:

```text
ctx_switches
voluntary_switches
involuntary_switches
cpu_migrations
total_runtime_ns
stall_ns
avg_stall_ns
max_stall_ns
latency_count
minor_faults
major_faults
kmalloc_count
kfree_count
total_alloc_bytes
syscall_count
avg_syscall_latency_ns
max_syscall_latency_ns
read_count
write_count
read_bytes
write_bytes
mmap_count
futex_count
avg_futex_latency_ns
epoll_count
...
```

The complete feature list is stored in the notebook's `ATTR_FEATURES` variable and saved as:

```text
attribution_feature_cols.json
```

---

# 11. Feature Audit

A feature audit was performed before training.

For each feature, the following were examined:

* number of unique values
* percentage of zero values
* minimum
* maximum

This was used to identify:

* constant features
* near-constant features
* identifier-like variables
* unsuitable metadata
* potentially informative telemetry variables

This audit resulted in the final 33-feature input representation.

---

# 12. Feature–Stressor Association Audit

Before training the MLP, a univariate feature/stressor association analysis was performed.

For every target dimension, each feature was evaluated using ROC-AUC.

The ranking criterion was:

```text
absolute(AUC - 0.5)
```

A value near 0.5 indicates little univariate separation.

A value further from 0.5 indicates stronger univariate association.

For example, the CPU audit showed:

```text
avg_stall_ns          AUC = 0.7101
total_runtime_ns      AUC = 0.7005
stall_ns              AUC = 0.6598
max_stall_ns          AUC = 0.6594
involuntary_switches  AUC = 0.6222
ctx_switches          AUC = 0.6145
```

This provides an important sanity check that the telemetry contains signal associated with the attribution targets.

However, these univariate associations are **not the final attribution model**.

The MLP is expected to learn nonlinear combinations and interactions between features.

---

# 13. Train/Validation/Test Split

The split was changed to align more closely with the methodology of the reference paper.

The first split uses:

```text
60% development
40% final test
```

Then the development portion is split:

```text
70% training
30% validation
```

Therefore the final proportions are:

```text
Training   = 42%
Validation = 18%
Testing    = 40%
```

For the 694,819 retained rows, the actual sizes are:

```text
Training   = 291,823
Validation = 125,068
Testing    = 277,928
```

The random seed is:

```text
42
```

The validation and test datasets remain fixed across the different experimental pipelines.

This allows fair comparison of the four training strategies.

---

# 14. Target Distribution After Splitting

The target distributions are highly imbalanced.

## CPU

```text
Training:   271,822 / 291,823 = 93.15%
Validation: 116,491 / 125,068 = 93.14%
Testing:    259,162 / 277,928 = 93.25%
```

## Memory

```text
Training:   262,395 = 89.92%
Validation: 112,457 = 89.92%
Testing:    249,786 = 89.87%
```

## Cache

```text
Training:    55,893 = 19.15%
Validation:  24,057 = 19.24%
Testing:     53,478 = 19.24%
```

## IO

```text
Training:    48,423 = 16.59%
Validation:  20,834 = 16.66%
Testing:     46,136 = 16.60%
```

## Lock

```text
Training:    38,207 = 13.09%
Validation:  16,318 = 13.05%
Testing:     36,097 = 12.99%
```

## ContextSwitch

```text
Training:    51,053 = 17.49%
Validation:  21,792 = 17.42%
Testing:     48,684 = 17.52%
```

Therefore the main difficulty is not CPU/Memory detection but attribution of the relatively less frequent dimensions:

```text
Cache
IO
Lock
ContextSwitch
```

---

# 15. Feature Scaling

Because the final model is an MLP, feature scaling was introduced.

A `StandardScaler` is fitted **only on the training data**.

The transformation is then applied to:

```text
training
validation
test
```

using the training-derived statistics.

This prevents information from the validation/test sets from leaking into the feature normalization process.

The resulting tensors are:

```text
X_train: 291,823 × 33
Y_train: 291,823 × 6

X_val:   125,068 × 33
Y_val:   125,068 × 6

X_test:  277,928 × 33
Y_test:  277,928 × 6
```

---

# 16. PyTorch Dataset Configuration

The implementation uses PyTorch.

The device detected in the notebook is:

```text
CUDA
```

Therefore GPU acceleration is available.

Batch size:

```text
256
```

The training data are represented using PyTorch `TensorDataset` and `DataLoader`.

Training uses shuffled batches.

Validation and test loaders are not shuffled.

---

# 17. MLP Architecture

The final model is:

```text
Input: 33 features

Linear(33 → 64)
ReLU

Linear(64 → 24)
ReLU

Linear(24 → 16)
ReLU

Linear(16 → 6)
```

Output ordering:

```text
0 → CPU
1 → Memory
2 → Cache
3 → IO
4 → Lock
5 → ContextSwitch
```

Total trainable parameters:

```text
4,238
```

This is the project's adaptation of the reference paper's:

```text
76 → 64 → 24 → 16 → 10
```

architecture.

The only structural change is:

```text
76 input features → 33 input features
10 output bottleneck classes → 6 output dimensions
```

The hidden architecture remains:

```text
64 → 24 → 16
```

---

# 18. Multi-Label Output

The MLP outputs six logits.

Conceptually:

```text
z_CPU
z_Memory
z_Cache
z_IO
z_Lock
z_ContextSwitch
```

Each logit is converted to a probability using sigmoid:

```text
p = sigmoid(z)
```

Thus:

```text
0 ≤ p ≤ 1
```

The default classification threshold is:

```text
0.50
```

Therefore:

```text
p >= 0.50 → bottleneck present
p <  0.50 → bottleneck absent
```

This allows multiple outputs to be active simultaneously.

---

# 19. Initial Training Configuration

The original paper-aligned baseline configuration was:

```text
Loss: BCEWithLogitsLoss
Optimizer: Adam
Learning rate: 0.001
Epochs: 200
Batch size: 256
Threshold: 0.50
```

`BCEWithLogitsLoss` is used instead of explicitly placing sigmoid inside the network during training.

This combines the sigmoid operation and binary cross-entropy computation in a numerically stable implementation.

---

# 20. PyTorch/optree Environment Issue

When initially attempting to instantiate the Adam optimizer, the notebook encountered:

```text
ModuleNotFoundError: No module named 'optree._C'
```

The traceback showed that:

```text
torch.optim.Adam
    ↓
torch._dynamo
    ↓
optree
    ↓
optree._C
```

was failing because the compiled `optree._C` extension was unavailable.

The environment uses:

```text
Python 3.13
```

inside the project's:

```text
.venv
```

The proposed environment repair was:

```bash
python -m pip install --upgrade pip
python -m pip uninstall -y optree
python -m pip install optree
```

followed by verification:

```bash
python -c "import optree; print(optree.__version__); import optree._C; print('optree OK')"
```

The notebook was then restarted.

This issue is an **environment/dependency problem**, not a methodological problem with the MLP.

---

# 21. Class Imbalance Became the Main Experimental Issue

The target distribution showed severe imbalance:

```text
CPU            ≈ 93%
Memory         ≈ 90%

Cache          ≈ 19%
IO             ≈ 17%
Lock           ≈ 13%
ContextSwitch  ≈ 17%
```

A baseline BCE-trained MLP therefore tends to perform very well on common labels while performing poorly on minority bottlenecks.

Consequently, the notebook was extended to compare imbalance-handling strategies.

This is an experimental extension beyond the basic paper-aligned MLP.

---

# 22. Four MLP Training Pipelines

The final notebook compares four configurations.

## Pipeline 1 — baseline_original

```text
Original training data
        ↓
MLP
        ↓
BCEWithLogitsLoss
```

Configuration:

```python
use_smote = False
loss_type = "bce"
```

Purpose:

> Establish the basic paper-aligned MLP baseline.

---

## Pipeline 2 — mlsmote_only

```text
Training data
      ↓
MLSMOTE
      ↓
MLP
      ↓
BCEWithLogitsLoss
```

MLSMOTE is applied only to the training split.

The validation and test data remain untouched.

The notebook identified four tail labels:

```text
Cache
IO
Lock
ContextSwitch
```

It generated:

```text
20,000 synthetic training rows
```

in addition to the original:

```text
291,823
```

training rows.

Final training size:

```text
311,823
```

---

# 23. Pipeline 3 — focal_asl_only

This pipeline uses **Asymmetric Loss (ASL)** instead of ordinary BCE.

Configuration:

```text
MLSMOTE: No
Loss: ASL
```

Parameters used:

```text
gamma_neg = 4.0
gamma_pos = 1.0
clip = 0.05
```

The motivation is to make the loss less dominated by the very common negative/majority cases and improve learning for minority multi-label targets.

This is an experimental improvement and should be described as such rather than presented as part of the original paper's exact training procedure.

---

# 24. Pipeline 4 — hybrid_mlsmote_asl

This combines both approaches:

```text
Training data
      ↓
MLSMOTE
      ↓
MLP
      ↓
ASL
```

Configuration:

```text
MLSMOTE: Yes
ASL: Yes
```

Training rows:

```text
311,823
```

---

# 25. Early Stopping

The experimental pipeline uses:

```text
patience = 15 epochs
min_delta = 0.0001
```

The checkpoint criterion is:

```text
Validation Macro F1
```

rather than simply final validation loss.

This is important because the primary issue is imbalanced multi-label classification.

The best model is therefore the model with the highest validation Macro F1.

Each pipeline saves its best model checkpoint.

---

# 26. Final Experimental Results

The four pipelines produced the following test results.

| Pipeline      | Hamming Loss | Micro F1 |  Macro F1 | Weighted F1 | Exact Match |
| ------------- | -----------: | -------: | --------: | ----------: | ----------: |
| Baseline BCE  |        0.125 |    0.838 |     0.508 |       0.775 |       0.396 |
| MLSMOTE + BCE |        0.127 |    0.838 |     0.519 |       0.779 |       0.394 |
| ASL           |        0.281 |    0.743 | **0.619** |   **0.823** |       0.132 |
| MLSMOTE + ASL |        0.292 |    0.736 |     0.613 |       0.821 |       0.118 |

Validation Macro F1:

| Pipeline      | Best Validation Macro F1 |
| ------------- | -----------------------: |
| Baseline BCE  |                    0.509 |
| MLSMOTE + BCE |                    0.519 |
| ASL           |                **0.619** |
| MLSMOTE + ASL |                    0.612 |

---

# 27. Baseline MLP Results

The baseline BCE model achieved:

```text
Micro F1       = 0.838
Macro F1       = 0.508
Weighted F1    = 0.775
Hamming Loss   = 0.125
Exact Match    = 0.396
```

Per-class results:

| Bottleneck    | Precision | Recall |   F1 |
| ------------- | --------: | -----: | ---: |
| CPU           |      0.94 |   1.00 | 0.97 |
| Memory        |      0.92 |   0.99 | 0.95 |
| Cache         |      0.76 |   0.10 | 0.17 |
| IO            |      0.63 |   0.26 | 0.37 |
| Lock          |      0.80 |   0.34 | 0.48 |
| ContextSwitch |      0.58 |   0.06 | 0.11 |

This is a clear example of the class-imbalance problem.

The model is excellent at:

```text
CPU
Memory
```

but has poor recall for:

```text
Cache
IO
Lock
ContextSwitch
```

Therefore the apparently strong Micro F1 of `0.838` should **not** be interpreted as uniformly strong bottleneck attribution.

Macro F1 is much more revealing:

```text
Macro F1 = 0.508
```

---

# 28. MLSMOTE + BCE Results

MLSMOTE increased the training set from:

```text
291,823
```

to:

```text
311,823
```

The test results were:

```text
Hamming Loss   = 0.127
Micro F1       = 0.838
Macro F1       = 0.519
Weighted F1    = 0.779
Exact Match    = 0.394
```

Per-class results:

| Bottleneck    | Precision | Recall |   F1 |
| ------------- | --------: | -----: | ---: |
| CPU           |      0.94 |   1.00 | 0.97 |
| Memory        |      0.91 |   1.00 | 0.95 |
| Cache         |      0.70 |   0.11 | 0.19 |
| IO            |      0.54 |   0.39 | 0.45 |
| Lock          |      0.69 |   0.39 | 0.50 |
| ContextSwitch |      0.82 |   0.03 | 0.06 |

Compared with baseline:

```text
Macro F1:
0.508 → 0.519
```

So MLSMOTE + BCE provided only a modest overall improvement.

It improved some minority labels, particularly:

```text
IO
Lock
Cache
```

but ContextSwitch recall remained extremely low.

---

# 29. ASL Results

The ASL model produced:

```text
Hamming Loss   = 0.281
Micro F1       = 0.743
Macro F1       = 0.619
Weighted F1    = 0.823
Exact Match    = 0.132
```

Per-class:

| Bottleneck    | Precision | Recall |   F1 |
| ------------- | --------: | -----: | ---: |
| CPU           |      0.93 |   1.00 | 0.97 |
| Memory        |      0.91 |   1.00 | 0.95 |
| Cache         |      0.28 |   0.96 | 0.44 |
| IO            |      0.34 |   0.90 | 0.49 |
| Lock          |      0.34 |   0.76 | 0.47 |
| ContextSwitch |      0.25 |   0.97 | 0.39 |

This is a very important result.

ASL dramatically increased minority-label recall:

```text
Cache       0.10 → 0.96
IO          0.26 → 0.90
Lock        0.34 → 0.76
Context     0.06 → 0.97
```

However, precision dropped substantially.

Therefore the model became much more aggressive in predicting minority bottlenecks.

This explains the increase in:

```text
Hamming Loss
```

and the reduction in:

```text
Exact Match Ratio
```

while simultaneously increasing:

```text
Macro F1
Weighted F1
```

The most important result is therefore not that ASL is universally "better", but that it changes the precision/recall balance substantially in favour of minority-bottleneck recall.

---

# 30. Hybrid MLSMOTE + ASL Results

The hybrid model achieved:

```text
Hamming Loss   = 0.292
Micro F1       = 0.736
Macro F1       = 0.613
Weighted F1    = 0.821
Exact Match    = 0.118
```

Per-class:

| Bottleneck    | Precision | Recall |   F1 |
| ------------- | --------: | -----: | ---: |
| CPU           |      0.93 |   1.00 | 0.97 |
| Memory        |      0.91 |   1.00 | 0.95 |
| Cache         |      0.28 |   0.98 | 0.43 |
| IO            |      0.35 |   0.88 | 0.50 |
| Lock          |      0.30 |   0.82 | 0.44 |
| ContextSwitch |      0.25 |   0.97 | 0.39 |

The hybrid approach did **not** outperform ASL alone on Macro F1:

```text
ASL             = 0.619
MLSMOTE + ASL   = 0.613
```

It also produced a worse Exact Match Ratio:

```text
ASL             = 0.132
MLSMOTE + ASL   = 0.118
```

Therefore, based on the current experiment, there is no evidence that combining MLSMOTE with ASL is beneficial.

---

# 31. Main Experimental Finding So Far

The most important result is that **the choice of evaluation metric strongly changes the apparent conclusion**.

Baseline:

```text
Micro F1 = 0.838
Macro F1 = 0.508
```

ASL:

```text
Micro F1 = 0.743
Macro F1 = 0.619
```

If only Micro F1 were considered, the baseline would appear substantially better.

However, Macro F1 shows that ASL provides substantially more balanced performance across the six bottleneck dimensions.

Therefore:

> **Micro F1 alone is insufficient for evaluating this bottleneck attribution problem because CPU and Memory dominate the label distribution.**

Macro F1 and per-bottleneck F1 are necessary.

---

# 32. Interpretation of Exact Match Ratio

Exact Match Ratio measures whether **every predicted bottleneck dimension for a sample is correct simultaneously**.

For the current models:

```text
Baseline       = 0.396
MLSMOTE+BCE    = 0.394
ASL            = 0.132
Hybrid         = 0.118
```

The ASL models have much lower exact-match performance because they produce many more minority-bottleneck predictions.

This does not necessarily mean that the model has become useless.

It means that the model is making a different trade-off:

```text
Baseline:
high precision / low minority recall

ASL:
high minority recall / lower precision
```

This trade-off must be discussed explicitly.

---

# 33. Important Methodological Conclusion

The project should not simply select a model based on the largest overall accuracy or Micro F1.

The attribution task is imbalanced and multi-label.

The evaluation should therefore consider:

```text
Per-label Precision
Per-label Recall
Per-label F1

Macro F1
Micro F1
Weighted F1

Hamming Loss
Exact Match Ratio
```

For composite bottlenecks, Exact Match Ratio is particularly strict because the entire attribution vector must be correct.

Hamming Loss is more forgiving because it evaluates each label independently.

This follows the general evaluation logic used by the reference paper for composite bottleneck classification.

---

# 34. Current Model Selection Status

At the current stage:

### If the priority is balanced attribution across bottleneck types:

```text
ASL
```

is currently the strongest candidate because:

```text
Macro F1 = 0.619
```

which is higher than:

```text
Baseline = 0.508
MLSMOTE+BCE = 0.519
Hybrid = 0.613
```

### If the priority is Micro F1 / conservative predictions:

```text
Baseline BCE
```

remains stronger:

```text
Micro F1 = 0.838
```

### Current empirical ranking by Macro F1:

```text
1. ASL                0.619
2. MLSMOTE + ASL      0.613
3. MLSMOTE + BCE      0.519
4. Baseline BCE       0.508
```

The current evidence therefore favours **ASL without MLSMOTE** as the best balanced-attribution candidate.

However, this should not yet be treated as the final model-selection conclusion until threshold tuning and additional validation are performed.

---

# 35. Important Distinction: Paper Method vs Project Extension

The methodology should clearly distinguish between what is inherited from the reference paper and what is an adaptation/extension.

## Directly inspired by the paper

```text
MLP-based bottleneck attribution
Multi-label output representation
ReLU hidden layers
Sigmoid output probabilities
64 → 24 → 16 hidden architecture
0.5 prediction threshold
Composite bottleneck evaluation
F1-based evaluation
Hamming Loss
Exact Match Ratio
```

The paper itself uses a single MLP capable of handling single and composite bottlenecks.

## Project-specific adaptation

```text
33 input features instead of 76
6 bottleneck dimensions instead of 10
Linux/eBPF telemetry instead of mobile-network telemetry
Kernel-level performance metrics instead of distributed mobile-network telemetry
PyTorch implementation adapted to the project's dataset
```

## Experimental extensions

```text
MLSMOTE
Asymmetric Loss
Hybrid MLSMOTE + ASL
Early stopping on validation Macro F1
StandardScaler preprocessing
```

These should be presented as **extensions for handling the project's dataset characteristics**, not as components of the reference paper unless directly supported by the paper.

---

# 36. Why the Single-Model Formulation Is Better for This Research

The final formulation is:

```text
33 kernel telemetry features
              ↓
        Single MLP
              ↓
 ┌──────┬───────┬───────┬────┬──────┬──────────────┐
 │ CPU  │ Memory│ Cache │ IO │ Lock │ ContextSwitch│
 └──────┴───────┴───────┴────┴──────┴──────────────┘
              ↓
       multi-label vector
```

This has several methodological advantages over separate models:

1. One model represents the overall attribution problem.
2. Composite bottlenecks are naturally represented.
3. The same telemetry representation is shared across all attribution dimensions.
4. The architecture is directly inspired by the reference paper.
5. The model can produce multiple simultaneous bottleneck predictions.
6. The approach avoids creating separate models for every possible bottleneck combination.

---

# 37. Why Separate Models for Every Composite Condition Are Avoided

Suppose there are six possible bottleneck dimensions.

Treating every possible combination as a separate class could result in up to:

```text
2^6 = 64
```

possible combinations.

This is undesirable because the model would have to learn:

```text
CPU
Memory
Cache
...
CPU + Memory
CPU + Cache
...
CPU + Memory + Cache
...
```

as independent classes.

Instead, the six-dimensional representation is:

```text
[CPU, Memory, Cache, IO, Lock, ContextSwitch]
```

and composite combinations emerge naturally from the active labels.

This is much more scalable.

---

# 38. Current Notebook Structure

The final notebook has evolved into approximately the following sequence:

```text
Step 1
Load / inspect dataset

Step 1
Inspect label-associated metadata

Step 1
Identify session/run identifiers

Step 2
Audit all session labels for attribution mapping

Step 3
Remove UNKNOWN mappings
Construct attribution targets

Step 4
Prepare multi-label attribution dataset

Step 5
Feature audit

Step 6
Clean attribution feature matrix

Step 7
Feature/stressor association audit

Step 8
Train / validation / test split

Step 9
Feature scaling + PyTorch tensors

Step 10
Define MLP architecture

Step 11
Configure MLP training

Step 11.1
MLSMOTE utility

Step 11.2
Focal/ASL loss utility

Step 11.3
Isolated pipeline runner

Step 11.4
Execute four experimental pipelines

Step 11.5
Global comparative summary
```

The original single-pipeline cells for:

```text
training
prediction
evaluation
composite evaluation
```

were superseded by the experimental pipeline harness.

---

# 39. Artifact Saving

The notebook saves model-comparison artifacts under:

```text
pipeline_comparison_artifacts/
```

The artifacts include:

```text
baseline_original_best.pt
mlsmote_only_best.pt
focal_asl_only_best.pt
hybrid_mlsmote_asl_best.pt

pipeline_comparison_summary.csv

attribution_feature_cols.json
attribution_target_cols.json
feature_scaler.pkl

baseline_original_classification_report.txt
mlsmote_only_classification_report.txt
focal_asl_only_classification_report.txt
hybrid_mlsmote_asl_classification_report.txt
```

Each `.pt` file contains the best MLP weights for the corresponding experimental configuration.

---

# 40. Current Research Interpretation

The current results indicate that the telemetry contains sufficient signal for strong attribution of the dominant bottlenecks:

```text
CPU
Memory
```

but attribution of minority bottlenecks is substantially harder under ordinary BCE training.

The baseline model effectively learns:

```text
CPU → strong
Memory → strong

Cache → weak
IO → weak
Lock → moderate
ContextSwitch → very weak
```

The ASL experiment demonstrates that the MLP can learn substantially higher recall for the minority bottlenecks.

Therefore, the problem is not simply:

> "Is the MLP powerful enough?"

The more significant issue is:

> **How should the multi-label training objective handle the highly imbalanced attribution targets?**

This is now a central experimental question.

---

# 41. Important Caveat About the Current Results

The current ASL result should **not yet be declared the final optimal model**.

ASL was trained with the default threshold:

```text
0.50
```

The results show a very strong shift toward recall for minority classes and a corresponding drop in precision.

Therefore, a future step should investigate **per-label decision thresholds** rather than assuming 0.50 is optimal for all six outputs.

For example, the optimal threshold for:

```text
CPU
```

may differ substantially from:

```text
ContextSwitch
```

because their class frequencies and probability distributions differ.

This is particularly important before final model selection.

---

# 42. Future Experimental Direction

The next methodological steps should be:

## A. Threshold analysis

Evaluate thresholds such as:

```text
0.10
0.15
0.20
...
0.90
```

for each label.

Determine thresholds using the **validation set only**.

Do not tune thresholds on the final test set.

Then freeze the selected thresholds and evaluate once on the test set.

---

## B. Per-label precision–recall analysis

Generate precision/recall curves for:

```text
CPU
Memory
Cache
IO
Lock
ContextSwitch
```

This will show where the optimal operating point lies for each bottleneck.

---

## C. Composite bottleneck analysis

Separate samples into:

```text
single bottleneck
composite bottleneck
```

and evaluate them independently.

This will answer whether the model works primarily because it recognizes simple single-resource stress or whether it genuinely handles simultaneous bottlenecks.

---

## D. Confusion/attribution analysis

For composite samples, determine which labels are:

```text
correctly detected
missed
incorrectly added
```

This is more informative than only reporting Exact Match Ratio.

---

## E. SHAP / model interpretability

The previous RF implementation used:

```python
shap.TreeExplainer
```

because Random Forest is tree-based.

That approach should **not** be carried over to the MLP.

The MLP requires a neural-network-compatible explanation method such as an appropriate SHAP explainer.

The goal is to determine:

```text
Which kernel telemetry features
contribute most strongly to
CPU attribution?

Which contribute to Memory?

Which contribute to Cache?

...
```

This is particularly important for the research objective because attribution should be interpretable rather than functioning only as a black-box classifier.

---

# 43. Important Research Positioning

The intended contribution is not simply:

> "We trained an MLP to classify six classes."

The stronger formulation is:

> **A unified multi-label bottleneck attribution framework maps kernel-level telemetry to multiple simultaneous system bottleneck dimensions, enabling attribution of both individual and composite resource bottlenecks from eBPF-derived performance telemetry.**

The reference paper demonstrates the usefulness of a unified MLP for bottleneck attribution in a different domain. The project adapts this formulation to kernel-level/eBPF telemetry and the project's specific performance-bottleneck use case.

---

# 44. Current Methodology — Compact Formal Version

The current attribution methodology can be represented as:

```text
Raw kernel telemetry
        ↓
Session-label attribution mapping
        ↓
Remove unmapped/UNKNOWN sessions
        ↓
Construct six-dimensional binary target
        ↓
Feature audit
        ↓
Remove identifiers / constants / metadata
        ↓
33-feature attribution matrix
        ↓
42/18/40 train-validation-test split
        ↓
StandardScaler fitted on training set
        ↓
MLP:
33 → 64 → 24 → 16 → 6
        ↓
Multi-label loss
        ↓
Sigmoid probabilities
        ↓
Thresholding
        ↓
Six-dimensional bottleneck attribution
```

The experimental loss variants are:

```text
BCE
BCE + MLSMOTE
ASL
ASL + MLSMOTE
```

---

# 45. Current Best Evidence

Current test results:

```text
                Hamming   Micro    Macro    Weighted   Exact
Baseline BCE     0.125    0.838    0.508     0.775     0.396
MLSMOTE+BCE      0.127    0.838    0.519     0.779     0.394
ASL              0.281    0.743    0.619     0.823     0.132
MLSMOTE+ASL      0.292    0.736    0.613     0.821     0.118
```

Current interpretation:

```text
Best Macro F1:
ASL = 0.619

Best Micro F1:
Baseline / MLSMOTE+BCE = 0.838

Best Weighted F1:
ASL = 0.823

Best Exact Match:
Baseline = 0.396

Lowest Hamming Loss:
Baseline = 0.125
```

Therefore no single metric declares one pipeline universally superior.

The current evidence supports:

> **ASL provides the most balanced per-bottleneck attribution according to Macro F1, while BCE provides better conservative/global agreement according to Micro F1, Hamming Loss and Exact Match Ratio.**

This is an important result rather than a contradiction.

---

# 46. Final State of the Research Implementation

At this point, the research implementation has moved from:

```text
Separate RF-based attribution
```

to:

```text
Unified MLP-based multi-label attribution
```

with:

```text
33 input telemetry features
6 bottleneck outputs
64 → 24 → 16 hidden architecture
BCE / ASL experimental losses
optional MLSMOTE
GPU training
validation-based model checkpointing
multi-label evaluation
composite-bottleneck evaluation
```

The main unresolved research/implementation questions are now:

1. What thresholds should be used for each bottleneck?
2. Should ASL or BCE be selected as the final training objective?
3. Does threshold optimization improve ASL's Exact Match Ratio without destroying minority recall?
4. How well does the model attribute genuinely composite workloads?
5. Which telemetry features drive each bottleneck prediction?
6. Does the model generalize to unseen workload severity?
7. Does it generalize to unseen workload combinations?
8. What is the inference latency of the final trained model in the intended realtime pipeline?
9. How stable are the results across different train/test random seeds?
10. Can the final model be integrated directly with the existing pinned-BPF-map userspace reader?

---

# 47. Key Principle to Preserve for Future Work

The most important methodological decision made so far is:

> **Do not create a separate ML model for every bottleneck combination or workload condition.**

Instead:

```text
One unified attribution model
        ↓
multiple bottleneck outputs
```

with the target represented as:

```text
[CPU, Memory, Cache, IO, Lock, ContextSwitch]
```

This keeps the implementation aligned with the reference paper's multi-label bottleneck-attribution formulation while adapting it to the project's kernel/eBPF telemetry domain.

Severity/condition should be treated as a separate experimental dimension where appropriate rather than automatically creating unrelated bottleneck classes.

For example:

```text
CPU + low severity
CPU + medium severity
CPU + extreme severity
```

should conceptually be understood as:

```text
Bottleneck type = CPU
Severity = low / medium / extreme
```

rather than necessarily being three unrelated bottleneck classes.

The reference paper itself observed confusion between bottleneck types that share the same cause but differ in severity, reinforcing the importance of distinguishing **bottleneck identity** from **bottleneck intensity**.
