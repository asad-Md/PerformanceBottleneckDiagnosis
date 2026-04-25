# eBPF ML Bottleneck Diagnosis — Data Collection & Training Plan

---

## 0. Where You Are Right Now

Your 134s idle session produced **5,461 rows**. Each row is one (pid, cpu)
pair that was active during the session — all cumulative counters, one snapshot
at session end. This is the right shape for the dataset but it is **not yet
time-series data**; it is a cross-sectional snapshot. The plan below explains
exactly how to handle this, how many sessions to run, and what to do in
Jupyter.

---

## 1. The Core Problem With the Current Schema

Every row is a *lifetime aggregate* for one (pid, cpu) pair across the whole
session. If you run a 134s session, a process that was busy only in second 10
and a process busy only in second 120 look the same in the CSV — you cannot
tell when they were stressed.

You have two ways to fix this, and they are not mutually exclusive:

**Option A — Many short sessions (recommended to start)**
Run 60–90 second sessions, each with one clean stressor. The entire session
represents one condition. Label the whole CSV with `--label cpu_high` etc.
Each row already gets that label. In Jupyter you train on session-level rows.
This is simpler and is what you should do for the first 3–4 weeks.

**Option B — Periodic snapshots (for time-series models later)**
Modify reader.c to be called in a loop every N seconds (without unpinning the
maps), appending rows each time with a monotonic snapshot timestamp. This
creates true time-series data. The `timestamp_ns` column then becomes
meaningful for bucketing. This is the upgrade path after you have a working
classifier.

**Start with Option A. This README covers Option A end-to-end.**

---

## 2. Class Definitions

You are building a 4-class classifier. The primary signal is `avg_stall_ns`
(average runqueue latency per context switch for that process).

```
Class 0 — Normal      avg_stall_ns <  2,000,000 ns   ( < 2 ms )
Class 1 — Low         avg_stall_ns <  10,000,000 ns  ( < 10 ms )
Class 2 — Medium      avg_stall_ns <  50,000,000 ns  ( < 50 ms )
Class 3 — High        avg_stall_ns >= 50,000,000 ns  ( >= 50 ms )
```

Secondary confirmation signals (used for validation, not primary labeling):

- `avg_runq_ratio > 100`  → system is oversubscribed (confirms class 2–3)
- `involuntary_switches / ctx_switches > 0.7` → CPU-bound victim
- `voluntary_switches / ctx_switches > 0.7`   → I/O or lock-bound victim
- `mutex_contentions > 0 AND avg_mutex_wait_ns > 5,000,000` → lock-bound

The `session_label` column (from `--label`) is a *session descriptor*, not the
ML target. The ML target `y` is derived in Jupyter from `avg_stall_ns`.

---

## 3. Bottleneck Types to Test

Test these in order, from easiest to produce to hardest. Each type exercises a
different kernel path and produces a distinguishable fingerprint in the CSV.

### 3.1 CPU Saturation (start here)

**What it stresses:** Runqueue depth. More runnable threads than CPU cores.
**Signal:** High `involuntary_switches`, high `avg_runq_ratio`, high
`avg_stall_ns` on victim processes (everything else on the box).

**Incremental stressor ladder:**

```bash
# Step 1 — baseline, no stressor, system idle
# Run for 90s, label: idle_baseline
# Expected: class 0 for nearly all rows

# Step 2 — mild CPU (4 threads on 16-core box = 25% utilisation)
stress-ng --cpu 4 --timeout 90s
# Label: cpu_low
# Expected: class 0–1 mix

# Step 3 — moderate CPU (8 threads = 50%)
stress-ng --cpu 8 --timeout 90s
# Label: cpu_medium
# Expected: class 1–2 mix; avg_runq_ratio ~50

# Step 4 — saturation (16 threads = 100%, matches your thread count)
stress-ng --cpu 16 --timeout 90s
# Label: cpu_high
# Expected: class 2–3; avg_runq_ratio ~100

# Step 5 — oversubscription (32 threads = 200%)
stress-ng --cpu 32 --timeout 90s
# Label: cpu_overloaded
# Expected: class 3 dominant; avg_runq_ratio > 150;
#           max_stall_ns in hundreds of ms
```

**How to verify you have real signal before moving on:**
After each session, in Jupyter run:
```python
print(df[df.session_label == 'cpu_high']['avg_stall_ns'].describe())
# The 75th percentile should be above 10_000_000 for cpu_high
# If it is still under 2_000_000 the stressor is not landing — increase threads
```

### 3.2 Memory Pressure

**What it stresses:** Page reclaim, kmalloc fragmentation, TLB pressure.
**Signal:** High `minor_faults`, `kmalloc_count`, `total_alloc_bytes`.
Stall may be moderate (class 1–2) because the CPU isn't always busy, but
memory bandwidth is saturated.

```bash
# Step 1 — small allocation flood (stays in L3 cache)
stress-ng --vm 2 --vm-bytes 512M --timeout 90s
# Label: mem_low

# Step 2 — fill physical RAM (forces page reclaim)
# On your 4800HS with 16GB: use ~80% of RAM
stress-ng --vm 4 --vm-bytes 3G --timeout 90s
# Label: mem_high

# Step 3 — combined with CPU to force worst-case stall
stress-ng --cpu 8 --vm 4 --vm-bytes 3G --timeout 90s
# Label: mem_cpu_combined
```

**Distinctive fingerprint vs CPU-only:**
Memory-bound sessions show high `minor_faults` and `avg_syscall_latency_ns`
but moderate `involuntary_switches`. CPU-only shows the reverse.

### 3.3 I/O Saturation (syscall-bound)

**What it stresses:** Block I/O path, syscall latency, voluntary yield rate.
**Signal:** High `voluntary_switches` (process yields waiting for I/O), high
`avg_syscall_latency_ns` on read/write, high `read_bytes`/`write_bytes`.

```bash
# IMPORTANT: target /tmp or a tmpfs first to test without killing your SSD
# Step 1 — sequential write flood to tmpfs
stress-ng --io 4 --timeout 90s
# Label: io_low

# Step 2 — mixed read/write, direct I/O (bypasses page cache, stresses device)
fio --name=test --rw=randrw --bs=4k --size=1G \
    --numjobs=4 --iodepth=16 --direct=1 \
    --runtime=90 --time_based --filename=/tmp/fio_test
# Label: io_high

# Step 3 — combined with CPU
stress-ng --cpu 8 --io 4 --timeout 90s
# Label: io_cpu_combined
```

**Do NOT run disk I/O stress against your NVMe at full throttle in early
testing.** Use tmpfs (`/tmp` or `mount -t tmpfs tmpfs /mnt/test`). You can
test real disk after the model works, with shorter sessions.

### 3.4 Lock Contention

**What it stresses:** Mutex and rwsem wait queues.
**Signal:** High `mutex_contentions`, high `avg_mutex_wait_ns`, moderate
`voluntary_switches` (threads sleep waiting for lock).

Lock contention is the hardest to manufacture cleanly. The best approach:

```bash
# stress-ng has a mutex and futex stressor
stress-ng --mutex 8 --timeout 90s
# Label: lock_low

stress-ng --mutex 16 --futex 8 --timeout 90s
# Label: lock_high
```

Note: `lock_map` requires `CONFIG_LOCKDEP=y` in your kernel. Fedora ships
kernels with lockdep disabled in production configs. Run:
```bash
grep CONFIG_LOCKDEP /boot/config-$(uname -r)
```
If it says `# CONFIG_LOCKDEP is not set`, your lock_map will always be empty.
This is fine — skip the lock stressor for now and treat `lock_*` columns as
always-zero. The model will learn to ignore them via feature importance.

### 3.5 Mixed / Realistic Workload

After you have clean single-stressor data:

```bash
# Simulate a "production server under load" — mixed everything
stress-ng --cpu 8 --vm 2 --vm-bytes 1G --io 2 --mutex 4 --timeout 120s
# Label: mixed_load

# Simulate compile-time pressure (close to real development workload)
make -j$(nproc) some_large_project
# Label: compile_load
```

---

## 4. Session Protocol (Do This Exactly)

For every session:

```bash
# Terminal 1: start pinner
sudo ./pinner --obj perf_monitor.bpf.o

# Terminal 2: wait 10 seconds for warmup, then start stressor
sleep 10
stress-ng --cpu 16 --timeout 70s   # 70s stressor inside 90s window

# After stressor finishes, wait 5 more seconds, then Ctrl+C pinner
# Total pinner runtime: ~90s

# Terminal 1: Ctrl+C  →  pinner writes end_ns to meta file

# Immediately (within 60s, before kernel reclaims pinned maps):
sudo ./reader --label cpu_high --skip-start 10 --skip-end 5 --append
```

Using `--append` lets you accumulate all sessions into one CSV. Keep a
session log file manually:

```
sessions.log
────────────
2024-01-15 14:32  idle_baseline    134s  5461 rows
2024-01-15 15:01  cpu_high          90s  ~5000 rows
...
```

---

## 5. How Many Rows Do You Need?

### Minimum viable dataset

```
Per class (0–3):   at least 5,000 rows
Total minimum:     20,000 rows
Recommended:       50,000+ rows for reliable generalisation
```

With 5,461 rows per 134s idle session, you are getting roughly **40 rows per
second** of collection. A 90s session gives ~3,600 rows.

**Practical target: 12–15 sessions = ~50,000–70,000 rows.**

```
idle_baseline   ×3 sessions   → ~16,000 rows  (class 0 dominated)
cpu_low         ×2 sessions   → ~10,000 rows  (class 0–1 mix)
cpu_medium      ×2 sessions   → ~10,000 rows  (class 1–2 mix)
cpu_high        ×2 sessions   → ~10,000 rows  (class 2–3)
cpu_overloaded  ×2 sessions   → ~10,000 rows  (class 3 dominated)
io_high         ×2 sessions   → ~10,000 rows  (class 1–2, different features)
mem_high        ×2 sessions   → ~10,000 rows  (class 1–2, memory features active)
mixed_load      ×2 sessions   → ~10,000 rows  (mixed classes)
────────────────────────────────────────────
Total                          ~86,000 rows
```

The class distribution will be naturally imbalanced (lots of class 0 and 1,
fewer class 3). Handle this in Jupyter with `class_weight='balanced'` on the
classifier, or SMOTE oversampling. Do not try to fix it during collection.

---

## 6. Jupyter Preprocessing Pipeline — Exact Steps

### 6.1 Load and Inspect

```python
import pandas as pd
import numpy as np

df = pd.read_csv('perf_metrics.csv')
print(df.shape)           # expect (N, 42)
print(df.dtypes)
print(df['session_label'].value_counts())
```

### 6.2 Drop Useless Rows

```python
# Drop rows where the process had zero context switches
# (process was registered in the map but never actually ran)
df = df[df['ctx_switches'] > 0].copy()

# Drop kernel threads with pid=0 (idle process — always class 0, adds noise)
df = df[df['pid'] > 0].copy()

# Drop rows with zero latency samples (no wakeup events recorded)
# These processes ran but were never woken up (e.g. initial exec)
# They can't be reliably classified
df = df[df['latency_count'] > 0].copy()

print(f"After cleaning: {len(df)} rows")
```

### 6.3 Derive the ML Target Label

```python
# Primary label from avg_stall_ns (runqueue latency)
bins   = [0, 2e6, 10e6, 50e6, float('inf')]
labels = [0,   1,    2,    3]
df['y'] = pd.cut(df['avg_stall_ns'], bins=bins, labels=labels).astype(int)

print(df['y'].value_counts().sort_index())
```

**Verify the distribution makes sense:**
- `idle_baseline` sessions should be 90%+ class 0
- `cpu_high` sessions should push the stressed processes (stress-ng workers)
  to class 2–3; kernel threads and idle processes still appear as class 0

```python
# Cross-check: session label vs derived ML class
print(pd.crosstab(df['session_label'], df['y']))
```

If `cpu_high` shows almost no class 2–3, your stressor is not causing enough
contention. Re-run with more threads.

### 6.4 Feature Engineering

```python
# Ratio features (more informative than raw counts)
df['involuntary_ratio'] = (
    df['involuntary_switches'] / df['ctx_switches'].clip(lower=1)
)
df['voluntary_ratio'] = (
    df['voluntary_switches'] / df['ctx_switches'].clip(lower=1)
)
df['read_write_ratio'] = (
    df['read_bytes'] / (df['write_bytes'] + 1)   # +1 avoids div-by-zero
)
df['alloc_pressure'] = (
    df['total_alloc_bytes'] / df['ctx_switches'].clip(lower=1)
)
df['runtime_per_switch'] = (
    df['total_runtime_ns'] / df['ctx_switches'].clip(lower=1)
)
df['lock_pressure'] = (
    df['mutex_contentions'] + df['rwsem_read_contentions'] +
    df['rwsem_write_contentions']
)

# Log-scale the highly skewed latency columns
# (tree models handle raw values fine, but linear models need this)
for col in ['stall_ns', 'avg_stall_ns', 'max_stall_ns',
            'total_runtime_ns', 'avg_syscall_latency_ns']:
    df[f'log_{col}'] = np.log1p(df[col])
```

### 6.5 Feature Selection — What to Keep

```python
# Columns to DROP before training:
drop_cols = [
    'timestamp_ns',    # raw timestamp, not a feature
    'pid',             # process identity, not a pattern
    'cpu',             # CPU affinity, not a bottleneck signal
    'comm',            # process name — encode separately if you want it
    'session_label',   # text descriptor, not the ML target
    'y',               # this is your target, not a feature
]

# Optional: encode comm as a categorical if you want process-type signal
# df['comm_enc'] = df['comm'].astype('category').cat.codes

feature_cols = [c for c in df.columns if c not in drop_cols]
X = df[feature_cols]
y = df['y']
```

### 6.6 Train/Test Split

```python
from sklearn.model_selection import train_test_split

# IMPORTANT: split by session, not by row
# If you split by row, train and test will have rows from the same session
# and the model will memorise session-level statistics, not generalise.

# Assign each row a session index based on session_label
df['session_id'] = df['session_label'].astype('category').cat.codes

from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df['session_id']))

X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
```

### 6.7 Model Training

Start with Random Forest. It handles class imbalance, is robust to unscaled
features, and gives feature importance for free.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

clf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    class_weight='balanced',   # handles class imbalance
    n_jobs=-1,
    random_state=42
)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred,
      target_names=['Normal', 'Low', 'Medium', 'High']))
print(confusion_matrix(y_test, y_pred))
```

**Expected first-pass accuracy with 50K+ rows:**
- Normal (class 0) vs High (class 3): >95% precision/recall
- Normal vs Low (class 1): harder, expect 75–85% initially
- The model will confuse class 1 and class 2 most often — this is expected
  because the boundary (10ms) is arbitrary

Then try XGBoost or LightGBM for a few percent improvement:

```python
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

weights = compute_sample_weight('balanced', y_train)
xgb = XGBClassifier(n_estimators=500, max_depth=6,
                    learning_rate=0.05, n_jobs=-1, random_state=42)
xgb.fit(X_train, y_train, sample_weight=weights)
```

### 6.8 Feature Importance — Sanity Check

```python
import matplotlib.pyplot as plt

importances = pd.Series(clf.feature_importances_, index=feature_cols)
importances.nlargest(15).plot(kind='barh')
plt.title('Top 15 Features')
plt.tight_layout()
plt.show()
```

**Expected top features (if data is correct):**
1. `avg_stall_ns` or `log_avg_stall_ns` — should be #1 by far
2. `max_stall_ns` — worst-case latency
3. `involuntary_ratio` — CPU contention proxy
4. `avg_runq_ratio` — system load proxy
5. `voluntary_ratio` — I/O wait proxy

If `pid` or `cpu` sneak into the top 10, you forgot to drop them.
If `avg_stall_ns` is NOT the top feature, check that your BPF program is
actually recording wakeup timestamps (look for non-zero `latency_count` in raw
CSV).

---

## 7. Incremental Testing Protocol — Week by Week

### Week 1: Validate the Pipeline

Goal: confirm the CSV columns contain real signal before collecting a large
dataset.

1. Run `idle_baseline` for 90s → ~5000 rows
2. Run `cpu_high` (stress-ng --cpu 32) for 90s → ~5000 rows
3. Concatenate both CSVs in Jupyter
4. Check: `df.groupby('session_label')['avg_stall_ns'].median()`
   - `idle_baseline` median should be < 500,000 ns (< 0.5ms)
   - `cpu_high` median should be > 10,000,000 ns (> 10ms)
   - If both are similar, the BPF program is not measuring latency correctly

5. Train a binary classifier (class 0 vs class 3 only) — expect >90% accuracy
6. If accuracy is poor, check `latency_count` distribution. If 80%+ of rows
   have `latency_count == 0`, the `sched_wakeup` probe is not firing.

**Debug command to check live:**
```bash
sudo cat /sys/kernel/debug/tracing/trace_pipe | grep -i wakeup | head -20
```

### Week 2: Build the Normal Class Properly

Run 3–4 idle sessions at different times of day (boot, after heavy use, after
Docker activity). The system is never truly idle — background daemons, timers,
and kernel threads generate real class 0 data. This variety prevents the model
from learning "idle means no Docker containers running."

```bash
# Morning, system freshly booted
sudo ./pinner --obj perf_monitor.bpf.o &
sleep 90
sudo ./reader --label idle_fresh --append

# After running a build
sudo ./pinner --obj perf_monitor.bpf.o &
sleep 90
sudo ./reader --label idle_post_build --append

# With browser and IDE open (realistic developer machine idle)
sudo ./pinner --obj perf_monitor.bpf.o &
sleep 90
sudo ./reader --label idle_desktop --append
```

### Week 3: CPU and Memory Stressors

Run the full CPU ladder (steps 1–5 from section 3.1) and the memory stressors
(section 3.2). After each pair of sessions, retrain the model and check if
class boundary accuracy improves.

Target at this point: ~35,000 rows, model accuracy > 80% on 4-class.

### Week 4: Mixed and Realistic Workloads

Run the mixed stressors and realistic workloads (compile jobs, Docker builds).
These are your hardest test cases because the model must generalise beyond
synthetic stress patterns.

Also run: a session where you run `stress-ng --cpu 4` (low stress) but use
memory-intensive operations simultaneously. This tests whether the model
correctly classifies the memory-bound processes as higher class than the lightly
CPU-stressed ones.

---

## 8. Known Gotchas and How to Handle Them

**Gotcha 1: Most rows will be class 0 even during stress sessions.**
This is correct. stress-ng workers will be class 2–3 but the 5000 other
processes on your system remain class 0. This is not a data problem — it
reflects reality. `class_weight='balanced'` handles it.

**Gotcha 2: `latency_count` is 0 for many rows.**
If a process ran continuously without ever being put to sleep and woken up
(e.g. a kernel thread that never blocks), it has no wakeup latency samples.
Drop these rows as shown in 6.2. They are not informative for classification.

**Gotcha 3: `avg_stall_ns` is the same as `max_stall_ns` for many rows.**
This means `latency_count == 1` — the process was only woken up once. These
rows are low-confidence samples. You can add a minimum sample filter:
```python
df = df[df['latency_count'] >= 3].copy()
```
This reduces row count but improves label reliability.

**Gotcha 4: `avg_runq_ratio` is 0 even during CPU stress.**
The runq_len PERCPU_ARRAY in the BPF program is per-CPU. If stress-ng workers
are all pinned to different CPUs than the processes you are measuring, the
ratio will be locally 0 even though the system is overloaded. This is a
limitation of per-CPU tracking. Use it as a secondary signal, not primary.

**Gotcha 5: Docker processes appear in every session.**
Docker's shim and containerd processes always appear in the CSV. They are not
noise — they are legitimate class 0 processes during idle sessions and may
show class 1 activity if Docker is doing something. Leave them in.

**Gotcha 6: Same (pid, cpu) combination appears with very different values
across sessions.**
PIDs are reused by the kernel. PID 49488 in session 1 is a different process
than PID 49488 in session 2. This is fine because you never use PID as a
feature. The `comm` (process name) is stable and more meaningful.

---

## 9. File and Directory Layout

```
project/
├── perf_monitor.bpf.c      # BPF kernel program (compile with clang)
├── perf_monitor.bpf.o      # compiled BPF object
├── pinner.c                # loader and session manager
├── pinner                  # compiled
├── reader.c                # map reader and CSV exporter
├── reader                  # compiled
├── perf_metrics.csv        # accumulated CSV (all sessions --append)
├── sessions.log            # manual log of what you ran and when
└── notebooks/
    ├── 01_eda.ipynb        # exploration, label distribution check
    ├── 02_preprocessing.ipynb  # feature engineering, split
    └── 03_model.ipynb      # training, evaluation, feature importance
```

Keep sessions.log updated manually. Without it, when you see a suspicious
cluster of rows in Jupyter you will not know which real-world condition
produced them.

---

## 10. Compile Reminders

```bash
# BPF object (run on the TARGET machine — kernel headers must match)
clang -O2 -g -target bpf \
      -D__TARGET_ARCH_x86 \
      -I/usr/include/bpf \
      -c perf_monitor.bpf.c -o perf_monitor.bpf.o

# pinner (unchanged from original)
gcc -O2 -o pinner pinner.c -lbpf -lelf -lz

# reader (updated)
gcc -O2 -o reader reader.c -lbpf -lelf -lz

# Verify attachment (should show ~15+ programs attached):
sudo ./pinner --obj perf_monitor.bpf.o
# look for: "[pinner] N programs attached, 0 failed"
# If you see failed attachments, check dmesg for verifier errors:
sudo dmesg | tail -30
```












