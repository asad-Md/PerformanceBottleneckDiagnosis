# perf_monitor

eBPF-based performance bottleneck logger and ML dataset builder.
Minimal overhead during collection — BPF writes to kernel maps only,
userspace reads once at the end.

```
perf_monitor.bpf.c   BPF program — all probes in one file
pinner.c             loads BPF, pins maps, attaches probes, waits
reader.c             opens pinned maps, trims warmup/teardown, dumps CSV
labeler.py           fix or fill labels in existing CSV
perf_analysis.ipynb  exploratory analysis and bottleneck visualisation
plot.py              same as notebook but command-line
```

---

## Build (once)

```bash
# compile BPF program
ecc perf_monitor.bpf.c

# compile pinner and reader
gcc -O2 -o pinner pinner.c -lbpf -lelf -lz
gcc -O2 -o reader reader.c -lbpf -lelf -lz

# make sure BPF filesystem is mounted (usually already is)
mount | grep bpf
# if nothing shows: sudo mount -t bpf bpf /sys/fs/bpf

# install benchmark tools
sudo apt install stress-ng fio
```

---

## Collection workflow

Every session follows this exact pattern:

```bash
# terminal 1 — start pinner
sudo ./pinner --obj perf_monitor.bpf.o

# terminal 2 — run your workload
<benchmark command>

# when workload finishes, Ctrl+C pinner in terminal 1

# dump maps to CSV
sudo ./reader --label <label_name> --append
```

First session ever: omit `--append` (creates fresh CSV).
All subsequent sessions: always use `--append`.

---

## How session trimming works

Benchmarks take a few seconds to ramp up (threads spawning, memory allocating)
and a few seconds to wind down (threads exiting, memory freeing). These
startup/teardown phases do not represent the steady-state bottleneck you want
to label. Reader automatically trims them.

**Pinner** writes a metadata file `/tmp/perf_session.meta` containing:
- `start_ns` — timestamp when pinner started (after all probes attached)
- `end_ns`   — timestamp when you pressed Ctrl+C

**Reader** computes the valid window:
```
valid_window = [start_ns + skip_start,  end_ns - skip_end]
```

For each `(pid, cpu)` entry it checks `last_seen_ns` — the timestamp of
the most recent kernel event for that process. Entries last active outside
the valid window are dropped before writing to CSV.

Defaults: skip first **5 seconds**, skip last **2 seconds**.

Override with flags on reader:
```bash
sudo ./reader --label cpu_bound --skip-start 8 --skip-end 3
sudo ./reader --label cpu_bound --no-trim      # disable entirely
```

If the session is shorter than `skip_start + skip_end`, reader warns and
disables trimming automatically rather than producing empty output.

---

## Pinner options

```
--obj <path>    BPF object file to load   (default: perf_monitor.bpf.o)
```

Pinner prints session start time and duration on exit:
```
[pinner] session started at 1718123456.789
[pinner] collecting... Ctrl+C to stop
[pinner] stopped. session duration: 92.4s
[pinner] maps still pinned at /sys/fs/bpf/
[pinner] now run: sudo ./reader --label <your_label>
```

---

## Reader options

```
--label <str>        workload label for every row   (default: unknown)
--out <path>         CSV output path                (default: perf_metrics.csv)
--append             append to existing CSV
--skip-start <secs>  trim first N seconds           (default: 5)
--skip-end <secs>    trim last N seconds            (default: 2)
--no-trim            disable all trimming
```

Reader prints a summary on exit:
```
[reader] session duration: 92.4s
[reader] trimming: skip first 5s, skip last 2s
[reader] raw entries: sched=2891 mem=1043 syscall=987 lock=81
[reader] 3102 unique (pid,cpu) entries before trim
[reader] trimmed 247 entries (warmup/teardown noise)
[reader] wrote 2855 rows → perf_metrics.csv
[reader] unpinned maps, removed session meta
```

---

## Labeler (emergency fix tool)

Only needed if you forgot `--label` on reader or need to correct a label.

```bash
# preview what would change without writing
python3 labeler.py --label cpu_bound --last 500 --dry-run

# fix last N rows (forgot --label on reader)
python3 labeler.py --label cpu_bound --last 500

# fix a specific row range (0-indexed)
python3 labeler.py --label memory_bound --from 1200 --to 1800

# fill only rows where label is empty or 'unknown'
python3 labeler.py --label io_bound

# overwrite every row (re-label entire file)
python3 labeler.py --label contention --overwrite-all
```

---

## Full benchmark plan

### Recommended collection order

Run in this order so you can validate the pipeline works before
collecting all sessions:

```
1. normal_idle       ← baseline, should look very quiet
2. cpu_bound × 2    ← most distinct signal
3. memory_bound × 2
4. Open perf_analysis.ipynb → check Figure 1 + 2
   If labels visibly cluster → continue collecting
   If not → check BPF program is running, check metric values are non-zero
5. Collect remaining sessions
```

---

### BASELINE / NORMAL

```bash
# normal_idle — laptop sitting, nothing running
sudo ./pinner --obj perf_monitor.bpf.o
sleep 90
sudo ./reader --label normal_idle --append

# normal_light — typical light laptop usage
sudo ./pinner --obj perf_monitor.bpf.o
# browse, scroll, read for 90 seconds
sleep 90
sudo ./reader --label normal_light --append

# normal_mixed — light mixed load
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --cpu 4 --vm 1 --vm-bytes 20% --timeout 90s
sudo ./reader --label normal_mixed --append
```

---

### CPU BOUND

Goal: oversubscribe CPU so threads compete for cores.
On 4780HS (16 logical threads): always use ≥ 32 threads.
Signal: high `involuntary_pct`, high `avg_runq_latency_ns`.

```bash
# cpu_1 — pure compute, 2× oversubscribe
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --cpu 32 --timeout 90s
sudo ./reader --label cpu_bound --append

# cpu_2 — matrix multiply
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --matrix 16 --timeout 90s
sudo ./reader --label cpu_bound --append

# cpu_3 — FFT (floating point heavy)
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --cpu 16 --cpu-method fft --timeout 90s
sudo ./reader --label cpu_bound --append

# cpu_4 — fork pressure (many short-lived processes)
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --fork 32 --timeout 90s
sudo ./reader --label cpu_bound --append

# cpu_5 — matrix product
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --cpu 16 --cpu-method matrixprod --timeout 90s
sudo ./reader --label cpu_bound --append
```

---

### MEMORY BOUND

Goal: exceed comfortable working set → memory pressure.
On 16GB RAM: use ≥ 85% (≥ 13.6GB).
Signal: high `total_faults`, high `large_page_allocs`, high `total_alloc_bytes`.

```bash
# mem_1 — large working set, random access
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --vm 4 --vm-bytes 85% --vm-method random-walk --timeout 90s
sudo ./reader --label memory_bound --append

# mem_2 — cache thrash (exceeds L3 repeatedly)
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --cache 8 --timeout 90s
sudo ./reader --label memory_bound --append

# mem_3 — mmap pressure
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --mmap 8 --mmap-bytes 1G --timeout 90s
sudo ./reader --label memory_bound --append

# mem_4 — zero-one pattern (page fault heavy)
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --vm 4 --vm-bytes 90% --vm-method zero-one --timeout 90s
sudo ./reader --label memory_bound --append

# mem_5 — malloc/free churn (kmalloc pressure)
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --malloc 8 --timeout 90s
sudo ./reader --label memory_bound --append
```

---

### IO BOUND

Goal: saturate disk queue → high syscall latency, high read/write bytes.
Signal: high `avg_syscall_latency_ns`, high `read_bytes`/`write_bytes`,
high `avg_epoll_latency_ns`.

```bash
# io_1 — random read/write
sudo ./pinner --obj perf_monitor.bpf.o
fio --name=randio --ioengine=libaio --rw=randrw --bs=4k \
    --numjobs=8 --iodepth=64 --size=2G --runtime=90 \
    --time_based --filename=/tmp/fio_test
sudo ./reader --label io_bound --append

# io_2 — sequential read
sudo ./pinner --obj perf_monitor.bpf.o
fio --name=seqread --ioengine=libaio --rw=read --bs=1M \
    --numjobs=4 --iodepth=32 --size=2G --runtime=90 \
    --time_based --filename=/tmp/fio_test
sudo ./reader --label io_bound --append

# io_3 — metadata heavy (dentry pressure)
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --dentry 8 --timeout 90s
sudo ./reader --label io_bound --append

# io_4 — pipe I/O (inter-process data movement)
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --pipe 16 --timeout 90s
sudo ./reader --label io_bound --append

# io_5 — sync writes (fsync heavy)
sudo ./pinner --obj perf_monitor.bpf.o
fio --name=syncwrite --ioengine=sync --rw=write --bs=4k \
    --numjobs=8 --size=1G --runtime=90 \
    --time_based --fsync=1 --filename=/tmp/fio_test
sudo ./reader --label io_bound --append

# cleanup fio test files
rm /tmp/fio_test
```

---

### LOCK CONTENTION

Goal: many threads competing for the same lock → high futex/mutex counts.
Note: lock probes only fire on the slow path (actual contention).
Signal: high `futex_count`, high `avg_futex_latency_ns`,
high `mutex_contentions`, high `avg_mutex_wait_ns`.

```bash
# lock_1 — kernel mutex contention
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --mutex 16 --timeout 90s
sudo ./reader --label contention --append

# lock_2 — futex (userspace locks: pthreads, std::mutex)
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --futex 16 --timeout 90s
sudo ./reader --label contention --append

# lock_3 — semaphore contention
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --sem 16 --timeout 90s
sudo ./reader --label contention --append

# lock_4 — spinlock
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --spinlock 16 --timeout 90s
sudo ./reader --label contention --append

# lock_5 — reader-writer lock
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --rwlock 16 --timeout 90s
sudo ./reader --label contention --append
```

---

### MIXED

```bash
# mixed_1 — CPU + memory
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --cpu 16 --vm 2 --vm-bytes 60% --timeout 90s
sudo ./reader --label mixed --append

# mixed_2 — I/O + contention
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --mutex 8 --hdd 4 --timeout 90s
sudo ./reader --label mixed --append
```

---

## What is being tracked

Each CSV row = one (pid, cpu) pair, cumulative totals for the entire
pinner session (minus trimmed warmup/teardown entries).

---

### CPU Scheduling
Probes: `tp/sched/sched_switch`, `tp_btf/sched_wakeup`,
        `tp_btf/sched_wakeup_new`, `tp_btf/sched_migrate_task`

| Column | Source event | What it measures | High value → |
|--------|-------------|-----------------|--------------|
| `ctx_switches` | sched_switch | Total context switches | Busy or contended process |
| `voluntary_switches` | sched_switch (prev_state ≠ RUNNING) | Task slept waiting for I/O, lock, or timer | I/O or lock wait |
| `involuntary_switches` | sched_switch (prev_state == RUNNING) | Task forcibly preempted by kernel | CPU oversubscription |
| `cpu_migrations` | sched_migrate_task | Task moved to a different CPU core | Cache thrash, load balancing |
| `total_runtime_ns` | sched_switch | Cumulative nanoseconds on CPU | Raw CPU consumption |
| `avg_runq_latency_ns` | wakeup → sched_switch delta | Mean time waiting in run queue to be scheduled | Scheduler contention |
| `max_runq_latency_ns` | same | Worst-case scheduler delay | Scheduling stalls |

Derived column (added in notebook):
```
involuntary_pct = involuntary_switches / ctx_switches × 100
```
\> 80% = strong CPU bottleneck signal.

---

### Memory
Probes: `tp/exceptions/page_fault_user`, `tp/exceptions/page_fault_kernel`,
        `tp_btf/kmalloc`, `tp_btf/kfree`, `tp_btf/mm_page_alloc`

| Column | Source event | What it measures | High value → |
|--------|-------------|-----------------|--------------|
| `minor_faults` | page_fault_user | Page not present, no disk I/O needed (anon memory, CoW) | Memory pressure building |
| `kernel_faults` | page_fault_kernel | Kernel-level page fault | Kernel memory pressure |
| `kmalloc_count` | kmalloc | Kernel memory allocations | Kernel memory activity |
| `kfree_count` | kfree | Kernel memory frees | |
| `total_alloc_bytes` | kmalloc | Total bytes allocated in kernel | Memory-intensive kernel path |
| `total_free_bytes` | kfree | Total bytes freed | |
| `large_page_allocs` | mm_page_alloc (order > 0) | Multi-page contiguous allocs — hard to satisfy under pressure | Memory fragmentation / pressure |

---

### Syscall Latency
Probes: `tp/syscalls/sys_enter_*` + `tp/syscalls/sys_exit_*`
Tracked: read, write, mmap, futex, epoll_wait, poll

| Column | What it measures | High value → |
|--------|-----------------|--------------|
| `syscall_count` | Total tracked syscalls | |
| `avg_syscall_latency_ns` | Mean time inside kernel per syscall | Slow kernel paths, blocking I/O |
| `max_syscall_latency_ns` | Worst single syscall duration | Blocking I/O or kernel lock |
| `read_count` / `read_bytes` | Read syscall count and bytes transferred | I/O read bound |
| `write_count` / `write_bytes` | Write syscall count and bytes transferred | I/O write bound |
| `mmap_count` | Memory mapping calls | Frequent memory remapping |
| `futex_count` | Userspace mutex/condvar calls (all pthreads go through this) | Userspace lock contention |
| `avg_futex_latency_ns` | Mean time blocked in futex() | Threads waiting on each other |
| `epoll_count` | Event loop I/O wait calls | |
| `avg_epoll_latency_ns` | Mean time blocked in epoll_wait() | Process blocked waiting for I/O events |
| `poll_count` | Poll syscall count | |
| `syscall_error_count` | Syscalls returning negative (errors) | Frequent failures |

---

### Lock Contention
Probes: `kprobe/__mutex_lock_slowpath`, `kretprobe/__mutex_lock_slowpath`,
        `kprobe/rwsem_down_read_slowpath`, `kprobe/rwsem_down_write_slowpath`

**These probes only fire on the slow path — actual contention where a thread
had to wait. Zero = no contention, not no locks.**

| Column | What it measures | High value → |
|--------|-----------------|--------------|
| `mutex_contentions` | Kernel mutex contested | Threads blocking on kernel locks |
| `avg_mutex_wait_ns` | Mean kernel mutex wait time | Severe lock bottleneck |
| `max_mutex_wait_ns` | Worst kernel mutex wait | |
| `rwsem_read_contentions` | Read lock contested | Writer blocking readers |
| `avg_rwsem_read_wait_ns` | Mean read lock wait | |
| `rwsem_write_contentions` | Write lock contested | Write-heavy bottleneck |
| `avg_rwsem_write_wait_ns` | Mean write lock wait | |
| `max_rwsem_write_wait_ns` | Worst write lock wait | |

---

## Bottleneck interpretation

| Pattern in data | Bottleneck type |
|----------------|----------------|
| High `involuntary_pct` + high `ctx_switches` | CPU oversubscription |
| High `avg_runq_latency_ns` | Scheduler contention |
| High `total_faults` + high `large_page_allocs` | Memory pressure |
| High `total_alloc_bytes` + high `kmalloc_count` | Memory-intensive kernel path |
| High `avg_syscall_latency_ns` + high `read/write_bytes` | I/O bound |
| High `avg_epoll_latency_ns` | Event loop blocked on I/O |
| High `futex_count` + high `avg_futex_latency_ns` | Userspace lock contention |
| High `mutex_contentions` + high `avg_mutex_wait_ns` | Kernel lock contention |
| High `rwsem_write_contentions` | Writer blocking readers |
| High `cpu_migrations` | Cache thrash from task bouncing cores |

---

## Lock probe symbol check

Kernel symbol names vary by version. Verify before compiling:

```bash
grep -E "mutex_lock_slow|rwsem_down" /proc/kallsyms | head
```

If names differ, update the `SEC("kprobe/...")` lines in
`perf_monitor.bpf.c` and recompile with `ecc`.
If not exported at all, delete those 6 kprobe functions —
`futex_count` + `avg_futex_latency_ns` is a sufficient
userspace lock contention proxy.

---

## What is not tracked yet

- Hardware perf counters (LLC misses, cycles, IPC) — needs `perf_event_open`
- Network I/O latency — needs `tp/net/` tracepoints
- Disk block I/O latency — needs `tp/block/` tracepoints
- Per-file I/O breakdown — needs `kprobe/vfs_read` etc.
- GPU metrics — outside eBPF scope