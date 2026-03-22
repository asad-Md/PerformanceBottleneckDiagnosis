# perf_monitor

Minimal-overhead eBPF performance bottleneck logger.
BPF writes to kernel maps only during collection. Reader dumps once at the end.

```
perf_monitor.bpf.c   BPF program — all probes in one file
pinner.c             loads BPF, maps auto-pin, probes attach, then waits
reader.c             opens pinned maps after workload, dumps CSV once
plot.py              offline graphs from CSV
perf_analysis.ipynb  same as plot.py but interactive Jupyter
```

---

## Workflow

```bash
# once — compile everything
ecc perf_monitor.bpf.c
gcc -O2 -o pinner pinner.c -lbpf -lelf -lz
gcc -O2 -o reader reader.c -lbpf -lelf -lz

# each collection session
sudo ./pinner --obj perf_monitor.bpf.o          # terminal 1: loads BPF, waits
stress-ng --cpu $(nproc) --timeout 60s          # terminal 2: your workload
# Ctrl+C pinner when workload finishes
sudo ./reader --label cpu_bound                 # dumps CSV

# repeat for each workload type, append to same CSV
sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --vm 2 --vm-bytes 80% --timeout 60s
sudo ./reader --label memory_bound --append

sudo ./pinner --obj perf_monitor.bpf.o
fio --name=io --rw=randrw --bs=4k --runtime=60
sudo ./reader --label io_bound --append

sudo ./pinner --obj perf_monitor.bpf.o
stress-ng --mutex 8 --timeout 60s
sudo ./reader --label contention --append
```

---

## What is being tracked

Each row in the CSV = one (pid, cpu) pair from one collection session.
Values are cumulative totals for that process on that CPU core for the
entire duration the pinner was running.

---

### SECTION 1 — CPU Scheduling
**Probe:** `tp/sched/sched_switch`
Fires on every context switch — when the kernel switches from one task to another.

| Column | What it measures | Bottleneck signal |
|--------|-----------------|-------------------|
| `ctx_switches` | Total context switches for this (pid,cpu) | High overall = busy or contended process |
| `voluntary_switches` | Task gave up CPU willingly (went to sleep waiting for I/O, lock, timer) | High = I/O bound or waiting on locks |
| `involuntary_switches` | Task was preempted — kernel forcibly took CPU away (time slice expired or higher-priority task arrived) | High = CPU oversubscription or CPU-bound |
| `involuntary_pct` | involuntary / total × 100 (derived) | >80% = strong CPU bottleneck indicator |
| `cpu_migrations` | Task moved from one CPU core to another | High = scheduler load balancing, cache thrash risk |
| `total_runtime_ns` | Cumulative nanoseconds this pid actually ran on CPU | Raw on-CPU time |
| `avg_runq_latency_ns` | Average time from task wakeup → task actually scheduled (run queue wait time) | High = scheduler contention, overloaded CPUs |
| `max_runq_latency_ns` | Worst-case run queue wait | Spikes = scheduler stalls |

**Probe:** `tp_btf/sched_wakeup` + `tp_btf/sched_wakeup_new`
Used internally to compute runq latency (records wakeup timestamp, sched_switch computes the delta).

**Probe:** `tp_btf/sched_migrate_task`
Fires when kernel moves a task to a different CPU core. Used to count migrations.

---

### SECTION 2 — Memory
**Probe:** `tp/exceptions/page_fault_user`
Fires on every userspace page fault.

**Probe:** `tp/exceptions/page_fault_kernel`
Fires on every kernel page fault.

| Column | What it measures | Bottleneck signal |
|--------|-----------------|-------------------|
| `minor_faults` | Page was not in memory but no disk I/O needed (e.g. anonymous memory, CoW) | Moderate counts normal; very high = memory pressure |
| `kernel_faults` | Kernel-level page faults — heavier than user faults | High = kernel memory pressure |

**Probe:** `tp_btf/kmalloc`
Fires on every kernel memory allocation.

**Probe:** `tp_btf/kfree`
Fires on every kernel memory free.

| Column | What it measures | Bottleneck signal |
|--------|-----------------|-------------------|
| `kmalloc_count` | Number of kernel allocations made | High = kernel doing lots of memory work |
| `kfree_count` | Number of kernel frees | Compare with kmalloc — large gap = possible accumulation |
| `total_alloc_bytes` | Total bytes allocated via kmalloc | High = memory-intensive kernel path |
| `total_free_bytes` | Total bytes freed | |
| `large_page_allocs` | Multi-page (order > 0) allocations — each is 2^order × 4KB | High = memory pressure, kernel struggling to find contiguous pages |

**Probe:** `tp_btf/mm_page_alloc`
Fires on page allocations. Only order > 0 (multi-page) are counted as they indicate memory pressure.

---

### SECTION 3 — Syscall Latency
**Probes:** `tp/syscalls/sys_enter_*` + `tp/syscalls/sys_exit_*`

Latency = time from syscall entry to syscall return. Measures time the process spent inside the kernel for each syscall type.

| Column | What it measures | Bottleneck signal |
|--------|-----------------|-------------------|
| `syscall_count` | Total syscalls made | |
| `avg_syscall_latency_ns` | Mean time inside kernel per syscall | High = slow kernel paths, I/O waits |
| `max_syscall_latency_ns` | Worst single syscall | Spikes = blocking I/O or lock contention in kernel |
| `read_count` | Number of read() syscalls | |
| `write_count` | Number of write() syscalls | |
| `read_bytes` | Bytes read (from return value) | High = I/O bound reading |
| `write_bytes` | Bytes written | High = I/O bound writing |
| `mmap_count` | Number of mmap() calls | High = frequent memory mappings |
| `futex_count` | Number of futex() calls | futex is the underlying syscall for userspace mutexes/condvars (pthreads, C++ std::mutex all go through this) |
| `avg_futex_latency_ns` | Mean time spent in futex() | High = userspace lock contention — threads waiting on each other |
| `epoll_count` | Number of epoll_wait() calls | |
| `avg_epoll_latency_ns` | Mean time blocked in epoll_wait() | High = process spending lots of time waiting for I/O events |
| `poll_count` | Number of poll() calls | |
| `syscall_error_count` | Syscalls that returned negative (errors) | High = frequent failures, worth investigating |

---

### SECTION 4 — Lock Contention
**Important:** These probes only fire on the **slow path** — meaning actual contention where a thread had to wait. If a lock is taken uncontended (fast path), nothing is recorded. Zero counts mean no contention, not that no locks were used.

**Probes:** `kprobe/__mutex_lock_slowpath` + `kretprobe/__mutex_lock_slowpath`
Fires only when a kernel mutex is contested (another thread holds it).

| Column | What it measures | Bottleneck signal |
|--------|-----------------|-------------------|
| `mutex_contentions` | Times a kernel mutex was contended | High = threads blocking on kernel-level locks |
| `avg_mutex_wait_ns` | Mean time waiting for a mutex | High wait time = severe lock bottleneck |
| `max_mutex_wait_ns` | Worst single mutex wait | |

**Probes:** `kprobe/rwsem_down_read_slowpath` + `kprobe/rwsem_down_write_slowpath`
Fires only when a reader-writer semaphore is contested.

| Column | What it measures | Bottleneck signal |
|--------|-----------------|-------------------|
| `rwsem_read_contentions` | Contended read locks | Multiple readers blocked by a writer |
| `avg_rwsem_read_wait_ns` | Mean read lock wait time | |
| `rwsem_write_contentions` | Contended write locks | Writer blocked by active readers or another writer |
| `avg_rwsem_write_wait_ns` | Mean write lock wait time | High = write-heavy workload causing reader starvation |
| `max_rwsem_write_wait_ns` | Worst write lock wait | |

**Note on symbol names:** `__mutex_lock_slowpath` and `rwsem_down_*_slowpath`
vary by kernel version. Check yours with:
```bash
grep -E "mutex_lock_slow|rwsem_down" /proc/kallsyms | head
```
If not found, delete those 6 kprobe functions from the BPF program.
`futex_count` + `avg_futex_latency_ns` from Section 3 is a good
userspace-level lock proxy that always works.

---

## Bottleneck interpretation guide

| What you see | Likely bottleneck |
|---|---|
| High `involuntary_pct` + high `ctx_switches` | CPU oversubscription — more threads than cores |
| High `avg_runq_latency_ns` | Scheduler contention — tasks waiting a long time to get CPU |
| High `total_faults` + high `large_page_allocs` | Memory pressure — working set doesn't fit in RAM |
| High `total_alloc_bytes` + high `kmalloc_count` | Memory-intensive kernel path |
| High `avg_syscall_latency_ns` + high `read/write_bytes` | I/O bound — time spent waiting on disk/network |
| High `avg_epoll_latency_ns` | Event-driven process blocked waiting for I/O |
| High `futex_count` + high `avg_futex_latency_ns` | Userspace lock contention (pthreads, std::mutex) |
| High `mutex_contentions` + high `avg_mutex_wait_ns` | Kernel-level lock contention |
| High `rwsem_write_contentions` | Writer bottleneck — one writer blocking many readers |
| High `cpu_migrations` | Tasks moving between cores frequently — cache thrash |

---

## CSV schema summary

```
timestamp_s          unix timestamp when reader ran
pid                  process ID
cpu                  CPU core this entry is for
comm                 process name (first 15 chars)

-- scheduling --
ctx_switches
voluntary_switches
involuntary_switches
cpu_migrations
total_runtime_ns
avg_runq_latency_ns
max_runq_latency_ns

-- memory --
minor_faults
kernel_faults
kmalloc_count
kfree_count
total_alloc_bytes
total_free_bytes
large_page_allocs

-- syscalls --
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
avg_epoll_latency_ns
poll_count
syscall_error_count

-- locks --
mutex_contentions
avg_mutex_wait_ns
max_mutex_wait_ns
rwsem_read_contentions
avg_rwsem_read_wait_ns
rwsem_write_contentions
avg_rwsem_write_wait_ns
max_rwsem_write_wait_ns

-- label --
label                workload type set via --label flag on reader
```

---

## What is NOT tracked (yet)

- Hardware perf counters (LLC misses, cycles, IPC) — need `perf_event_open`, planned for Phase 2
- Network I/O — would need `tp/net/` tracepoints
- Disk block I/O latency — would need `tp/block/` tracepoints
- Per-file I/O — would need `kprobe/vfs_read` etc.
- GPU metrics — outside eBPF scope