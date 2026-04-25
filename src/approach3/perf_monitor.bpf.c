// SPDX-License-Identifier: GPL-2.0
/*
 * perf_monitor.bpf.c
 *
 * Kernel-Verified Labeling eBPF program.
 *
 * Design:
 *   - sched_wakeup / sched_wakeup_new : record per-task wakeup timestamp
 *   - sched_switch                    : measure runqueue latency (wakeup→run),
 *                                       update per-(pid,cpu) aggregate map,
 *                                       track runqueue length per CPU
 *   - sched_process_exit              : clean up scratch entries
 *   - kmem_cache_alloc / kmem_cache_free : memory pressure signals
 *   - sys_enter_* / sys_exit_*        : syscall latency (read/write/mmap/futex)
 *
 * Maps (all pinned automatically via SEC(".maps") + pin_path):
 *   sched_map   – per (pid,cpu) scheduling + latency aggregate
 *   mem_map     – per (pid,cpu) memory pressure signals
 *   syscall_map – per (pid,cpu) syscall latency aggregate
 *   lock_map    – per (pid,cpu) contention signals
 *   wakeup_scratch – transient per-pid wakeup timestamp (not pinned/exported)
 *   runq_len    – per-CPU runqueue length counter
 *
 * All counters are cumulative since map creation.  reader.c reads the maps
 * once at the end of the session and exports raw values.  Post-processing
 * in Jupyter groups them into time-series buckets using the `timestamp_ns`
 * column (bpf_ktime_get_ns() at the moment of last update).
 *
 * Build:
 *   clang -O2 -g -target bpf \
 *         -D__TARGET_ARCH_x86 \
 *         -I/usr/include/bpf \
 *         -c perf_monitor.bpf.c -o perf_monitor.bpf.o
 * clang -O2 -g -target bpf -D__TARGET_ARCH_x86 -I/usr/include/bpf -c perf_monitor.bpf.c -o perf_monitor.bpf.o
 */

#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-declarations"
#include "vmlinux.h"
#pragma clang diagnostic pop
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

#define TASK_COMM_LEN 16
#define MAX_ENTRIES 65536
#define MAX_CPUS 64

/* ── helper: integer divide avoiding div-by-zero ──────────────────────────── */
static __always_inline __u64 safe_div(__u64 a, __u64 b)
{
    return b ? a / b : 0;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * Map value structs
 * ══════════════════════════════════════════════════════════════════════════════ */

struct agg_key
{
    __u32 pid;
    __u32 cpu;
};

/*
 * sched_val
 *
 * latency fields carry WAKEUP→RUN latency, i.e. time spent in runqueue.
 * runq_ratio_sum / runq_ratio_count gives average system load at event time.
 * runq_ratio is stored as (nrunnable * 100) / ncpus to avoid floats.
 *
 * "Victim" signals:
 *   voluntary_switches   – task called schedule() voluntarily (I/O wait, sleep)
 *   involuntary_switches – task was preempted (CPU contention)
 *   stall_ns             – cumulative time the task was runnable but not running
 *                          (= sum of runqueue latency; renamed for clarity)
 *   max_stall_ns         – worst single runqueue latency observed for this task
 */
struct sched_val
{
    __u64 ctx_switches;         /* total on-CPU events          */
    __u64 voluntary_switches;   /* gave up CPU (I/O / sleep)    */
    __u64 involuntary_switches; /* preempted (CPU contention)   */
    __u64 cpu_migrations;       /* moved across CPUs            */
    __u64 total_runtime_ns;     /* sum of on-CPU durations      */

    /* wakeup → run latency (runqueue stall) */
    __u64 stall_ns;      /* cumulative runq wait         */
    __u64 max_stall_ns;  /* worst single runq wait       */
    __u64 latency_count; /* #samples contributing        */

    /* system load proxy at event time */
    __u64 runq_ratio_sum;   /* sum of (nrunnable*100/ncpus) */
    __u64 runq_ratio_count; /* #samples                     */

    __u64 last_update_ns; /* bpf_ktime_get_ns() at last write */
    char comm[TASK_COMM_LEN];
};

struct mem_val
{
    __u64 minor_faults;
    __u64 major_faults;
    __u64 kmalloc_count;
    __u64 kfree_count;
    __u64 total_alloc_bytes;
    __u64 total_free_bytes;
    __u64 large_page_allocs; /* allocs >= 2 MiB              */
    __u64 last_update_ns;
};

struct syscall_val
{
    __u64 total_count;
    __u64 latency_sum_ns;
    __u64 latency_max_ns;
    __u64 read_count;
    __u64 write_count;
    __u64 read_bytes;
    __u64 write_bytes;
    __u64 mmap_count;
    __u64 futex_count;
    __u64 futex_latency_sum_ns;
    __u64 epoll_count;
    __u64 epoll_latency_sum_ns;
    __u64 poll_count;
    __u64 error_count;
    __u64 last_update_ns;
};

struct lock_val
{
    __u64 mutex_contentions;
    __u64 mutex_wait_sum_ns;
    __u64 mutex_wait_max_ns;
    __u64 rwsem_read_contentions;
    __u64 rwsem_read_wait_sum_ns;
    __u64 rwsem_write_contentions;
    __u64 rwsem_write_wait_sum_ns;
    __u64 rwsem_write_wait_max_ns;
    __u64 last_update_ns;
};

/* transient scratch: pid → wakeup timestamp */
struct wakeup_entry
{
    __u64 wakeup_ns;
    __u32 wakeup_cpu;
};

/* transient scratch: pid → syscall start timestamp + nr */
struct syscall_entry
{
    __u64 enter_ns;
    __u64 nr;
};

/* transient scratch: pid → on-CPU start timestamp */
struct oncpu_entry
{
    __u64 start_ns;
};

/* ══════════════════════════════════════════════════════════════════════════════
 * Map declarations
 *
 * pin_path causes libbpf to automatically pin the map to
 * /sys/fs/bpf/<map_name> when the object is loaded.
 * ══════════════════════════════════════════════════════════════════════════════ */

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, struct agg_key);
    __type(value, struct sched_val);
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} sched_map SEC(".maps");

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, struct agg_key);
    __type(value, struct mem_val);
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} mem_map SEC(".maps");

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, struct agg_key);
    __type(value, struct syscall_val);
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} syscall_map SEC(".maps");

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, struct agg_key);
    __type(value, struct lock_val);
    __uint(pinning, LIBBPF_PIN_BY_NAME);
} lock_map SEC(".maps");

/* scratch maps — NOT pinned (pid-keyed, short-lived) */
struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, __u32); /* pid */
    __type(value, struct wakeup_entry);
} wakeup_scratch SEC(".maps");

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, __u32); /* pid */
    __type(value, struct syscall_entry);
} syscall_scratch SEC(".maps");

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, __u32); /* pid */
    __type(value, struct oncpu_entry);
} oncpu_scratch SEC(".maps");

/*
 * Per-CPU runqueue length.
 * Updated on sched_switch: +1 when prev is still runnable (stays in runq),
 * -1 when next leaves runq (starts running).
 * Using PERCPU_ARRAY for lock-free per-CPU access.
 */
struct
{
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} runq_len SEC(".maps");

/* ══════════════════════════════════════════════════════════════════════════════
 * Helper: get or create a zeroed sched_val for (pid, cpu)
 * ══════════════════════════════════════════════════════════════════════════════ */
static __always_inline struct sched_val *
get_or_create_sched(struct agg_key *k)
{
    struct sched_val *v = bpf_map_lookup_elem(&sched_map, k);
    if (!v)
    {
        struct sched_val zero = {};
        bpf_map_update_elem(&sched_map, k, &zero, BPF_NOEXIST);
        v = bpf_map_lookup_elem(&sched_map, k);
    }
    return v;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * sched_wakeup / sched_wakeup_new
 *
 * Record the timestamp when a task becomes runnable.  We store this in
 * wakeup_scratch keyed by pid.  sched_switch will consume it.
 * ══════════════════════════════════════════════════════════════════════════════ */
SEC("tp/sched/sched_wakeup")
int handle_wakeup(struct trace_event_raw_sched_wakeup_template *ctx)
{
    __u32 pid = ctx->pid;
    __u64 now = bpf_ktime_get_ns();

    struct wakeup_entry we = {
        .wakeup_ns = now,
        .wakeup_cpu = ctx->target_cpu,
    };
    bpf_map_update_elem(&wakeup_scratch, &pid, &we, BPF_ANY);
    return 0;
}

SEC("tp/sched/sched_wakeup_new")
int handle_wakeup_new(struct trace_event_raw_sched_wakeup_template *ctx)
{
    __u32 pid = ctx->pid;
    __u64 now = bpf_ktime_get_ns();

    struct wakeup_entry we = {
        .wakeup_ns = now,
        .wakeup_cpu = ctx->target_cpu,
    };
    bpf_map_update_elem(&wakeup_scratch, &pid, &we, BPF_ANY);
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * sched_switch
 *
 * Fires on every context switch.
 *
 *  prev_pid / prev_state: task that is leaving the CPU
 *  next_pid / next_comm:  task that is going onto the CPU
 *
 * For the INCOMING task (next):
 *   - Compute wakeup→run latency using wakeup_scratch
 *   - Update runq_ratio sample (nrunnable on this CPU / online CPUs)
 *   - Record on-CPU start time in oncpu_scratch
 *
 * For the OUTGOING task (prev):
 *   - Compute on-CPU duration (start → now)
 *   - Classify as voluntary (TASK_INTERRUPTIBLE/UNINTERRUPTIBLE) or involuntary
 *   - Detect CPU migration (prev cpu != map cpu — only approximate here)
 *   - If prev is still runnable (TASK_RUNNING), increment runq_len (stays queued)
 * ══════════════════════════════════════════════════════════════════════════════ */

/* task state bits (Linux kernel) */
#define TASK_RUNNING 0x0000
#define TASK_INTERRUPTIBLE 0x0001
#define TASK_UNINTERRUPTIBLE 0x0002

SEC("tp/sched/sched_switch")
int handle_switch(struct trace_event_raw_sched_switch *ctx)
{
    __u64 now = bpf_ktime_get_ns();
    __u32 cpu = bpf_get_smp_processor_id();
    __u32 prev_pid = ctx->prev_pid;
    __u32 next_pid = ctx->next_pid;
    long prev_state = ctx->prev_state;

    /* ── runq_len update ──────────────────────────────────────────────────── */
    __u32 zero = 0;
    __u64 *rql = bpf_map_lookup_elem(&runq_len, &zero);
    if (rql)
    {
        /*
         * prev is leaving CPU.  If it is still runnable (TASK_RUNNING == 0)
         * it goes back to the runqueue → length stays same or +1.
         * If sleeping, length decreases.
         * next is starting to run → always -1 from runqueue.
         * Net change when prev is runnable: 0  (leaves and enters queue)
         * Net change when prev sleeps:     -1
         * We track it simply: increment if prev stays runnable.
         */
        if ((prev_state & (TASK_INTERRUPTIBLE | TASK_UNINTERRUPTIBLE)) == 0 &&
            prev_pid != 0)
            __sync_fetch_and_add(rql, 1); /* stays in runq               */
        if (*rql > 0)
            __sync_fetch_and_add(rql, (__u64)-1); /* next leaves runq      */
    }

    /* ── approximate runq length for ratio sample ─────────────────────────── */
    __u64 runq_snapshot = rql ? *rql : 0;
    /*
     * ncpus: we use a compile-time constant matching the target machine.
     * AMD Ryzen 7 4800HS has 8 cores / 16 threads.
     * runq_ratio = (nrunnable * 100) / ncpus  (integer, no floats in BPF)
     */
#define NCPUS 16
    __u64 runq_ratio = (runq_snapshot * 100) >> 4; /* /16 via shift */

    /* ── handle INCOMING task (next) ─────────────────────────────────────── */
    if (next_pid != 0)
    {
        struct agg_key nk = {.pid = next_pid, .cpu = cpu};
        struct sched_val *nv = get_or_create_sched(&nk); // <--- Updated call
        if (nv)
        {
            nv->ctx_switches++;
            nv->last_update_ns = now;

            // SAFELY read the process name from the context
            bpf_probe_read_kernel_str(nv->comm, TASK_COMM_LEN, ctx->next_comm);

            /* wakeup→run latency */
            struct wakeup_entry *we = bpf_map_lookup_elem(&wakeup_scratch, &next_pid);
            if (we && we->wakeup_ns > 0 && now > we->wakeup_ns)
            {
                __u64 stall = now - we->wakeup_ns;
                nv->stall_ns += stall;
                nv->latency_count++;
                if (stall > nv->max_stall_ns)
                    nv->max_stall_ns = stall;
            }
            bpf_map_delete_elem(&wakeup_scratch, &next_pid);

            /* runq ratio sample */
            nv->runq_ratio_sum += runq_ratio;
            nv->runq_ratio_count++;
        }

        /* record on-CPU start for runtime measurement */
        struct oncpu_entry oe = {.start_ns = now};
        bpf_map_update_elem(&oncpu_scratch, &next_pid, &oe, BPF_ANY);
    }

    /* ── handle OUTGOING task (prev) ─────────────────────────────────────── */
    if (prev_pid != 0)
    {
        struct agg_key pk = {.pid = prev_pid, .cpu = cpu};
        struct sched_val *pv = get_or_create_sched(&pk); // <--- Updated call
        if (pv)
        {
            pv->last_update_ns = now;

            // SAFELY read the process name from the context
            bpf_probe_read_kernel_str(pv->comm, TASK_COMM_LEN, ctx->prev_comm);

            /* on-CPU runtime */
            struct oncpu_entry *oe = bpf_map_lookup_elem(&oncpu_scratch, &prev_pid);
            if (oe && oe->start_ns > 0 && now > oe->start_ns)
                pv->total_runtime_ns += now - oe->start_ns;
            bpf_map_delete_elem(&oncpu_scratch, &prev_pid);

            /* voluntary vs involuntary */
            if (prev_state & (TASK_INTERRUPTIBLE | TASK_UNINTERRUPTIBLE))
                pv->voluntary_switches++;
            else
                pv->involuntary_switches++;
        }
    }

    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * sched_migrate_task  — CPU migration signal
 * ══════════════════════════════════════════════════════════════════════════════ */
SEC("tp/sched/sched_migrate_task")
int handle_migrate(struct trace_event_raw_sched_migrate_task *ctx)
{
    __u32 pid = ctx->pid;
    __u32 cpu = ctx->dest_cpu;
    __u64 now = bpf_ktime_get_ns();

    struct agg_key k = {.pid = pid, .cpu = cpu};
    struct sched_val *v = get_or_create_sched(&k);
    if (v)
    {
        v->cpu_migrations++;
        v->last_update_ns = now;

        /* * We intentionally skip reading the 'comm' string here.
         * The handle_switch tracepoint already reliably captures
         * the process name when it actually runs!
         */
    }
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * Page faults — memory pressure "victim" signals
 *
 * handle_page_fault_user  : minor faults (page reclaim, CoW)
 * handle_page_fault_kernel: kernel-space faults (rare; indicates pressure)
 * ══════════════════════════════════════════════════════════════════════════════ */
SEC("tp/exceptions/page_fault_user")
int handle_page_fault_user(void *ctx)
{
    __u64 now = bpf_ktime_get_ns();
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    __u32 cpu = bpf_get_smp_processor_id();

    struct agg_key k = {.pid = pid, .cpu = cpu};
    struct mem_val *v = bpf_map_lookup_elem(&mem_map, &k);
    if (!v)
    {
        struct mem_val zero = {};
        bpf_map_update_elem(&mem_map, &k, &zero, BPF_NOEXIST);
        v = bpf_map_lookup_elem(&mem_map, &k);
    }
    if (v)
    {
        v->minor_faults++;
        v->last_update_ns = now;
    }
    return 0;
}

SEC("tp/exceptions/page_fault_kernel")
int handle_page_fault_kernel(void *ctx)
{
    __u64 now = bpf_ktime_get_ns();
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    __u32 cpu = bpf_get_smp_processor_id();

    struct agg_key k = {.pid = pid, .cpu = cpu};
    struct mem_val *v = bpf_map_lookup_elem(&mem_map, &k);
    if (!v)
    {
        struct mem_val zero = {};
        bpf_map_update_elem(&mem_map, &k, &zero, BPF_NOEXIST);
        v = bpf_map_lookup_elem(&mem_map, &k);
    }
    if (v)
    {
        v->major_faults++;
        v->last_update_ns = now;
    }
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * kmem_cache_alloc / kmem_cache_free — kernel allocator pressure
 * ══════════════════════════════════════════════════════════════════════════════ */
SEC("tp_btf/kmalloc")
int BPF_PROG(handle_kmalloc, unsigned long call_site, const void *ptr,
             size_t bytes_req, size_t bytes_alloc, gfp_t gfp_flags)
{
    __u64 now = bpf_ktime_get_ns();
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    __u32 cpu = bpf_get_smp_processor_id();
    __u64 size = bytes_alloc;

    struct agg_key k = {.pid = pid, .cpu = cpu};
    struct mem_val *v = bpf_map_lookup_elem(&mem_map, &k);
    if (!v)
    {
        struct mem_val zero = {};
        bpf_map_update_elem(&mem_map, &k, &zero, BPF_NOEXIST);
        v = bpf_map_lookup_elem(&mem_map, &k);
    }
    if (v)
    {
        v->kmalloc_count++;
        v->total_alloc_bytes += size;
        if (size >= (2ULL << 20)) /* >= 2 MiB */
            v->large_page_allocs++;
        v->last_update_ns = now;
    }
    return 0;
}

SEC("tp/kmem/kfree")
int handle_kfree(void *ctx)
{
    __u64 now = bpf_ktime_get_ns();
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    __u32 cpu = bpf_get_smp_processor_id();

    struct agg_key k = {.pid = pid, .cpu = cpu};
    struct mem_val *v = bpf_map_lookup_elem(&mem_map, &k);
    if (!v)
    {
        struct mem_val zero = {};
        bpf_map_update_elem(&mem_map, &k, &zero, BPF_NOEXIST);
        v = bpf_map_lookup_elem(&mem_map, &k);
    }
    if (v)
    {
        v->kfree_count++;
        v->last_update_ns = now;
    }
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * Syscall entry / exit — latency tracking
 *
 * We intercept sys_enter_* to record (timestamp, syscall_nr) and
 * sys_exit_*  to compute latency and update the aggregate.
 *
 * Tracked individually: read, write, mmap, futex, epoll_wait, ppoll.
 * All others contribute to total_count + latency aggregate only.
 * ══════════════════════════════════════════════════════════════════════════════ */

/* syscall numbers (x86-64) */
#define SYS_read 0
#define SYS_write 1
#define SYS_mmap 9
#define SYS_mprotect 10
#define SYS_futex 202
#define SYS_epoll_wait 232
#define SYS_ppoll 271

SEC("tp/syscalls/sys_enter_read")
int handle_sys_enter_read(struct trace_event_raw_sys_enter *ctx)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct syscall_entry se = {.enter_ns = bpf_ktime_get_ns(), .nr = SYS_read};
    bpf_map_update_elem(&syscall_scratch, &pid, &se, BPF_ANY);
    return 0;
}

SEC("tp/syscalls/sys_enter_write")
int handle_sys_enter_write(struct trace_event_raw_sys_enter *ctx)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct syscall_entry se = {.enter_ns = bpf_ktime_get_ns(), .nr = SYS_write};
    bpf_map_update_elem(&syscall_scratch, &pid, &se, BPF_ANY);
    return 0;
}

SEC("tp/syscalls/sys_enter_mmap")
int handle_sys_enter_mmap(struct trace_event_raw_sys_enter *ctx)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct syscall_entry se = {.enter_ns = bpf_ktime_get_ns(), .nr = SYS_mmap};
    bpf_map_update_elem(&syscall_scratch, &pid, &se, BPF_ANY);
    return 0;
}

SEC("tp/syscalls/sys_enter_futex")
int handle_sys_enter_futex(struct trace_event_raw_sys_enter *ctx)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct syscall_entry se = {.enter_ns = bpf_ktime_get_ns(), .nr = SYS_futex};
    bpf_map_update_elem(&syscall_scratch, &pid, &se, BPF_ANY);
    return 0;
}

SEC("tp/syscalls/sys_enter_epoll_wait")
int handle_sys_enter_epoll(struct trace_event_raw_sys_enter *ctx)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct syscall_entry se = {.enter_ns = bpf_ktime_get_ns(), .nr = SYS_epoll_wait};
    bpf_map_update_elem(&syscall_scratch, &pid, &se, BPF_ANY);
    return 0;
}

SEC("tp/syscalls/sys_enter_ppoll")
int handle_sys_enter_ppoll(struct trace_event_raw_sys_enter *ctx)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct syscall_entry se = {.enter_ns = bpf_ktime_get_ns(), .nr = SYS_ppoll};
    bpf_map_update_elem(&syscall_scratch, &pid, &se, BPF_ANY);
    return 0;
}

/* ── unified exit handler macro ─────────────────────────────────────────── */
#define HANDLE_SYS_EXIT(tracepoint_name)                                        \
    SEC("tp/syscalls/" #tracepoint_name)                                        \
    int handle_##tracepoint_name(struct trace_event_raw_sys_exit *ctx)          \
    {                                                                           \
        __u64 now = bpf_ktime_get_ns();                                         \
        __u32 pid = bpf_get_current_pid_tgid() >> 32;                           \
        __u32 cpu = bpf_get_smp_processor_id();                                 \
                                                                                \
        struct syscall_entry *se = bpf_map_lookup_elem(&syscall_scratch, &pid); \
        if (!se)                                                                \
            return 0;                                                           \
        __u64 lat = (now > se->enter_ns) ? now - se->enter_ns : 0;              \
        __u64 nr = se->nr;                                                      \
        bpf_map_delete_elem(&syscall_scratch, &pid);                            \
                                                                                \
        struct agg_key k = {.pid = pid, .cpu = cpu};                            \
        struct syscall_val *v = bpf_map_lookup_elem(&syscall_map, &k);          \
        if (!v)                                                                 \
        {                                                                       \
            struct syscall_val zero = {};                                       \
            bpf_map_update_elem(&syscall_map, &k, &zero, BPF_NOEXIST);          \
            v = bpf_map_lookup_elem(&syscall_map, &k);                          \
        }                                                                       \
        if (!v)                                                                 \
            return 0;                                                           \
                                                                                \
        v->total_count++;                                                       \
        v->latency_sum_ns += lat;                                               \
        if (lat > v->latency_max_ns)                                            \
            v->latency_max_ns = lat;                                            \
        if (ctx->ret < 0)                                                       \
            v->error_count++;                                                   \
        v->last_update_ns = now;                                                \
                                                                                \
        if (nr == SYS_read)                                                     \
        {                                                                       \
            v->read_count++;                                                    \
            if (ctx->ret > 0)                                                   \
                v->read_bytes += (__u64)ctx->ret;                               \
        }                                                                       \
        if (nr == SYS_write)                                                    \
        {                                                                       \
            v->write_count++;                                                   \
            if (ctx->ret > 0)                                                   \
                v->write_bytes += (__u64)ctx->ret;                              \
        }                                                                       \
        if (nr == SYS_mmap)                                                     \
            v->mmap_count++;                                                    \
        if (nr == SYS_futex)                                                    \
        {                                                                       \
            v->futex_count++;                                                   \
            v->futex_latency_sum_ns += lat;                                     \
        }                                                                       \
        if (nr == SYS_epoll_wait)                                               \
        {                                                                       \
            v->epoll_count++;                                                   \
            v->epoll_latency_sum_ns += lat;                                     \
        }                                                                       \
        if (nr == SYS_ppoll)                                                    \
            v->poll_count++;                                                    \
        return 0;                                                               \
    }

HANDLE_SYS_EXIT(sys_exit_read)
HANDLE_SYS_EXIT(sys_exit_write)
HANDLE_SYS_EXIT(sys_exit_mmap)
HANDLE_SYS_EXIT(sys_exit_futex)
HANDLE_SYS_EXIT(sys_exit_epoll_wait)
HANDLE_SYS_EXIT(sys_exit_ppoll)

/* ══════════════════════════════════════════════════════════════════════════════
 * Mutex / rwsem contention — lock pressure "victim" signals
 *
 * lock_acquire fires when a task tries to acquire a lock.
 * lock_acquired fires when it succeeds (latency = acquired_ns - acquire_ns).
 *
 * We use the lock address as a transient key in wakeup_scratch (reusing it
 * to avoid another map) — actually we use a separate treatment:
 * store (try_ns, lock_type) in a scratch map keyed by pid.
 *
 * For simplicity we track:
 *   - mutex_contentions when the task had to wait (lock was contended)
 *   - rwsem_read / rwsem_write contentions similarly
 *
 * The kernel lockdep tracepoints (lock_acquire / lock_acquired / lock_released)
 * require CONFIG_LOCKDEP=y.  If not available, these probes will fail to
 * attach — pinner.c logs the failure and continues; other maps still work.
 * ══════════════════════════════════════════════════════════════════════════════ */

struct lock_scratch_entry
{
    __u64 try_ns;
    __u8 is_read; /* 1 = rwsem read, 2 = rwsem write, 0 = mutex */
};

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, __u32);
    __type(value, struct lock_scratch_entry);
} lock_scratch SEC(".maps");

/* ══════════════════════════════════════════════════════════════════════════════
 * Mutex / rwsem contention via Kprobes (Works without CONFIG_LOCKDEP)
 * ══════════════════════════════════════════════════════════════════════════════ */

/* 1. Mutexes */
SEC("kprobe/mutex_lock")
int BPF_KPROBE(handle_mutex_lock_enter)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct lock_scratch_entry le = {
        .try_ns = bpf_ktime_get_ns(),
        .is_read = 0, /* 0 = mutex */
    };
    bpf_map_update_elem(&lock_scratch, &pid, &le, BPF_ANY);
    return 0;
}

SEC("kretprobe/mutex_lock")
int BPF_KRETPROBE(handle_mutex_lock_exit)
{
    __u64 now = bpf_ktime_get_ns();
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    __u32 cpu = bpf_get_smp_processor_id();

    struct lock_scratch_entry *le = bpf_map_lookup_elem(&lock_scratch, &pid);
    if (!le || le->try_ns == 0 || le->is_read != 0)
        return 0;

    __u64 wait = (now > le->try_ns) ? now - le->try_ns : 0;
    bpf_map_delete_elem(&lock_scratch, &pid);

    if (wait == 0)
        return 0;

    struct agg_key k = {.pid = pid, .cpu = cpu};
    struct lock_val *v = bpf_map_lookup_elem(&lock_map, &k);
    if (!v)
    {
        struct lock_val zero = {};
        bpf_map_update_elem(&lock_map, &k, &zero, BPF_NOEXIST);
        v = bpf_map_lookup_elem(&lock_map, &k);
    }
    if (v)
    {
        v->mutex_contentions++;
        v->mutex_wait_sum_ns += wait;
        if (wait > v->mutex_wait_max_ns)
            v->mutex_wait_max_ns = wait;
        v->last_update_ns = now;
    }
    return 0;
}

/* 2. Read-Write Semaphores (Readers) */
SEC("kprobe/down_read")
int BPF_KPROBE(handle_down_read_enter)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct lock_scratch_entry le = {
        .try_ns = bpf_ktime_get_ns(),
        .is_read = 1, /* 1 = rwsem read */
    };
    bpf_map_update_elem(&lock_scratch, &pid, &le, BPF_ANY);
    return 0;
}

SEC("kretprobe/down_read")
int BPF_KRETPROBE(handle_down_read_exit)
{
    __u64 now = bpf_ktime_get_ns();
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    __u32 cpu = bpf_get_smp_processor_id();

    struct lock_scratch_entry *le = bpf_map_lookup_elem(&lock_scratch, &pid);
    if (!le || le->try_ns == 0 || le->is_read != 1)
        return 0;

    __u64 wait = (now > le->try_ns) ? now - le->try_ns : 0;
    bpf_map_delete_elem(&lock_scratch, &pid);

    if (wait == 0)
        return 0;

    struct agg_key k = {.pid = pid, .cpu = cpu};
    struct lock_val *v = bpf_map_lookup_elem(&lock_map, &k);
    if (!v)
    {
        struct lock_val zero = {};
        bpf_map_update_elem(&lock_map, &k, &zero, BPF_NOEXIST);
        v = bpf_map_lookup_elem(&lock_map, &k);
    }
    if (v)
    {
        v->rwsem_read_contentions++;
        v->rwsem_read_wait_sum_ns += wait;
        v->last_update_ns = now;
    }
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════════
 * Process exit — clean up scratch entries to avoid stale data
 * ══════════════════════════════════════════════════════════════════════════════ */
SEC("tp/sched/sched_process_exit")
int handle_exit(struct trace_event_raw_sched_process_template *ctx)
{
    __u32 pid = ctx->pid;
    bpf_map_delete_elem(&wakeup_scratch, &pid);
    bpf_map_delete_elem(&syscall_scratch, &pid);
    bpf_map_delete_elem(&oncpu_scratch, &pid);
    bpf_map_delete_elem(&lock_scratch, &pid);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";