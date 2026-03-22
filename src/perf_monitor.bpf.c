// perf_monitor.bpf.c  — ecc-compatible version
//
// Fix: handlers that use structs not in ecc's bundled vmlinux headers are
// rewritten using BPF_PROG() typed tracepoints, which pull fields from BTF
// at load time and never need the struct definition at compile time.
//
// Affected handlers (FIXED):
//   sched_wakeup / sched_wakeup_new  → tp_btf + BPF_PROG
//   sched_migrate_task               → tp_btf + BPF_PROG
//   page_fault_user / kernel         → void* ctx (no fields accessed)
//   kmalloc / kfree                  → tp_btf + BPF_PROG
//   mm_page_alloc                    → tp_btf + BPF_PROG
//
// Handlers that were already fine (kept as-is):
//   sched_switch    → trace_event_raw_sched_switch   (in ecc vmlinux)
//   sys_enter_*     → trace_event_raw_sys_enter       (in ecc vmlinux)
//   sys_exit_*      → trace_event_raw_sys_exit        (in ecc vmlinux)
//   kprobe/*        → pt_regs                         (always fine)
//
// Build:
//   ecc perf_monitor.bpf.c

#include "vmlinux.h"
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>
#include <bpf/bpf_core_read.h>

#define TASK_COMM_LEN 16
#define MAX_ENTRIES 65536

// ─── Shared map key ───────────────────────────────────────────────────────────
struct agg_key
{
    u32 pid;
    u32 cpu;
};

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 1 — CPU SCHEDULING
// ═══════════════════════════════════════════════════════════════════════════════

struct sched_val
{
    u64 ctx_switches;
    u64 voluntary_switches;
    u64 involuntary_switches;
    u64 cpu_migrations;
    u64 total_runtime_ns;
    u64 runq_latency_sum_ns;
    u64 runq_latency_count;
    u64 runq_latency_max_ns;
    u64 last_seen_ns;
    char comm[TASK_COMM_LEN];
};

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, struct agg_key);
    __type(value, struct sched_val);
    __uint(pinning, LIBBPF_PIN_BY_NAME); // auto-pins to /sys/fs/bpf/sched_map
} sched_map SEC(".maps");

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, u32);
    __type(value, u64);
} wakeup_ts SEC(".maps");

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, u32);
    __type(value, u64);
} run_start SEC(".maps");

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, u32);
    __type(value, u32);
} last_cpu SEC(".maps");

#define TASK_RUNNING 0

// sched_switch: trace_event_raw_sched_switch is present in ecc's vmlinux
SEC("tp/sched/sched_switch")
int handle_sched_switch(struct trace_event_raw_sched_switch *ctx)
{
    u64 ts = bpf_ktime_get_ns();
    u32 prev_pid = ctx->prev_pid;
    u32 next_pid = ctx->next_pid;
    u32 cpu = bpf_get_smp_processor_id();

    // outgoing task: measure how long it ran
    u64 *rs = bpf_map_lookup_elem(&run_start, &prev_pid);
    u64 runtime = rs ? (ts - *rs) : 0;
    if (rs)
        bpf_map_delete_elem(&run_start, &prev_pid);

    struct agg_key prev_key = {.pid = prev_pid, .cpu = cpu};
    struct sched_val *sv = bpf_map_lookup_elem(&sched_map, &prev_key);
    if (!sv)
    {
        static struct sched_val zero_sv; // static = BSS, not stack
        bpf_map_update_elem(&sched_map, &prev_key, &zero_sv, BPF_NOEXIST);
        sv = bpf_map_lookup_elem(&sched_map, &prev_key);
    }
    if (sv)
    {
        sv->ctx_switches++;
        sv->total_runtime_ns += runtime;
        sv->last_seen_ns = ts;
        if (ctx->prev_state == TASK_RUNNING)
            sv->involuntary_switches++;
        else
            sv->voluntary_switches++;
        bpf_probe_read_kernel(sv->comm, TASK_COMM_LEN, ctx->prev_comm);
    }

    // incoming task: record schedule-in time, compute runq latency
    bpf_map_update_elem(&run_start, &next_pid, &ts, BPF_ANY);

    u64 runq_lat = 0;
    u64 *wt = bpf_map_lookup_elem(&wakeup_ts, &next_pid);
    if (wt)
    {
        runq_lat = ts - *wt;
        bpf_map_delete_elem(&wakeup_ts, &next_pid);
    }

    u32 migrated = 0;
    u32 *lc = bpf_map_lookup_elem(&last_cpu, &next_pid);
    if (lc && *lc != cpu)
        migrated = 1;
    bpf_map_update_elem(&last_cpu, &next_pid, &cpu, BPF_ANY);

    struct agg_key next_key = {.pid = next_pid, .cpu = cpu};
    struct sched_val *sv2 = bpf_map_lookup_elem(&sched_map, &next_key);
    if (!sv2)
    {
        static struct sched_val zero_sv2;
        bpf_map_update_elem(&sched_map, &next_key, &zero_sv2, BPF_NOEXIST);
        sv2 = bpf_map_lookup_elem(&sched_map, &next_key);
    }
    if (sv2)
    {
        if (migrated)
            sv2->cpu_migrations++;
        if (runq_lat > 0)
        {
            sv2->runq_latency_sum_ns += runq_lat;
            sv2->runq_latency_count++;
            if (runq_lat > sv2->runq_latency_max_ns)
                sv2->runq_latency_max_ns = runq_lat;
        }
        sv2->last_seen_ns = ts;
        bpf_probe_read_kernel(sv2->comm, TASK_COMM_LEN, ctx->next_comm);
    }

    return 0;
}

// FIX: use tp_btf + BPF_PROG so the task_struct arg is resolved from BTF,
// no trace_event_raw_sched_wakeup struct needed at compile time.
SEC("tp_btf/sched_wakeup")
int BPF_PROG(handle_sched_wakeup, struct task_struct *p)
{
    u64 ts = bpf_ktime_get_ns();
    u32 pid = BPF_CORE_READ(p, pid);
    bpf_map_update_elem(&wakeup_ts, &pid, &ts, BPF_ANY);
    return 0;
}

SEC("tp_btf/sched_wakeup_new")
int BPF_PROG(handle_sched_wakeup_new, struct task_struct *p)
{
    u64 ts = bpf_ktime_get_ns();
    u32 pid = BPF_CORE_READ(p, pid);
    bpf_map_update_elem(&wakeup_ts, &pid, &ts, BPF_ANY);
    return 0;
}

SEC("tp_btf/sched_migrate_task")
int BPF_PROG(handle_migrate, struct task_struct *p, int dest_cpu)
{
    u32 pid = BPF_CORE_READ(p, pid);
    u32 dcpu = (u32)dest_cpu;
    bpf_map_update_elem(&last_cpu, &pid, &dcpu, BPF_ANY);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 2 — MEMORY
// ═══════════════════════════════════════════════════════════════════════════════

struct mem_val
{
    u64 minor_faults;
    u64 kernel_faults;
    u64 kmalloc_count;
    u64 kfree_count;
    u64 total_alloc_bytes;
    u64 total_free_bytes;
    u64 large_page_allocs;
    u64 last_seen_ns;
};

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, struct agg_key);
    __type(value, struct mem_val);
    __uint(pinning, LIBBPF_PIN_BY_NAME); // auto-pins to /sys/fs/bpf/mem_map
} mem_map SEC(".maps");

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, u64);
    __type(value, u64);
} alloc_sizes SEC(".maps");

static __always_inline struct mem_val *get_mem_val(u32 pid, u32 cpu)
{
    struct agg_key k = {.pid = pid, .cpu = cpu};
    struct mem_val *v = bpf_map_lookup_elem(&mem_map, &k);
    if (!v)
    {
        static struct mem_val zero_mv;
        bpf_map_update_elem(&mem_map, &k, &zero_mv, BPF_NOEXIST);
        v = bpf_map_lookup_elem(&mem_map, &k);
    }
    return v;
}

// FIX: page fault handlers don't need any fields from ctx,
// so use void* — completely sidesteps the missing struct.
SEC("tp/exceptions/page_fault_user")
int handle_pf_user(void *ctx)
{
    u64 ts = bpf_ktime_get_ns();
    u64 ptg = bpf_get_current_pid_tgid();
    u32 pid = ptg >> 32;
    u32 cpu = bpf_get_smp_processor_id();
    struct mem_val *v = get_mem_val(pid, cpu);
    if (v)
    {
        v->minor_faults++;
        v->last_seen_ns = ts;
    }
    return 0;
}

SEC("tp/exceptions/page_fault_kernel")
int handle_pf_kernel(void *ctx)
{
    u64 ts = bpf_ktime_get_ns();
    u64 ptg = bpf_get_current_pid_tgid();
    u32 pid = ptg >> 32;
    u32 cpu = bpf_get_smp_processor_id();
    struct mem_val *v = get_mem_val(pid, cpu);
    if (v)
    {
        v->kernel_faults++;
        v->last_seen_ns = ts;
    }
    return 0;
}

// FIX: kmalloc/kfree — tp_btf resolves args from BTF, no raw struct needed.
//
// Kernel tracepoint signature for kmalloc:
//   (unsigned long call_site, const void *ptr,
//    size_t bytes_req, size_t bytes_alloc, gfp_t gfp_flags)
//
// NOTE: kernel 6.x changed this signature. If ecc fails here, see the
// #if block at the bottom of this comment.
SEC("tp_btf/kmalloc")
int BPF_PROG(handle_kmalloc,
             unsigned long call_site,
             const void *ptr,
             size_t bytes_req,
             size_t bytes_alloc,
             gfp_t gfp_flags)
{
    u64 ts = bpf_ktime_get_ns();
    u64 ptg = bpf_get_current_pid_tgid();
    u32 pid = ptg >> 32;
    u32 cpu = bpf_get_smp_processor_id();
    u64 addr = (u64)ptr;
    u64 size = (u64)bytes_alloc;

    bpf_map_update_elem(&alloc_sizes, &addr, &size, BPF_ANY);

    struct mem_val *v = get_mem_val(pid, cpu);
    if (v)
    {
        v->kmalloc_count++;
        v->total_alloc_bytes += size;
        v->last_seen_ns = ts;
    }
    return 0;
}

// kfree signature: (const void *ptr)
SEC("tp_btf/kfree")
int BPF_PROG(handle_kfree, const void *ptr)
{
    u64 ts = bpf_ktime_get_ns();
    u64 ptg = bpf_get_current_pid_tgid();
    u32 pid = ptg >> 32;
    u32 cpu = bpf_get_smp_processor_id();
    u64 addr = (u64)ptr;

    u64 *sz = bpf_map_lookup_elem(&alloc_sizes, &addr);
    u64 size = sz ? *sz : 0;
    if (sz)
        bpf_map_delete_elem(&alloc_sizes, &addr);

    struct mem_val *v = get_mem_val(pid, cpu);
    if (v)
    {
        v->kfree_count++;
        v->total_free_bytes += size;
        v->last_seen_ns = ts;
    }
    return 0;
}

// mm_page_alloc signature:
//   (struct page *page, unsigned int order, gfp_t gfp_flags, int migratetype)
SEC("tp_btf/mm_page_alloc")
int BPF_PROG(handle_page_alloc,
             struct page *page,
             unsigned int order,
             gfp_t gfp_flags,
             int migratetype)
{
    if (order == 0)
        return 0; // ignore single-page allocs

    u64 ts = bpf_ktime_get_ns();
    u64 ptg = bpf_get_current_pid_tgid();
    u32 pid = ptg >> 32;
    u32 cpu = bpf_get_smp_processor_id();
    struct mem_val *v = get_mem_val(pid, cpu);
    if (v)
    {
        v->large_page_allocs++;
        v->last_seen_ns = ts;
    }
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 3 — SYSCALL LATENCY
// trace_event_raw_sys_enter/exit are in ecc's vmlinux — unchanged
// ═══════════════════════════════════════════════════════════════════════════════

struct syscall_val
{
    u64 total_count;
    u64 latency_sum_ns;
    u64 latency_max_ns;
    u64 read_count;
    u64 write_count;
    u64 read_bytes;
    u64 write_bytes;
    u64 mmap_count;
    u64 futex_count;
    u64 futex_latency_sum_ns;
    u64 epoll_count;
    u64 epoll_latency_sum_ns;
    u64 poll_count;
    u64 error_count;
    u64 last_seen_ns;
};

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, struct agg_key);
    __type(value, struct syscall_val);
    __uint(pinning, LIBBPF_PIN_BY_NAME); // auto-pins to /sys/fs/bpf/syscall_map
} syscall_map SEC(".maps");

struct sc_entry
{
    u64 ts;
    u64 arg0;
    u64 arg1;
};

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, u64);
    __type(value, struct sc_entry);
} sc_enter SEC(".maps");

#define SC_READ 0
#define SC_WRITE 1
#define SC_MMAP 9
#define SC_FUTEX 202
#define SC_EPOLL_WAIT 281
#define SC_POLL 7

static __always_inline void sc_enter_fn(u64 id, u64 a0, u64 a1)
{
    u64 ptg = bpf_get_current_pid_tgid();
    u32 pid = ptg >> 32;
    u64 key = ((u64)pid << 32) | (id & 0xFFFFFFFF);
    struct sc_entry e = {.ts = bpf_ktime_get_ns(), .arg0 = a0, .arg1 = a1};
    bpf_map_update_elem(&sc_enter, &key, &e, BPF_ANY);
}

static __always_inline void sc_exit_fn(u64 id, s64 ret)
{
    u64 ts = bpf_ktime_get_ns();
    u64 ptg = bpf_get_current_pid_tgid();
    u32 pid = ptg >> 32;
    u32 cpu = bpf_get_smp_processor_id();
    u64 key = ((u64)pid << 32) | (id & 0xFFFFFFFF);

    struct sc_entry *e = bpf_map_lookup_elem(&sc_enter, &key);
    if (!e)
        return;
    u64 lat = ts - e->ts;
    bpf_map_delete_elem(&sc_enter, &key);

    struct agg_key ak = {.pid = pid, .cpu = cpu};
    struct syscall_val *v = bpf_map_lookup_elem(&syscall_map, &ak);
    if (!v)
    {
        static struct syscall_val zero_scv;
        bpf_map_update_elem(&syscall_map, &ak, &zero_scv, BPF_NOEXIST);
        v = bpf_map_lookup_elem(&syscall_map, &ak);
    }
    if (!v)
        return;

    v->total_count++;
    v->latency_sum_ns += lat;
    if (lat > v->latency_max_ns)
        v->latency_max_ns = lat;
    if (ret < 0)
        v->error_count++;
    v->last_seen_ns = ts;

    if (id == SC_READ)
    {
        v->read_count++;
        v->read_bytes += (ret > 0 ? (u64)ret : 0);
    }
    if (id == SC_WRITE)
    {
        v->write_count++;
        v->write_bytes += (ret > 0 ? (u64)ret : 0);
    }
    if (id == SC_MMAP)
    {
        v->mmap_count++;
    }
    if (id == SC_FUTEX)
    {
        v->futex_count++;
        v->futex_latency_sum_ns += lat;
    }
    if (id == SC_EPOLL_WAIT)
    {
        v->epoll_count++;
        v->epoll_latency_sum_ns += lat;
    }
    if (id == SC_POLL)
    {
        v->poll_count++;
    }
}

SEC("tp/syscalls/sys_enter_read")
int enter_read(struct trace_event_raw_sys_enter *ctx)
{
    sc_enter_fn(SC_READ, ctx->args[0], ctx->args[2]);
    return 0;
}
SEC("tp/syscalls/sys_exit_read")
int exit_read(struct trace_event_raw_sys_exit *ctx)
{
    sc_exit_fn(SC_READ, ctx->ret);
    return 0;
}
SEC("tp/syscalls/sys_enter_write")
int enter_write(struct trace_event_raw_sys_enter *ctx)
{
    sc_enter_fn(SC_WRITE, ctx->args[0], ctx->args[2]);
    return 0;
}
SEC("tp/syscalls/sys_exit_write")
int exit_write(struct trace_event_raw_sys_exit *ctx)
{
    sc_exit_fn(SC_WRITE, ctx->ret);
    return 0;
}
SEC("tp/syscalls/sys_enter_mmap")
int enter_mmap(struct trace_event_raw_sys_enter *ctx)
{
    sc_enter_fn(SC_MMAP, ctx->args[1], ctx->args[2]);
    return 0;
}
SEC("tp/syscalls/sys_exit_mmap")
int exit_mmap(struct trace_event_raw_sys_exit *ctx)
{
    sc_exit_fn(SC_MMAP, ctx->ret);
    return 0;
}
SEC("tp/syscalls/sys_enter_futex")
int enter_futex(struct trace_event_raw_sys_enter *ctx)
{
    sc_enter_fn(SC_FUTEX, ctx->args[1], 0);
    return 0;
}
SEC("tp/syscalls/sys_exit_futex")
int exit_futex(struct trace_event_raw_sys_exit *ctx)
{
    sc_exit_fn(SC_FUTEX, ctx->ret);
    return 0;
}
SEC("tp/syscalls/sys_enter_epoll_wait")
int enter_epoll(struct trace_event_raw_sys_enter *ctx)
{
    sc_enter_fn(SC_EPOLL_WAIT, 0, 0);
    return 0;
}
SEC("tp/syscalls/sys_exit_epoll_wait")
int exit_epoll(struct trace_event_raw_sys_exit *ctx)
{
    sc_exit_fn(SC_EPOLL_WAIT, ctx->ret);
    return 0;
}
SEC("tp/syscalls/sys_enter_poll")
int enter_poll(struct trace_event_raw_sys_enter *ctx)
{
    sc_enter_fn(SC_POLL, 0, 0);
    return 0;
}
SEC("tp/syscalls/sys_exit_poll")
int exit_poll(struct trace_event_raw_sys_exit *ctx)
{
    sc_exit_fn(SC_POLL, ctx->ret);
    return 0;
}

// ═══════════════════════════════════════════════════════════════════════════════
// SECTION 4 — LOCK CONTENTION (kprobes — pt_regs always fine)
// ═══════════════════════════════════════════════════════════════════════════════

struct lock_val
{
    u64 mutex_contentions;
    u64 mutex_wait_sum_ns;
    u64 mutex_wait_max_ns;
    u64 rwsem_read_contentions;
    u64 rwsem_read_wait_sum_ns;
    u64 rwsem_write_contentions;
    u64 rwsem_write_wait_sum_ns;
    u64 rwsem_write_wait_max_ns;
    u64 last_seen_ns;
};

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, struct agg_key);
    __type(value, struct lock_val);
    __uint(pinning, LIBBPF_PIN_BY_NAME); // auto-pins to /sys/fs/bpf/lock_map
} lock_map SEC(".maps");

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, u32);
    __type(value, u64);
} lock_enter SEC(".maps");

struct
{
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, u32);
    __type(value, u8);
} lock_type_scratch SEC(".maps");

static __always_inline void lock_enter_fn(u8 ltype)
{
    u64 ts = bpf_ktime_get_ns();
    u64 ptg = bpf_get_current_pid_tgid();
    u32 pid = ptg >> 32;
    bpf_map_update_elem(&lock_enter, &pid, &ts, BPF_ANY);
    bpf_map_update_elem(&lock_type_scratch, &pid, &ltype, BPF_ANY);
}

static __always_inline void lock_exit_fn(void)
{
    u64 ts = bpf_ktime_get_ns();
    u64 ptg = bpf_get_current_pid_tgid();
    u32 pid = ptg >> 32;
    u32 cpu = bpf_get_smp_processor_id();

    u64 *enter = bpf_map_lookup_elem(&lock_enter, &pid);
    if (!enter)
        return;
    u64 wait = ts - *enter;
    bpf_map_delete_elem(&lock_enter, &pid);

    u8 *lt = bpf_map_lookup_elem(&lock_type_scratch, &pid);
    u8 ltype = lt ? *lt : 0;
    if (lt)
        bpf_map_delete_elem(&lock_type_scratch, &pid);

    struct agg_key ak = {.pid = pid, .cpu = cpu};
    struct lock_val *v = bpf_map_lookup_elem(&lock_map, &ak);
    if (!v)
    {
        static struct lock_val zero_lv;
        bpf_map_update_elem(&lock_map, &ak, &zero_lv, BPF_NOEXIST);
        v = bpf_map_lookup_elem(&lock_map, &ak);
    }
    if (!v)
        return;

    v->last_seen_ns = ts;
    if (ltype == 0)
    {
        v->mutex_contentions++;
        v->mutex_wait_sum_ns += wait;
        if (wait > v->mutex_wait_max_ns)
            v->mutex_wait_max_ns = wait;
    }
    else if (ltype == 1)
    {
        v->rwsem_read_contentions++;
        v->rwsem_read_wait_sum_ns += wait;
    }
    else
    {
        v->rwsem_write_contentions++;
        v->rwsem_write_wait_sum_ns += wait;
        if (wait > v->rwsem_write_wait_max_ns)
            v->rwsem_write_wait_max_ns = wait;
    }
}

// Verify symbol names on your kernel:
//   grep -E "mutex_lock_slow|rwsem_down" /proc/kallsyms | head
// Substitute the exact names if different.
SEC("kprobe/__mutex_lock_slowpath")
int probe_mutex_enter(struct pt_regs *ctx)
{
    lock_enter_fn(0);
    return 0;
}

SEC("kretprobe/__mutex_lock_slowpath")
int probe_mutex_exit(struct pt_regs *ctx)
{
    lock_exit_fn();
    return 0;
}

SEC("kprobe/rwsem_down_read_slowpath")
int probe_rwsem_r_enter(struct pt_regs *ctx)
{
    lock_enter_fn(1);
    return 0;
}

SEC("kretprobe/rwsem_down_read_slowpath")
int probe_rwsem_r_exit(struct pt_regs *ctx)
{
    lock_exit_fn();
    return 0;
}

SEC("kprobe/rwsem_down_write_slowpath")
int probe_rwsem_w_enter(struct pt_regs *ctx)
{
    lock_enter_fn(2);
    return 0;
}

SEC("kretprobe/rwsem_down_write_slowpath")
int probe_rwsem_w_exit(struct pt_regs *ctx)
{
    lock_exit_fn();
    return 0;
}

char LICENSE[] SEC("license") = "GPL";