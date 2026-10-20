// SPDX-License-Identifier: GPL-2.0
/*
 * Realtime collector for the v7.2 SHAP-reduced classifier and attribution MLP.
 * It records raw counters from which realtime model inputs are derived.
 */
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wmissing-declarations"
#include "../approach3/vmlinux.h"
#pragma clang diagnostic pop

#include <bpf/bpf_core_read.h>
#include <bpf/bpf_helpers.h>
#include <bpf/bpf_tracing.h>

#define MAX_ENTRIES 65536
#define TASK_COMM_LEN 16
#define MAX_LOCK_NESTING 4
#define TASK_INTERRUPTIBLE 0x0001
#define TASK_UNINTERRUPTIBLE 0x0002

struct agg_key {
    __u32 pid;
    __u32 cpu;
};

struct sched_value {
    __u64 ctx_switches;
    __u64 voluntary_switches;
    __u64 involuntary_switches;
    __u64 cpu_migrations;
    __u64 total_runtime_ns;
    __u64 stall_ns;
    __u64 max_stall_ns;
    __u64 latency_count;
    /* Match the v7 training collector: mean integer runqueue proxy per key. */
    __u64 runq_ratio_sum;
    __u64 runq_ratio_count;
    __u64 last_update_ns;
    char comm[TASK_COMM_LEN];
};

struct mem_value {
    __u64 minor_faults;
    __u64 major_faults;
    __u64 kmalloc_count;
    __u64 kfree_count;
    __u64 total_alloc_bytes;
    __u64 last_update_ns;
};

struct syscall_value {
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

struct lock_value {
    __u64 mutex_contentions;
    __u64 mutex_wait_sum_ns;
    __u64 mutex_wait_max_ns;
    __u64 rwsem_read_contentions;
    __u64 rwsem_read_wait_sum_ns;
    __u64 rwsem_write_contentions;
    __u64 last_update_ns;
};

struct oncpu_value {
    __u64 start_ns;
};

struct syscall_entry {
    __u64 enter_ns;
    __u64 nr;
};

struct lock_key {
    __u32 tid;
    __u8 kind;
    __u8 pad[3];
};

struct lock_entry {
    __u64 start_ns[MAX_LOCK_NESTING];
    __u32 depth;
    __u32 overflow_depth;
};

struct wakeup_entry {
    __u64 wakeup_ns;
};

#define HASH_MAP(name, value_type)                  \
    struct {                                        \
        __uint(type, BPF_MAP_TYPE_HASH);            \
        __uint(max_entries, MAX_ENTRIES);           \
        __type(key, struct agg_key);                \
        __type(value, struct value_type);           \
        __uint(pinning, LIBBPF_PIN_BY_NAME);        \
    } name SEC(".maps")

HASH_MAP(sched_map, sched_value);
HASH_MAP(mem_map, mem_value);
HASH_MAP(syscall_map, syscall_value);
HASH_MAP(lock_map, lock_value);

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, __u32); /* thread ID */
    __type(value, struct oncpu_value);
} oncpu_scratch SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, __u32);
    __type(value, struct syscall_entry);
} syscall_scratch SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, struct lock_key);
    __type(value, struct lock_entry);
} lock_scratch SEC(".maps");

struct {
    __uint(type, BPF_MAP_TYPE_HASH);
    __uint(max_entries, MAX_ENTRIES);
    __type(key, __u32); /* task ID from scheduler tracepoints */
    __type(value, struct wakeup_entry);
} wakeup_scratch SEC(".maps");

/* Per-CPU runqueue proxy used by the v7 training collector. */
struct {
    __uint(type, BPF_MAP_TYPE_PERCPU_ARRAY);
    __uint(max_entries, 1);
    __type(key, __u32);
    __type(value, __u64);
} runq_len SEC(".maps");

#define GET_OR_CREATE(name, value_type, key_ptr) ({                 \
    struct value_type *value = bpf_map_lookup_elem(&(name), key_ptr); \
    if (!value) {                                                    \
        struct value_type zero = {};                                 \
        bpf_map_update_elem(&(name), key_ptr, &zero, BPF_NOEXIST);   \
        value = bpf_map_lookup_elem(&(name), key_ptr);               \
    }                                                                \
    value;                                                           \
})

SEC("tp/sched/sched_switch")
int handle_switch(struct trace_event_raw_sched_switch *ctx)
{
    __u64 now = bpf_ktime_get_ns();
    __u32 cpu = bpf_get_smp_processor_id();
    __u32 prev_pid = ctx->prev_pid;
    __u32 next_pid = ctx->next_pid;

    /* Preserve the v7 collection definition and its fixed 16-CPU scale. */
    __u32 zero = 0;
    __u64 *runq = bpf_map_lookup_elem(&runq_len, &zero);
    if (runq) {
        if (prev_pid && !(ctx->prev_state & (TASK_INTERRUPTIBLE | TASK_UNINTERRUPTIBLE)))
            __sync_fetch_and_add(runq, 1);
        if (*runq > 0)
            __sync_fetch_and_add(runq, (__u64)-1);
    }
    __u64 runq_ratio = 0;
    if (runq)
        runq_ratio = (*runq * 100) >> 4;

    if (next_pid) {
        struct agg_key key = {.pid = next_pid, .cpu = cpu};
        struct sched_value *value = GET_OR_CREATE(sched_map, sched_value, &key);
        if (value) {
            value->ctx_switches++;
            if (runq) {
                value->runq_ratio_sum += runq_ratio;
                value->runq_ratio_count++;
            }
            value->last_update_ns = now;
            bpf_probe_read_kernel_str(value->comm, sizeof(value->comm), ctx->next_comm);

            struct wakeup_entry *wakeup = bpf_map_lookup_elem(&wakeup_scratch, &next_pid);
            if (wakeup && wakeup->wakeup_ns && now > wakeup->wakeup_ns) {
                __u64 stall_ns = now - wakeup->wakeup_ns;
                value->stall_ns += stall_ns;
                value->latency_count++;
                if (stall_ns > value->max_stall_ns)
                    value->max_stall_ns = stall_ns;
            }
        }
        bpf_map_delete_elem(&wakeup_scratch, &next_pid);
        struct oncpu_value start = {.start_ns = now};
        bpf_map_update_elem(&oncpu_scratch, &next_pid, &start, BPF_ANY);
    }

    if (prev_pid) {
        struct agg_key key = {.pid = prev_pid, .cpu = cpu};
        struct sched_value *value = GET_OR_CREATE(sched_map, sched_value, &key);
        if (value) {
            struct oncpu_value *start = bpf_map_lookup_elem(&oncpu_scratch, &prev_pid);
            if (start && now > start->start_ns)
                value->total_runtime_ns += now - start->start_ns;
            value->last_update_ns = now;
            bpf_probe_read_kernel_str(value->comm, sizeof(value->comm), ctx->prev_comm);
            if (ctx->prev_state & (TASK_INTERRUPTIBLE | TASK_UNINTERRUPTIBLE))
                value->voluntary_switches++;
            else
                value->involuntary_switches++;
        }
        bpf_map_delete_elem(&oncpu_scratch, &prev_pid);
    }
    return 0;
}

SEC("tp/sched/sched_migrate_task")
int handle_migrate(struct trace_event_raw_sched_migrate_task *ctx)
{
    struct agg_key key = {.pid = ctx->pid, .cpu = ctx->dest_cpu};
    struct sched_value *value = GET_OR_CREATE(sched_map, sched_value, &key);
    if (value) {
        value->cpu_migrations++;
        value->last_update_ns = bpf_ktime_get_ns();
    }
    return 0;
}

static __always_inline int record_wakeup(__u32 pid)
{
    if (!pid)
        return 0;

    struct wakeup_entry entry = {.wakeup_ns = bpf_ktime_get_ns()};
    bpf_map_update_elem(&wakeup_scratch, &pid, &entry, BPF_NOEXIST);
    return 0;
}

SEC("tp/sched/sched_wakeup")
int handle_wakeup(struct trace_event_raw_sched_wakeup_template *ctx)
{
    return record_wakeup(ctx->pid);
}

SEC("tp/sched/sched_wakeup_new")
int handle_wakeup_new(struct trace_event_raw_sched_wakeup_template *ctx)
{
    return record_wakeup(ctx->pid);
}

static __always_inline int update_fault(__u64 minor)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct agg_key key = {.pid = pid, .cpu = bpf_get_smp_processor_id()};
    struct mem_value *value = GET_OR_CREATE(mem_map, mem_value, &key);
    if (value) {
        if (minor)
            value->minor_faults++;
        else
            value->major_faults++;
        value->last_update_ns = bpf_ktime_get_ns();
    }
    return 0;
}

SEC("tp/exceptions/page_fault_user")
int handle_page_fault_user(void *ctx)
{
    return update_fault(1);
}

SEC("tp/exceptions/page_fault_kernel")
int handle_page_fault_kernel(void *ctx)
{
    return update_fault(0);
}

SEC("tp/kmem/kmalloc")
int handle_kmalloc(struct trace_event_raw_kmalloc *ctx)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct agg_key key = {.pid = pid, .cpu = bpf_get_smp_processor_id()};
    struct mem_value *value = GET_OR_CREATE(mem_map, mem_value, &key);
    if (value && ctx->ptr) {
        value->kmalloc_count++;
        value->total_alloc_bytes += ctx->bytes_alloc;
        value->last_update_ns = bpf_ktime_get_ns();
    }
    return 0;
}

SEC("tp/kmem/kfree")
int handle_kfree(void *ctx)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct agg_key key = {.pid = pid, .cpu = bpf_get_smp_processor_id()};
    struct mem_value *value = GET_OR_CREATE(mem_map, mem_value, &key);
    if (value) {
        value->kfree_count++;
        value->last_update_ns = bpf_ktime_get_ns();
    }
    return 0;
}

#define SYS_READ 0
#define SYS_WRITE 1
#define SYS_MMAP 9
#define SYS_FUTEX 202
#define SYS_EPOLL_WAIT 232
#define SYS_PPOLL 271
#define SYS_POLL 7

#define HANDLE_SYS_ENTER(name, number)                              \
    SEC("tp/syscalls/sys_enter_" #name)                             \
    int enter_##name(struct trace_event_raw_sys_enter *ctx)         \
    {                                                               \
        __u32 tid = (__u32)bpf_get_current_pid_tgid();              \
        struct syscall_entry entry = {                              \
            .enter_ns = bpf_ktime_get_ns(), .nr = number,           \
        };                                                          \
        bpf_map_update_elem(&syscall_scratch, &tid, &entry, BPF_ANY);\
        return 0;                                                   \
    }

HANDLE_SYS_ENTER(read, SYS_READ)
HANDLE_SYS_ENTER(write, SYS_WRITE)
HANDLE_SYS_ENTER(mmap, SYS_MMAP)
HANDLE_SYS_ENTER(futex, SYS_FUTEX)
HANDLE_SYS_ENTER(epoll_wait, SYS_EPOLL_WAIT)
HANDLE_SYS_ENTER(ppoll, SYS_PPOLL)
HANDLE_SYS_ENTER(poll, SYS_POLL)

#define HANDLE_SYS_EXIT(name)                                                  \
    SEC("tp/syscalls/sys_exit_" #name)                                         \
    int exit_##name(struct trace_event_raw_sys_exit *ctx)                      \
    {                                                                          \
        long syscall_ret = ctx->ret;                                           \
        __u64 pid_tgid = bpf_get_current_pid_tgid();                            \
        __u32 pid = pid_tgid >> 32;                                             \
        __u32 tid = (__u32)pid_tgid;                                            \
        struct syscall_entry *entry = bpf_map_lookup_elem(&syscall_scratch, &tid); \
        if (!entry)                                                            \
            return 0;                                                          \
        __u64 number = entry->nr;                                               \
        __u64 now = bpf_ktime_get_ns();                                         \
        __u64 latency = now > entry->enter_ns ? now - entry->enter_ns : 0;      \
        bpf_map_delete_elem(&syscall_scratch, &tid);                            \
        struct agg_key key = {.pid = pid, .cpu = bpf_get_smp_processor_id()};   \
        struct syscall_value *value = GET_OR_CREATE(syscall_map, syscall_value, &key); \
        if (!value)                                                             \
            return 0;                                                           \
        value->total_count++;                                                   \
        value->latency_sum_ns += latency;                                       \
        if (latency > value->latency_max_ns)                                    \
            value->latency_max_ns = latency;                                    \
        if (syscall_ret < 0 && syscall_ret != -512 && syscall_ret != -513 &&    \
            syscall_ret != -514 && syscall_ret != -516)                        \
            value->error_count++;                                               \
        if (number == SYS_READ) {                                               \
            value->read_count++;                                               \
            if (syscall_ret > 0)                                                  \
                value->read_bytes += (__u64)syscall_ret;                           \
        }                                                                       \
        if (number == SYS_WRITE) {                                              \
            value->write_count++;                                              \
            if (syscall_ret > 0)                                                  \
                value->write_bytes += (__u64)syscall_ret;                          \
        }                                                                       \
        if (number == SYS_MMAP)                                                 \
            value->mmap_count++;                                                \
        if (number == SYS_FUTEX) {                                              \
            value->futex_count++;                                               \
            value->futex_latency_sum_ns += latency;                             \
        }                                                                       \
        if (number == SYS_EPOLL_WAIT) {                                         \
            value->epoll_count++;                                               \
            value->epoll_latency_sum_ns += latency;                             \
        }                                                                       \
        if (number == SYS_PPOLL || number == SYS_POLL)                          \
            value->poll_count++;                                                \
        value->last_update_ns = now;                                            \
        return 0;                                                               \
    }

HANDLE_SYS_EXIT(read)
HANDLE_SYS_EXIT(write)
HANDLE_SYS_EXIT(mmap)
HANDLE_SYS_EXIT(futex)
HANDLE_SYS_EXIT(epoll_wait)
HANDLE_SYS_EXIT(ppoll)
HANDLE_SYS_EXIT(poll)

#define LOCK_MUTEX 0
#define LOCK_RWSEM_READ 1
#define LOCK_RWSEM_WRITE 2

static __always_inline int lock_wait_enter(__u8 kind)
{
    __u32 tid = (__u32)bpf_get_current_pid_tgid();
    struct lock_key key = {.tid = tid, .kind = kind};
    struct lock_entry *entry = bpf_map_lookup_elem(&lock_scratch, &key);
    if (!entry) {
        struct lock_entry zero = {};
        bpf_map_update_elem(&lock_scratch, &key, &zero, BPF_NOEXIST);
        entry = bpf_map_lookup_elem(&lock_scratch, &key);
    }
    if (!entry)
        return 0;

    if (entry->overflow_depth || entry->depth >= MAX_LOCK_NESTING) {
        entry->overflow_depth++;
        return 0;
    }

    __u64 now = bpf_ktime_get_ns();
    switch (entry->depth) {
    case 0:
        entry->start_ns[0] = now;
        break;
    case 1:
        entry->start_ns[1] = now;
        break;
    case 2:
        entry->start_ns[2] = now;
        break;
    case 3:
        entry->start_ns[3] = now;
        break;
    default:
        return 0;
    }
    entry->depth++;
    return 0;
}

static __always_inline int lock_wait_exit(__u8 kind)
{
    __u64 pid_tgid = bpf_get_current_pid_tgid();
    __u32 pid = pid_tgid >> 32;
    __u32 tid = (__u32)pid_tgid;
    struct lock_key scratch_key = {.tid = tid, .kind = kind};
    struct lock_entry *entry = bpf_map_lookup_elem(&lock_scratch, &scratch_key);
    if (!entry)
        return 0;
    if (entry->overflow_depth) {
        entry->overflow_depth--;
        return 0;
    }
    if (!entry->depth)
        return 0;

    __u64 start_ns = 0;
    switch (entry->depth) {
    case 1:
        start_ns = entry->start_ns[0];
        break;
    case 2:
        start_ns = entry->start_ns[1];
        break;
    case 3:
        start_ns = entry->start_ns[2];
        break;
    case 4:
        start_ns = entry->start_ns[3];
        break;
    default:
        return 0;
    }
    entry->depth--;
    int scratch_empty = !entry->depth && !entry->overflow_depth;
    if (scratch_empty)
        bpf_map_delete_elem(&lock_scratch, &scratch_key);

    __u64 now = bpf_ktime_get_ns();
    __u64 wait_ns = now > start_ns ? now - start_ns : 0;
    struct agg_key key = {.pid = pid, .cpu = bpf_get_smp_processor_id()};
    struct lock_value *value = GET_OR_CREATE(lock_map, lock_value, &key);
    if (!value)
        return 0;

    if (kind == LOCK_MUTEX) {
        value->mutex_contentions++;
        value->mutex_wait_sum_ns += wait_ns;
        if (wait_ns > value->mutex_wait_max_ns)
            value->mutex_wait_max_ns = wait_ns;
    } else if (kind == LOCK_RWSEM_READ) {
        value->rwsem_read_contentions++;
        value->rwsem_read_wait_sum_ns += wait_ns;
    } else {
        value->rwsem_write_contentions++;
    }
    value->last_update_ns = now;
    return 0;
}

#define HANDLE_LOCK_SLOWPATH(name, lock_kind)                       \
    SEC("kprobe/" #name)                                            \
    int BPF_KPROBE(enter_##name)                                    \
    {                                                               \
        return lock_wait_enter(lock_kind);                          \
    }                                                               \
    SEC("kretprobe/" #name)                                         \
    int BPF_KRETPROBE(exit_##name)                                  \
    {                                                               \
        return lock_wait_exit(lock_kind);                           \
    }

HANDLE_LOCK_SLOWPATH(__mutex_lock_slowpath, LOCK_MUTEX)
HANDLE_LOCK_SLOWPATH(rwsem_down_read_slowpath, LOCK_RWSEM_READ)
HANDLE_LOCK_SLOWPATH(rwsem_down_write_slowpath, LOCK_RWSEM_WRITE)

SEC("tp/sched/sched_process_exit")
int handle_exit(struct trace_event_raw_sched_process_template *ctx)
{
    __u32 pid = ctx->pid;
    struct lock_key mutex_key = {.tid = pid, .kind = LOCK_MUTEX};
    struct lock_key rwsem_read_key = {.tid = pid, .kind = LOCK_RWSEM_READ};
    struct lock_key rwsem_write_key = {.tid = pid, .kind = LOCK_RWSEM_WRITE};
    bpf_map_delete_elem(&oncpu_scratch, &pid);
    bpf_map_delete_elem(&syscall_scratch, &pid);
    bpf_map_delete_elem(&lock_scratch, &mutex_key);
    bpf_map_delete_elem(&lock_scratch, &rwsem_read_key);
    bpf_map_delete_elem(&lock_scratch, &rwsem_write_key);
    bpf_map_delete_elem(&wakeup_scratch, &pid);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
