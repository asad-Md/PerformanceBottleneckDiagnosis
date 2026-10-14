// SPDX-License-Identifier: GPL-2.0
/*
 * Realtime collector for the v7.2 SHAP-reduced classifier and attribution MLP.
 * It records only the raw counters needed to derive their shared 20 inputs.
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
    __u64 last_update_ns;
    char comm[TASK_COMM_LEN];
};

struct mem_value {
    __u64 minor_faults;
    __u64 major_faults;
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
    __u64 last_update_ns;
};

struct lock_value {
    __u64 mutex_contentions;
    __u64 mutex_wait_sum_ns;
    __u64 rwsem_read_contentions;
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

struct lock_entry {
    __u64 start_ns;
    __u8 kind;
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
    __type(key, __u32);
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
    __type(key, __u32);
    __type(value, struct lock_entry);
} lock_scratch SEC(".maps");

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

    if (next_pid) {
        struct agg_key key = {.pid = next_pid, .cpu = cpu};
        struct sched_value *value = GET_OR_CREATE(sched_map, sched_value, &key);
        if (value) {
            value->ctx_switches++;
            value->last_update_ns = now;
            bpf_probe_read_kernel_str(value->comm, sizeof(value->comm), ctx->next_comm);
        }
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

SEC("tp_btf/kmalloc")
int BPF_PROG(handle_kmalloc, unsigned long call_site, const void *ptr,
             size_t bytes_req, size_t bytes_alloc, gfp_t gfp_flags)
{
    __u32 pid = bpf_get_current_pid_tgid() >> 32;
    struct agg_key key = {.pid = pid, .cpu = bpf_get_smp_processor_id()};
    struct mem_value *value = GET_OR_CREATE(mem_map, mem_value, &key);
    if (value) {
        value->total_alloc_bytes += bytes_alloc;
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

#define HANDLE_SYS_ENTER(name, number)                              \
    SEC("tp/syscalls/sys_enter_" #name)                             \
    int enter_##name(struct trace_event_raw_sys_enter *ctx)         \
    {                                                               \
        __u32 pid = bpf_get_current_pid_tgid() >> 32;                \
        struct syscall_entry entry = {                              \
            .enter_ns = bpf_ktime_get_ns(), .nr = number,           \
        };                                                          \
        bpf_map_update_elem(&syscall_scratch, &pid, &entry, BPF_ANY);\
        return 0;                                                   \
    }

HANDLE_SYS_ENTER(read, SYS_READ)
HANDLE_SYS_ENTER(write, SYS_WRITE)
HANDLE_SYS_ENTER(mmap, SYS_MMAP)
HANDLE_SYS_ENTER(futex, SYS_FUTEX)
HANDLE_SYS_ENTER(epoll_wait, SYS_EPOLL_WAIT)
HANDLE_SYS_ENTER(ppoll, SYS_PPOLL)

#define HANDLE_SYS_EXIT(name)                                                  \
    SEC("tp/syscalls/sys_exit_" #name)                                         \
    int exit_##name(struct trace_event_raw_sys_exit *ctx)                      \
    {                                                                          \
        __u32 pid = bpf_get_current_pid_tgid() >> 32;                           \
        struct syscall_entry *entry = bpf_map_lookup_elem(&syscall_scratch, &pid); \
        if (!entry)                                                            \
            return 0;                                                          \
        __u64 number = entry->nr;                                               \
        __u64 now = bpf_ktime_get_ns();                                         \
        __u64 latency = now > entry->enter_ns ? now - entry->enter_ns : 0;      \
        bpf_map_delete_elem(&syscall_scratch, &pid);                            \
        struct agg_key key = {.pid = pid, .cpu = bpf_get_smp_processor_id()};   \
        struct syscall_value *value = GET_OR_CREATE(syscall_map, syscall_value, &key); \
        if (!value)                                                             \
            return 0;                                                           \
        value->total_count++;                                                   \
        value->latency_sum_ns += latency;                                       \
        if (latency > value->latency_max_ns)                                    \
            value->latency_max_ns = latency;                                    \
        if (number == SYS_READ) {                                               \
            value->read_count++;                                               \
            if (ctx->ret > 0)                                                  \
                value->read_bytes += (__u64)ctx->ret;                           \
        }                                                                       \
        if (number == SYS_WRITE) {                                              \
            value->write_count++;                                              \
            if (ctx->ret > 0)                                                  \
                value->write_bytes += (__u64)ctx->ret;                          \
        }                                                                       \
        value->last_update_ns = now;                                            \
        return 0;                                                               \
    }

HANDLE_SYS_EXIT(read)
HANDLE_SYS_EXIT(write)
HANDLE_SYS_EXIT(mmap)
HANDLE_SYS_EXIT(futex)
HANDLE_SYS_EXIT(epoll_wait)
HANDLE_SYS_EXIT(ppoll)

#define LOCK_MUTEX 0
#define LOCK_RWSEM_READ 1
#define LOCK_RWSEM_WRITE 2

#define HANDLE_LOCK_ENTER(name, lock_kind)                           \
    SEC("kprobe/" #name)                                             \
    int BPF_KPROBE(enter_##name)                                     \
    {                                                                \
        __u32 pid = bpf_get_current_pid_tgid() >> 32;                 \
        struct lock_entry entry = {                                  \
            .start_ns = bpf_ktime_get_ns(), .kind = lock_kind,       \
        };                                                           \
        bpf_map_update_elem(&lock_scratch, &pid, &entry, BPF_ANY);    \
        return 0;                                                    \
    }

#define HANDLE_LOCK_EXIT(name, lock_kind)                            \
    SEC("kretprobe/" #name)                                          \
    int BPF_KRETPROBE(exit_##name)                                   \
    {                                                                \
        __u32 pid = bpf_get_current_pid_tgid() >> 32;                 \
        struct lock_entry *entry = bpf_map_lookup_elem(&lock_scratch, &pid); \
        if (!entry || entry->kind != lock_kind)                       \
            return 0;                                                \
        __u64 now = bpf_ktime_get_ns();                               \
        __u64 wait_ns = now > entry->start_ns ? now - entry->start_ns : 0; \
        bpf_map_delete_elem(&lock_scratch, &pid);                     \
        if (!wait_ns)                                                 \
            return 0;                                                \
        struct agg_key key = {.pid = pid, .cpu = bpf_get_smp_processor_id()}; \
        struct lock_value *value = GET_OR_CREATE(lock_map, lock_value, &key); \
        if (value) {                                                  \
            if (lock_kind == LOCK_MUTEX) {                            \
                value->mutex_contentions++;                          \
                value->mutex_wait_sum_ns += wait_ns;                  \
            } else if (lock_kind == LOCK_RWSEM_READ) {                \
                value->rwsem_read_contentions++;                      \
            } else {                                                  \
                value->rwsem_write_contentions++;                     \
            }                                                        \
            value->last_update_ns = now;                              \
        }                                                            \
        return 0;                                                    \
    }

HANDLE_LOCK_ENTER(mutex_lock, LOCK_MUTEX)
HANDLE_LOCK_EXIT(mutex_lock, LOCK_MUTEX)
HANDLE_LOCK_ENTER(down_read, LOCK_RWSEM_READ)
HANDLE_LOCK_EXIT(down_read, LOCK_RWSEM_READ)
HANDLE_LOCK_ENTER(down_write, LOCK_RWSEM_WRITE)
HANDLE_LOCK_EXIT(down_write, LOCK_RWSEM_WRITE)

SEC("tp/sched/sched_process_exit")
int handle_exit(struct trace_event_raw_sched_process_template *ctx)
{
    __u32 pid = ctx->pid;
    bpf_map_delete_elem(&oncpu_scratch, &pid);
    bpf_map_delete_elem(&syscall_scratch, &pid);
    bpf_map_delete_elem(&lock_scratch, &pid);
    return 0;
}

char LICENSE[] SEC("license") = "GPL";
