/*
 * reader.c
 *
 * Opens pinned maps from /sys/fs/bpf/, reads session timestamps from
 * /tmp/perf_session.meta, filters out entries whose last_seen_ns falls
 * in the warmup window (first skip_start seconds) or teardown window
 * (last skip_end seconds), merges all four maps by (pid,cpu), writes CSV.
 *
 * Build:
 *   gcc -O2 -o reader reader.c -lbpf -lelf -lz
 *
 * Run:
 *   sudo ./reader --label cpu_bound
 *   sudo ./reader --label cpu_bound --skip-start 5 --skip-end 2
 *   sudo ./reader --label cpu_bound --append
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <time.h>
#include <unistd.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

#define PIN_BASE "/sys/fs/bpf"
#define META_FILE "/tmp/perf_session.meta"
#define MAX_KEYS 65536
#define TASK_COMM_LEN 16

// ── map types — must match perf_monitor.bpf.c ─────────────────────────────────

struct agg_key
{
    uint32_t pid;
    uint32_t cpu;
};

struct sched_val
{
    uint64_t ctx_switches;
    uint64_t voluntary_switches;
    uint64_t involuntary_switches;
    uint64_t cpu_migrations;
    uint64_t total_runtime_ns;
    uint64_t runq_latency_sum_ns;
    uint64_t runq_latency_count;
    uint64_t runq_latency_max_ns;
    uint64_t last_seen_ns;
    char comm[TASK_COMM_LEN];
};

struct mem_val
{
    uint64_t minor_faults;
    uint64_t kernel_faults;
    uint64_t kmalloc_count;
    uint64_t kfree_count;
    uint64_t total_alloc_bytes;
    uint64_t total_free_bytes;
    uint64_t large_page_allocs;
    uint64_t last_seen_ns;
};

struct syscall_val
{
    uint64_t total_count;
    uint64_t latency_sum_ns;
    uint64_t latency_max_ns;
    uint64_t read_count;
    uint64_t write_count;
    uint64_t read_bytes;
    uint64_t write_bytes;
    uint64_t mmap_count;
    uint64_t futex_count;
    uint64_t futex_latency_sum_ns;
    uint64_t epoll_count;
    uint64_t epoll_latency_sum_ns;
    uint64_t poll_count;
    uint64_t error_count;
    uint64_t last_seen_ns;
};

struct lock_val
{
    uint64_t mutex_contentions;
    uint64_t mutex_wait_sum_ns;
    uint64_t mutex_wait_max_ns;
    uint64_t rwsem_read_contentions;
    uint64_t rwsem_read_wait_sum_ns;
    uint64_t rwsem_write_contentions;
    uint64_t rwsem_write_wait_sum_ns;
    uint64_t rwsem_write_wait_max_ns;
    uint64_t last_seen_ns;
};

#define AVG(sum, count) ((count) > 0 ? (sum) / (count) : (uint64_t)0)

// ── drain a map into arrays ───────────────────────────────────────────────────

#define DEFINE_DRAIN(fname, vtype)                              \
    static int fname(int fd, struct agg_key *keys, vtype *vals) \
    {                                                           \
        struct agg_key prev = {}, next = {};                    \
        struct agg_key *prev_ptr = NULL;                        \
        int n = 0;                                              \
        while (n < MAX_KEYS &&                                  \
               bpf_map_get_next_key(fd, prev_ptr, &next) == 0)  \
        {                                                       \
            vtype v = {};                                       \
            if (bpf_map_lookup_elem(fd, &next, &v) == 0)        \
            {                                                   \
                keys[n] = next;                                 \
                vals[n] = v;                                    \
                n++;                                            \
            }                                                   \
            prev = next;                                        \
            prev_ptr = &prev;                                   \
        }                                                       \
        return n;                                               \
    }

DEFINE_DRAIN(drain_sched, struct sched_val)
DEFINE_DRAIN(drain_mem, struct mem_val)
DEFINE_DRAIN(drain_syscall, struct syscall_val)
DEFINE_DRAIN(drain_lock, struct lock_val)

// ── session metadata ──────────────────────────────────────────────────────────

static int read_session_meta(uint64_t *start_ns, uint64_t *end_ns)
{
    *start_ns = 0;
    *end_ns = 0;

    FILE *f = fopen(META_FILE, "r");
    if (!f)
    {
        fprintf(stderr, "[reader] WARNING: %s not found — no time trimming\n",
                META_FILE);
        return -1;
    }

    char line[128];
    while (fgets(line, sizeof(line), f))
    {
        unsigned long long v;
        if (sscanf(line, "start_ns=%llu", &v) == 1)
            *start_ns = (uint64_t)v;
        if (sscanf(line, "end_ns=%llu", &v) == 1)
            *end_ns = (uint64_t)v;
    }
    fclose(f);

    if (*end_ns == 0)
    {
        fprintf(stderr, "[reader] WARNING: end_ns=0 in meta — pinner may not have stopped cleanly\n");
        // use current time as fallback
        struct timespec ts;
        clock_gettime(CLOCK_REALTIME, &ts);
        *end_ns = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }

    return 0;
}

// ── CSV ───────────────────────────────────────────────────────────────────────

static void write_header(FILE *f)
{
    fprintf(f,
            "timestamp_s,pid,cpu,comm,"
            "ctx_switches,voluntary_switches,involuntary_switches,"
            "cpu_migrations,total_runtime_ns,"
            "avg_runq_latency_ns,max_runq_latency_ns,"
            "minor_faults,kernel_faults,"
            "kmalloc_count,kfree_count,"
            "total_alloc_bytes,total_free_bytes,large_page_allocs,"
            "syscall_count,avg_syscall_latency_ns,max_syscall_latency_ns,"
            "read_count,write_count,read_bytes,write_bytes,"
            "mmap_count,futex_count,avg_futex_latency_ns,"
            "epoll_count,avg_epoll_latency_ns,"
            "poll_count,syscall_error_count,"
            "mutex_contentions,avg_mutex_wait_ns,max_mutex_wait_ns,"
            "rwsem_read_contentions,avg_rwsem_read_wait_ns,"
            "rwsem_write_contentions,avg_rwsem_write_wait_ns,max_rwsem_write_wait_ns,"
            "label\n");
}

static void write_row(FILE *f,
                      double ts_s,
                      const struct agg_key *k,
                      const struct sched_val *sv,
                      const struct mem_val *mv,
                      const struct syscall_val *scv,
                      const struct lock_val *lv,
                      const char *comm,
                      const char *label)
{
    fprintf(f,
            "%.3f,%u,%u,%s,"
            "%llu,%llu,%llu,"
            "%llu,%llu,"
            "%llu,%llu,"
            "%llu,%llu,"
            "%llu,%llu,"
            "%llu,%llu,%llu,"
            "%llu,%llu,%llu,"
            "%llu,%llu,%llu,%llu,"
            "%llu,%llu,%llu,"
            "%llu,%llu,"
            "%llu,%llu,"
            "%llu,%llu,%llu,"
            "%llu,%llu,"
            "%llu,%llu,%llu,"
            "%s\n",

            ts_s, k->pid, k->cpu, comm,

            (unsigned long long)(sv ? sv->ctx_switches : 0),
            (unsigned long long)(sv ? sv->voluntary_switches : 0),
            (unsigned long long)(sv ? sv->involuntary_switches : 0),
            (unsigned long long)(sv ? sv->cpu_migrations : 0),
            (unsigned long long)(sv ? sv->total_runtime_ns : 0),
            (unsigned long long)(sv ? AVG(sv->runq_latency_sum_ns, sv->runq_latency_count) : 0),
            (unsigned long long)(sv ? sv->runq_latency_max_ns : 0),

            (unsigned long long)(mv ? mv->minor_faults : 0),
            (unsigned long long)(mv ? mv->kernel_faults : 0),
            (unsigned long long)(mv ? mv->kmalloc_count : 0),
            (unsigned long long)(mv ? mv->kfree_count : 0),
            (unsigned long long)(mv ? mv->total_alloc_bytes : 0),
            (unsigned long long)(mv ? mv->total_free_bytes : 0),
            (unsigned long long)(mv ? mv->large_page_allocs : 0),

            (unsigned long long)(scv ? scv->total_count : 0),
            (unsigned long long)(scv ? AVG(scv->latency_sum_ns, scv->total_count) : 0),
            (unsigned long long)(scv ? scv->latency_max_ns : 0),
            (unsigned long long)(scv ? scv->read_count : 0),
            (unsigned long long)(scv ? scv->write_count : 0),
            (unsigned long long)(scv ? scv->read_bytes : 0),
            (unsigned long long)(scv ? scv->write_bytes : 0),
            (unsigned long long)(scv ? scv->mmap_count : 0),
            (unsigned long long)(scv ? scv->futex_count : 0),
            (unsigned long long)(scv ? AVG(scv->futex_latency_sum_ns, scv->futex_count) : 0),
            (unsigned long long)(scv ? scv->epoll_count : 0),
            (unsigned long long)(scv ? AVG(scv->epoll_latency_sum_ns, scv->epoll_count) : 0),
            (unsigned long long)(scv ? scv->poll_count : 0),
            (unsigned long long)(scv ? scv->error_count : 0),

            (unsigned long long)(lv ? lv->mutex_contentions : 0),
            (unsigned long long)(lv ? AVG(lv->mutex_wait_sum_ns, lv->mutex_contentions) : 0),
            (unsigned long long)(lv ? lv->mutex_wait_max_ns : 0),
            (unsigned long long)(lv ? lv->rwsem_read_contentions : 0),
            (unsigned long long)(lv ? AVG(lv->rwsem_read_wait_sum_ns, lv->rwsem_read_contentions) : 0),
            (unsigned long long)(lv ? lv->rwsem_write_contentions : 0),
            (unsigned long long)(lv ? AVG(lv->rwsem_write_wait_sum_ns, lv->rwsem_write_contentions) : 0),
            (unsigned long long)(lv ? lv->rwsem_write_wait_max_ns : 0),

            label);
}

// ── main ──────────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
    const char *label = "unknown";
    const char *csv_path = "perf_metrics.csv";
    int append = 0;
    int skip_start = 5; // default: skip first 5 seconds
    int skip_end = 2;   // default: skip last 2 seconds

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--label") && i + 1 < argc)
            label = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)
            csv_path = argv[++i];
        else if (!strcmp(argv[i], "--skip-start") && i + 1 < argc)
            skip_start = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--skip-end") && i + 1 < argc)
            skip_end = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-trim"))
        {
            skip_start = 0;
            skip_end = 0;
        }
        else if (!strcmp(argv[i], "--append"))
            append = 1;
    }

    // ── read session timestamps ───────────────────────────────────────────────
    uint64_t session_start_ns = 0, session_end_ns = 0;
    read_session_meta(&session_start_ns, &session_end_ns);

    // compute trim window in nanoseconds
    uint64_t trim_start_ns = session_start_ns + (uint64_t)skip_start * 1000000000ULL;
    uint64_t trim_end_ns = session_end_ns - (uint64_t)skip_end * 1000000000ULL;

    double session_duration_s = (double)(session_end_ns - session_start_ns) / 1e9;
    fprintf(stderr, "[reader] session duration: %.1fs\n", session_duration_s);
    fprintf(stderr, "[reader] trimming: skip first %ds, skip last %ds\n",
            skip_start, skip_end);
    fprintf(stderr, "[reader] valid window: [+%ds, -%.0fs]\n",
            skip_start, (double)skip_end);

    if (session_start_ns > 0 && trim_end_ns <= trim_start_ns)
    {
        fprintf(stderr, "[reader] WARNING: session too short for trim window "
                        "(%.1fs). Using --no-trim.\n",
                session_duration_s);
        skip_start = 0;
        skip_end = 0;
        trim_start_ns = session_start_ns;
        trim_end_ns = session_end_ns;
    }

    // ── open pinned maps ──────────────────────────────────────────────────────
    const char *map_names[4] = {
        "sched_map", "mem_map", "syscall_map", "lock_map"};
    int fds[4] = {-1, -1, -1, -1};
    char path[128];

    for (int i = 0; i < 4; i++)
    {
        snprintf(path, sizeof(path), "%s/%s", PIN_BASE, map_names[i]);
        fds[i] = bpf_obj_get(path);
        if (fds[i] < 0)
            fprintf(stderr, "[reader] could not open %s: %s\n",
                    path, strerror(errno));
        else
            fprintf(stderr, "[reader] opened %s (fd=%d)\n", map_names[i], fds[i]);
    }

    if (fds[0] < 0 && fds[1] < 0 && fds[2] < 0 && fds[3] < 0)
    {
        fprintf(stderr, "[reader] no maps found — did pinner run?\n");
        return 1;
    }

    // ── drain all maps ────────────────────────────────────────────────────────
    static struct agg_key sched_keys[MAX_KEYS];
    static struct sched_val sched_vals[MAX_KEYS];
    static struct agg_key mem_keys[MAX_KEYS];
    static struct mem_val mem_vals[MAX_KEYS];
    static struct agg_key sc_keys[MAX_KEYS];
    static struct syscall_val sc_vals[MAX_KEYS];
    static struct agg_key lock_keys[MAX_KEYS];
    static struct lock_val lock_vals[MAX_KEYS];

    int ns = fds[0] >= 0 ? drain_sched(fds[0], sched_keys, sched_vals) : 0;
    int nm = fds[1] >= 0 ? drain_mem(fds[1], mem_keys, mem_vals) : 0;
    int nsc = fds[2] >= 0 ? drain_syscall(fds[2], sc_keys, sc_vals) : 0;
    int nl = fds[3] >= 0 ? drain_lock(fds[3], lock_keys, lock_vals) : 0;

    fprintf(stderr, "[reader] raw entries: sched=%d mem=%d syscall=%d lock=%d\n",
            ns, nm, nsc, nl);

    // ── collect unique (pid,cpu) keys ─────────────────────────────────────────
    struct agg_key all_keys[MAX_KEYS];
    int nk = 0;

#define MERGE_KEYS(src_keys, src_n)                       \
    for (int _i = 0; _i < (src_n) && nk < MAX_KEYS; _i++) \
    {                                                     \
        int _found = 0;                                   \
        for (int _j = 0; _j < nk; _j++)                   \
        {                                                 \
            if (all_keys[_j].pid == (src_keys)[_i].pid && \
                all_keys[_j].cpu == (src_keys)[_i].cpu)   \
            {                                             \
                _found = 1;                               \
                break;                                    \
            }                                             \
        }                                                 \
        if (!_found)                                      \
            all_keys[nk++] = (src_keys)[_i];              \
    }

    MERGE_KEYS(sched_keys, ns);
    MERGE_KEYS(mem_keys, nm);
    MERGE_KEYS(sc_keys, nsc);
    MERGE_KEYS(lock_keys, nl);
#undef MERGE_KEYS

    fprintf(stderr, "[reader] %d unique (pid,cpu) entries before trim\n", nk);

    // ── get snapshot timestamp ────────────────────────────────────────────────
    struct timespec ts_now;
    clock_gettime(CLOCK_REALTIME, &ts_now);
    double ts_s = (double)ts_now.tv_sec + (double)ts_now.tv_nsec / 1e9;

    // ── write CSV ─────────────────────────────────────────────────────────────
    FILE *f = fopen(csv_path, append ? "a" : "w");
    if (!f)
    {
        fprintf(stderr, "[reader] fopen %s: %s\n", csv_path, strerror(errno));
        return 1;
    }
    if (!append)
        write_header(f);

    int written = 0, trimmed = 0;

    for (int k = 0; k < nk; k++)
    {
        struct sched_val *sv = NULL;
        struct mem_val *mv = NULL;
        struct syscall_val *scv = NULL;
        struct lock_val *lv = NULL;
        const char *comm = "?";

        for (int i = 0; i < ns; i++)
            if (sched_keys[i].pid == all_keys[k].pid &&
                sched_keys[i].cpu == all_keys[k].cpu)
            {
                sv = &sched_vals[i];
                if (sv->comm[0])
                    comm = sv->comm;
                break;
            }
        for (int i = 0; i < nm; i++)
            if (mem_keys[i].pid == all_keys[k].pid &&
                mem_keys[i].cpu == all_keys[k].cpu)
            {
                mv = &mem_vals[i];
                break;
            }
        for (int i = 0; i < nsc; i++)
            if (sc_keys[i].pid == all_keys[k].pid &&
                sc_keys[i].cpu == all_keys[k].cpu)
            {
                scv = &sc_vals[i];
                break;
            }
        for (int i = 0; i < nl; i++)
            if (lock_keys[i].pid == all_keys[k].pid &&
                lock_keys[i].cpu == all_keys[k].cpu)
            {
                lv = &lock_vals[i];
                break;
            }

        // ── time trim ────────────────────────────────────────────────────────
        // Use last_seen_ns from whichever map has the most recent activity.
        // An entry is kept only if it was active within the valid window.
        // This filters out:
        //   - processes that only appeared during warmup (first skip_start s)
        //   - processes that only appeared during teardown (last skip_end s)
        if (session_start_ns > 0 && (skip_start > 0 || skip_end > 0))
        {
            uint64_t last_seen = 0;
            if (sv && sv->last_seen_ns > last_seen)
                last_seen = sv->last_seen_ns;
            if (mv && mv->last_seen_ns > last_seen)
                last_seen = mv->last_seen_ns;
            if (scv && scv->last_seen_ns > last_seen)
                last_seen = scv->last_seen_ns;
            if (lv && lv->last_seen_ns > last_seen)
                last_seen = lv->last_seen_ns;

            if (last_seen > 0 &&
                (last_seen < trim_start_ns || last_seen > trim_end_ns))
            {
                trimmed++;
                continue;
            }
        }

        write_row(f, ts_s, &all_keys[k], sv, mv, scv, lv, comm, label);
        written++;
    }

    fclose(f);

    fprintf(stderr, "[reader] trimmed %d entries (warmup/teardown noise)\n", trimmed);
    fprintf(stderr, "[reader] wrote %d rows → %s\n", written, csv_path);

    // ── clean up pins and meta ────────────────────────────────────────────────
    for (int i = 0; i < 4; i++)
    {
        snprintf(path, sizeof(path), "%s/%s", PIN_BASE, map_names[i]);
        unlink(path);
    }
    unlink(META_FILE);
    fprintf(stderr, "[reader] unpinned maps, removed session meta\n");

    return 0;
}