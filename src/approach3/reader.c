/*
 * reader.c  —  Kernel-Verified Labeling edition
 *
 * Opens pinned BPF maps from /sys/fs/bpf/, reads session timestamps from
 * /tmp/perf_session.meta, applies optional warmup/teardown trim, merges all
 * four aggregate maps by (pid,cpu), and writes a CSV suitable for Jupyter
 * post-processing.
 *
 * New columns vs. previous version:
 *   timestamp_ns        – raw bpf_ktime_get_ns() at last map update (for
 *                         time-bucket grouping in Jupyter; no truncation)
 *   stall_ns            – cumulative runqueue wait (wakeup→run latency sum)
 *   max_stall_ns        – worst single runqueue latency observed
 *   avg_stall_ns        – stall_ns / latency_count
 *   latency_count       – number of latency samples
 *   avg_runq_ratio      – average (nrunnable*100/ncpus) at event time
 *   voluntary_switches  – task slept (I/O / futex wait) — "victim" signal
 *   involuntary_switches– preempted — CPU contention signal
 *   major_faults        – kernel-space page faults (memory pressure)
 *
 * Labeling philosophy (Jupyter post-processing):
 *   avg_stall_ns < 2_000_000            → class 0 (Normal)
 *   2_000_000 <= avg_stall_ns < 10_000_000 → class 1 (Low)
 *   10_000_000 <= avg_stall_ns < 50_000_000 → class 2 (Medium)
 *   avg_stall_ns >= 50_000_000          → class 3 (High)
 *
 * Build:
 *   gcc -O2 -o reader reader.c -lbpf -lelf -lz
 *
 * Run:
 *   sudo ./reader --label cpu_bound
 *   sudo ./reader --label cpu_bound --skip-start 5 --skip-end 2
 *   sudo ./reader --label cpu_bound --append
 *   sudo ./reader --no-trim --label idle
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

/* ── map value structs — MUST match perf_monitor.bpf.c exactly ──────────── */

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

    uint64_t stall_ns;      /* cumulative runq wait          */
    uint64_t max_stall_ns;  /* worst single runq wait        */
    uint64_t latency_count; /* #latency samples              */

    uint64_t runq_ratio_sum;   /* sum of ratio samples          */
    uint64_t runq_ratio_count; /* #ratio samples                */

    uint64_t last_update_ns; /* raw bpf_ktime_get_ns()        */
    char comm[TASK_COMM_LEN];
};

struct mem_val
{
    uint64_t minor_faults;
    uint64_t major_faults;
    uint64_t kmalloc_count;
    uint64_t kfree_count;
    uint64_t total_alloc_bytes;
    uint64_t total_free_bytes;
    uint64_t large_page_allocs;
    uint64_t last_update_ns;
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
    uint64_t last_update_ns;
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
    uint64_t last_update_ns;
};

/* ── integer average helper ─────────────────────────────────────────────── */
static inline uint64_t iavg(uint64_t sum, uint64_t count)
{
    return count > 0 ? sum / count : 0;
}

/* ── generic map drain ──────────────────────────────────────────────────── */
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

/* ── session metadata ───────────────────────────────────────────────────── */
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
        fprintf(stderr, "[reader] WARNING: end_ns=0 — pinner may not have "
                        "stopped cleanly; using current time\n");
        struct timespec ts;
        clock_gettime(CLOCK_MONOTONIC, &ts);
        *end_ns = (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
    }
    return 0;
}

/* ══════════════════════════════════════════════════════════════════════════
 * CSV output
 *
 * Column design rationale:
 *
 *   timestamp_ns       – raw kernel monotonic timestamp (bpf_ktime_get_ns())
 *                        at the last time this (pid,cpu) entry was updated.
 *                        Use in Jupyter: df.groupby(df.timestamp_ns // 2e9)
 *                        to create 2-second buckets.
 *
 *   stall_ns           – cumulative time the task spent waiting in the runqueue
 *                        (wakeup → actually running).  This is the primary
 *                        "pain" signal.  High value → CPU contention victim.
 *
 *   avg_stall_ns       – stall_ns / latency_count.  Use this for thresholding:
 *                        < 2ms   → Normal
 *                        2–10ms  → Low pressure
 *                        10–50ms → Medium
 *                        > 50ms  → High / bottleneck
 *
 *   max_stall_ns       – worst single runqueue wait.  Useful for detecting
 *                        transient spikes missed by the average.
 *
 *   avg_runq_ratio     – average system load at event time, expressed as
 *                        (nrunnable * 100) / ncpus.  Values > 100 indicate
 *                        oversubscription.  This is the "why" for the latency.
 *
 *   voluntary_switches – times the task gave up the CPU voluntarily (sleep,
 *                        I/O wait, futex).  High count → I/O-bound or
 *                        lock-contended victim.
 *
 *   involuntary_switches – times the task was preempted.  High count →
 *                          CPU-bound or noisy-neighbour victim.
 *
 *   major_faults       – kernel-space page faults; proxy for memory pressure.
 * ══════════════════════════════════════════════════════════════════════════ */

static void write_header(FILE *f)
{
    fprintf(f,
            /* identity */
            "timestamp_ns,pid,cpu,comm,"
            /* scheduling & latency — primary label signals */
            "ctx_switches,voluntary_switches,involuntary_switches,"
            "cpu_migrations,total_runtime_ns,"
            "stall_ns,avg_stall_ns,max_stall_ns,latency_count,"
            "avg_runq_ratio,"
            /* memory pressure */
            "minor_faults,major_faults,"
            "kmalloc_count,kfree_count,"
            "total_alloc_bytes,total_free_bytes,large_page_allocs,"
            /* syscall latency */
            "syscall_count,avg_syscall_latency_ns,max_syscall_latency_ns,"
            "read_count,write_count,read_bytes,write_bytes,"
            "mmap_count,"
            "futex_count,avg_futex_latency_ns,"
            "epoll_count,avg_epoll_latency_ns,"
            "poll_count,syscall_error_count,"
            /* lock contention */
            "mutex_contentions,avg_mutex_wait_ns,max_mutex_wait_ns,"
            "rwsem_read_contentions,avg_rwsem_read_wait_ns,"
            "rwsem_write_contentions,avg_rwsem_write_wait_ns,max_rwsem_write_wait_ns,"
            /* run-time label (from --label flag; ML target created in Jupyter) */
            "session_label\n");
}

static void write_row(FILE *f,
                      const struct agg_key *k,
                      const struct sched_val *sv,
                      const struct mem_val *mv,
                      const struct syscall_val *scv,
                      const struct lock_val *lv,
                      const char *comm,
                      const char *label)
{
    /*
     * timestamp_ns: use the most recent last_update_ns across all maps for
     * this (pid,cpu) entry.  This gives the best estimate of when the process
     * was last active.  Raw nanoseconds — no truncation.
     */
    uint64_t ts_ns = 0;
    if (sv && sv->last_update_ns > ts_ns)
        ts_ns = sv->last_update_ns;
    if (mv && mv->last_update_ns > ts_ns)
        ts_ns = mv->last_update_ns;
    if (scv && scv->last_update_ns > ts_ns)
        ts_ns = scv->last_update_ns;
    if (lv && lv->last_update_ns > ts_ns)
        ts_ns = lv->last_update_ns;

    uint64_t stall_ns = sv ? sv->stall_ns : 0;
    uint64_t lat_count = sv ? sv->latency_count : 0;
    uint64_t avg_stall = iavg(stall_ns, lat_count);
    uint64_t avg_runq = sv ? iavg(sv->runq_ratio_sum, sv->runq_ratio_count) : 0;

    fprintf(f,
            /* identity */
            "%llu,%u,%u,%s,"
            /* scheduling */
            "%llu,%llu,%llu,"
            "%llu,%llu,"
            /* latency — key label signals */
            "%llu,%llu,%llu,%llu,"
            "%llu,"
            /* memory */
            "%llu,%llu,"
            "%llu,%llu,"
            "%llu,%llu,%llu,"
            /* syscall */
            "%llu,%llu,%llu,"
            "%llu,%llu,%llu,%llu,"
            "%llu,"
            "%llu,%llu,"
            "%llu,%llu,"
            "%llu,%llu,"
            /* locks */
            "%llu,%llu,%llu,"
            "%llu,%llu,"
            "%llu,%llu,%llu,"
            /* label */
            "%s\n",

            /* identity */
            (unsigned long long)ts_ns,
            k->pid, k->cpu, comm,

            /* scheduling */
            (unsigned long long)(sv ? sv->ctx_switches : 0),
            (unsigned long long)(sv ? sv->voluntary_switches : 0),
            (unsigned long long)(sv ? sv->involuntary_switches : 0),
            (unsigned long long)(sv ? sv->cpu_migrations : 0),
            (unsigned long long)(sv ? sv->total_runtime_ns : 0),

            /* latency */
            (unsigned long long)stall_ns,
            (unsigned long long)avg_stall,
            (unsigned long long)(sv ? sv->max_stall_ns : 0),
            (unsigned long long)lat_count,
            (unsigned long long)avg_runq,

            /* memory */
            (unsigned long long)(mv ? mv->minor_faults : 0),
            (unsigned long long)(mv ? mv->major_faults : 0),
            (unsigned long long)(mv ? mv->kmalloc_count : 0),
            (unsigned long long)(mv ? mv->kfree_count : 0),
            (unsigned long long)(mv ? mv->total_alloc_bytes : 0),
            (unsigned long long)(mv ? mv->total_free_bytes : 0),
            (unsigned long long)(mv ? mv->large_page_allocs : 0),

            /* syscall */
            (unsigned long long)(scv ? scv->total_count : 0),
            (unsigned long long)(scv ? iavg(scv->latency_sum_ns, scv->total_count) : 0),
            (unsigned long long)(scv ? scv->latency_max_ns : 0),
            (unsigned long long)(scv ? scv->read_count : 0),
            (unsigned long long)(scv ? scv->write_count : 0),
            (unsigned long long)(scv ? scv->read_bytes : 0),
            (unsigned long long)(scv ? scv->write_bytes : 0),
            (unsigned long long)(scv ? scv->mmap_count : 0),
            (unsigned long long)(scv ? scv->futex_count : 0),
            (unsigned long long)(scv ? iavg(scv->futex_latency_sum_ns, scv->futex_count) : 0),
            (unsigned long long)(scv ? scv->epoll_count : 0),
            (unsigned long long)(scv ? iavg(scv->epoll_latency_sum_ns, scv->epoll_count) : 0),
            (unsigned long long)(scv ? scv->poll_count : 0),
            (unsigned long long)(scv ? scv->error_count : 0),

            /* locks */
            (unsigned long long)(lv ? lv->mutex_contentions : 0),
            (unsigned long long)(lv ? iavg(lv->mutex_wait_sum_ns, lv->mutex_contentions) : 0),
            (unsigned long long)(lv ? lv->mutex_wait_max_ns : 0),
            (unsigned long long)(lv ? lv->rwsem_read_contentions : 0),
            (unsigned long long)(lv ? iavg(lv->rwsem_read_wait_sum_ns, lv->rwsem_read_contentions) : 0),
            (unsigned long long)(lv ? lv->rwsem_write_contentions : 0),
            (unsigned long long)(lv ? iavg(lv->rwsem_write_wait_sum_ns, lv->rwsem_write_contentions) : 0),
            (unsigned long long)(lv ? lv->rwsem_write_wait_max_ns : 0),

            label);
}

/* ── main ───────────────────────────────────────────────────────────────── */
int main(int argc, char **argv)
{
    const char *label = "unknown";
    const char *csv_path = "perf_metrics.csv";
    int append = 0;
    int skip_start = 5; /* default: drop first 5 s (warmup)    */
    int skip_end = 2;   /* default: drop last  2 s (teardown)  */

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

    /* ── session timestamps ─────────────────────────────────────────────── */
    uint64_t session_start_ns = 0, session_end_ns = 0;
    read_session_meta(&session_start_ns, &session_end_ns);

    uint64_t trim_start_ns = session_start_ns + (uint64_t)skip_start * 1000000000ULL;
    uint64_t trim_end_ns = session_end_ns - (uint64_t)skip_end * 1000000000ULL;

    double dur = (double)(session_end_ns - session_start_ns) / 1e9;
    fprintf(stderr, "[reader] session duration:  %.1f s\n", dur);
    fprintf(stderr, "[reader] trim: skip first %d s, skip last %d s\n",
            skip_start, skip_end);

    if (session_start_ns > 0 && trim_end_ns <= trim_start_ns)
    {
        fprintf(stderr, "[reader] WARNING: session too short for trim window "
                        "(%.1f s). Disabling trim.\n",
                dur);
        skip_start = skip_end = 0;
        trim_start_ns = session_start_ns;
        trim_end_ns = session_end_ns;
    }

    /* ── open pinned maps ───────────────────────────────────────────────── */
    const char *map_names[4] = {"sched_map", "mem_map", "syscall_map", "lock_map"};
    int fds[4] = {-1, -1, -1, -1};
    char path[128];

    for (int i = 0; i < 4; i++)
    {
        snprintf(path, sizeof(path), "%s/%s", PIN_BASE, map_names[i]);
        fds[i] = bpf_obj_get(path);
        if (fds[i] < 0)
            fprintf(stderr, "[reader] could not open %s: %s\n", path, strerror(errno));
        else
            fprintf(stderr, "[reader] opened %s (fd=%d)\n", map_names[i], fds[i]);
    }

    if (fds[0] < 0 && fds[1] < 0 && fds[2] < 0 && fds[3] < 0)
    {
        fprintf(stderr, "[reader] no maps found — did pinner run?\n");
        return 1;
    }

    /* ── drain all maps ─────────────────────────────────────────────────── */
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

    /* ── collect unique (pid,cpu) keys ─────────────────────────────────── */
    static struct agg_key all_keys[MAX_KEYS];
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

    /* ── write CSV ──────────────────────────────────────────────────────── */
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

        /* ── time trim ──────────────────────────────────────────────────── */
        /*
         * Use last_update_ns (= bpf_ktime_get_ns() from the kernel) for
         * trimming — this is the same clock as session_start/end_ns written
         * by pinner.c via clock_gettime(CLOCK_MONOTONIC).
         *
         * An entry is KEPT only if it was active inside the valid window.
         * We intentionally do NOT filter out "low activity" entries —
         * those are the Normal (class 0) rows we need for training.
         */
        if (session_start_ns > 0 && (skip_start > 0 || skip_end > 0))
        {
            uint64_t last_seen = 0;
            if (sv && sv->last_update_ns > last_seen)
                last_seen = sv->last_update_ns;
            if (mv && mv->last_update_ns > last_seen)
                last_seen = mv->last_update_ns;
            if (scv && scv->last_update_ns > last_seen)
                last_seen = scv->last_update_ns;
            if (lv && lv->last_update_ns > last_seen)
                last_seen = lv->last_update_ns;

            if (last_seen > 0 && last_seen < trim_start_ns)
            {
                trimmed++;
                continue; /* only active during warmup — skip */
            }
            /* Note: we do NOT trim last_seen > trim_end_ns because that would
             * drop all entries that were still running at shutdown, which is
             * most of the interesting processes.  Teardown trimming is better
             * handled in Jupyter by filtering on timestamp_ns. */
        }

        write_row(f, &all_keys[k], sv, mv, scv, lv, comm, label);
        written++;
    }

    fclose(f);

    fprintf(stderr, "[reader] trimmed %d entries (warmup noise)\n", trimmed);
    fprintf(stderr, "[reader] wrote   %d rows → %s\n", written, csv_path);
    fprintf(stderr, "\n");
    fprintf(stderr, "[reader] Jupyter labeling hints:\n");
    fprintf(stderr, "  # Time-bucket grouping (2s intervals):\n");
    fprintf(stderr, "  df['bucket'] = df['timestamp_ns'] // 2_000_000_000\n");
    fprintf(stderr, "  # Multi-class label from avg_stall_ns:\n");
    fprintf(stderr, "  bins   = [0, 2e6, 10e6, 50e6, float('inf')]\n");
    fprintf(stderr, "  labels = [0, 1, 2, 3]\n");
    fprintf(stderr, "  df['label'] = pd.cut(df['avg_stall_ns'], bins=bins, labels=labels)\n");
    fprintf(stderr, "  # CPU contention ratio > 100 confirms oversubscription:\n");
    fprintf(stderr, "  df['oversubscribed'] = df['avg_runq_ratio'] > 100\n");

    /* ── clean up pins and meta ─────────────────────────────────────────── */
    for (int i = 0; i < 4; i++)
    {
        snprintf(path, sizeof(path), "%s/%s", PIN_BASE, map_names[i]);
        unlink(path);
    }
    unlink(META_FILE);
    fprintf(stderr, "[reader] unpinned maps, removed session meta\n");

    return 0;
}