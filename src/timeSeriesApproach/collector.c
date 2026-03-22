/*
 * collector.c
 *
 * Loads perf_monitor.bpf.o, pins all four aggregation maps under
 * /sys/fs/bpf/, attaches all BPF programs, then every INTERVAL_MS:
 *
 *   1. Iterates sched_map, mem_map, syscall_map, lock_map
 *   2. Merges values by (pid, cpu) into one CSV row
 *   3. Zeros out each entry so next interval is a fresh delta
 *   4. Appends rows to perf_metrics.csv
 *
 * Build:
 *   gcc -O2 -o collector collector.c -lbpf -lelf -lz
 *
 * Run:
 *   sudo ./collector --obj perf_monitor.bpf.o --label baseline
 *   sudo ./collector --obj perf_monitor.bpf.o --label cpu_bound --interval 100
 *
 * Stop with Ctrl+C.
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <signal.h>
#include <time.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

// ─── tunables ─────────────────────────────────────────────────────────────────
#define DEFAULT_INTERVAL_MS 100
#define PIN_BASE_PATH "/sys/fs/bpf"
#define MAX_KEYS 8192
#define TASK_COMM_LEN 16

// ─── map key — must match BPF program ────────────────────────────────────────
struct agg_key
{
    uint32_t pid;
    uint32_t cpu;
};

// ─── map value types — must match BPF program ────────────────────────────────

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

// ─── globals ──────────────────────────────────────────────────────────────────
static volatile int running = 1;
static FILE *csv_fp = NULL;

static void sigint_handler(int sig)
{
    (void)sig;
    running = 0;
}

// ─── helpers ──────────────────────────────────────────────────────────────────

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static void ms_sleep(int ms)
{
    struct timespec ts = {
        .tv_sec = ms / 1000,
        .tv_nsec = (ms % 1000) * 1000000L};
    nanosleep(&ts, NULL);
}

// Safe average — avoids divide-by-zero
#define AVG(sum, count) ((count) > 0 ? (sum) / (count) : (uint64_t)0)

// ─── CSV header + row ─────────────────────────────────────────────────────────

static void write_csv_header(FILE *f)
{
    fprintf(f,
            "timestamp_s,interval_start_ns,pid,cpu,comm,"
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

static void write_csv_row(FILE *f,
                          double ts_s, uint64_t interval_start_ns,
                          const struct agg_key *k,
                          const struct sched_val *sv,
                          const struct mem_val *mv,
                          const struct syscall_val *scv,
                          const struct lock_val *lv,
                          const char *label,
                          const char *comm)
{
    // FIX: all uint64_t values use %llu, not %lu
    fprintf(f,
            "%.3f,%llu," // timestamp_s, interval_start_ns
            "%u,%u,%s,"  // pid, cpu, comm
            // sched
            "%llu,%llu,%llu," // ctx_switches, voluntary, involuntary
            "%llu,%llu,"      // migrations, total_runtime_ns
            "%llu,%llu,"      // avg_runq_latency_ns, max_runq_latency_ns
            // mem
            "%llu,%llu,"      // minor_faults, kernel_faults
            "%llu,%llu,"      // kmalloc_count, kfree_count
            "%llu,%llu,%llu," // total_alloc_bytes, total_free_bytes, large_page_allocs
            // syscall
            "%llu,%llu,%llu,"      // syscall_count, avg_latency, max_latency
            "%llu,%llu,%llu,%llu," // read_count, write_count, read_bytes, write_bytes
            "%llu,%llu,%llu,"      // mmap_count, futex_count, avg_futex_latency
            "%llu,%llu,"           // epoll_count, avg_epoll_latency
            "%llu,%llu,"           // poll_count, error_count
            // locks
            "%llu,%llu,%llu," // mutex_contentions, avg_wait, max_wait
            "%llu,%llu,"      // rwsem_read_contentions, avg_wait
            "%llu,%llu,%llu," // rwsem_write_contentions, avg_wait, max_wait
            // label
            "%s\n",

            ts_s,
            (unsigned long long)interval_start_ns,
            k->pid, k->cpu, comm,

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

// ─── map drain ────────────────────────────────────────────────────────────────
// Iterate every entry in fd, copy into keys[]/vals[], zero out the entry.
// Returns number of entries read.

#define DEFINE_DRAIN(fname, vtype)                                \
    static int fname(int fd, struct agg_key *keys, vtype *vals)   \
    {                                                             \
        struct agg_key prev = {}, next = {};                      \
        struct agg_key *prev_ptr = NULL;                          \
        int n = 0;                                                \
        while (n < MAX_KEYS &&                                    \
               bpf_map_get_next_key(fd, prev_ptr, &next) == 0)    \
        {                                                         \
            vtype v = {};                                         \
            if (bpf_map_lookup_elem(fd, &next, &v) == 0)          \
            {                                                     \
                keys[n] = next;                                   \
                vals[n] = v;                                      \
                n++;                                              \
                vtype zero = {};                                  \
                bpf_map_update_elem(fd, &next, &zero, BPF_EXIST); \
            }                                                     \
            prev = next;                                          \
            prev_ptr = &prev;                                     \
        }                                                         \
        return n;                                                 \
    }

DEFINE_DRAIN(drain_sched, struct sched_val)
DEFINE_DRAIN(drain_mem, struct mem_val)
DEFINE_DRAIN(drain_syscall, struct syscall_val)
DEFINE_DRAIN(drain_lock, struct lock_val)

// ─── usage ────────────────────────────────────────────────────────────────────

static void usage(const char *prog)
{
    fprintf(stderr,
            "Usage: %s [options]\n"
            "  --obj <path>      BPF object file      (default: perf_monitor.bpf.o)\n"
            "  --csv <path>      output CSV file       (default: perf_metrics.csv)\n"
            "  --interval <ms>   poll interval in ms   (default: 100)\n"
            "  --label <str>     workload label         (default: unknown)\n"
            "  --append          append to existing CSV\n",
            prog);
}

// ─── main ─────────────────────────────────────────────────────────────────────

int main(int argc, char **argv)
{
    const char *obj_path = "perf_monitor.bpf.o";
    const char *csv_path = "perf_metrics.csv";
    const char *label = "unknown";
    int interval = DEFAULT_INTERVAL_MS;
    int append = 0;

    for (int i = 1; i < argc; i++)
    {
        if (!strcmp(argv[i], "--obj") && i + 1 < argc)
            obj_path = argv[++i];
        else if (!strcmp(argv[i], "--csv") && i + 1 < argc)
            csv_path = argv[++i];
        else if (!strcmp(argv[i], "--label") && i + 1 < argc)
            label = argv[++i];
        else if (!strcmp(argv[i], "--interval") && i + 1 < argc)
            interval = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--append"))
            append = 1;
        else if (!strcmp(argv[i], "--help"))
        {
            usage(argv[0]);
            return 0;
        }
    }

    signal(SIGINT, sigint_handler);
    signal(SIGTERM, sigint_handler);

    // ── Load BPF object ───────────────────────────────────────────────────────
    struct bpf_object *obj = bpf_object__open(obj_path);
    if (libbpf_get_error(obj))
    {
        fprintf(stderr, "[collector] bpf_object__open(%s): %s\n",
                obj_path, strerror(errno));
        return 1;
    }
    if (bpf_object__load(obj) != 0)
    {
        fprintf(stderr, "[collector] bpf_object__load: %s\n", strerror(errno));
        return 1;
    }
    fprintf(stderr, "[collector] loaded: %s\n", obj_path);

    // ── Attach all programs individually ─────────────────────────────────────
    // FIX: bpf_object__attach_programs() does not exist in all libbpf versions.
    // Instead iterate programs and attach each one.
    {
        struct bpf_program *prog;
        int attach_ok = 0, attach_fail = 0;
        bpf_object__for_each_program(prog, obj)
        {
            struct bpf_link *link = bpf_program__attach(prog);
            if (libbpf_get_error(link))
            {
                fprintf(stderr, "[collector] attach failed: %s  (%s)\n",
                        bpf_program__name(prog), strerror(errno));
                attach_fail++;
            }
            else
            {
                attach_ok++;
            }
        }
        fprintf(stderr, "[collector] programs: %d attached, %d failed\n",
                attach_ok, attach_fail);
        if (attach_ok == 0)
        {
            fprintf(stderr, "[collector] no programs attached — exiting\n");
            return 1;
        }
    }

    // ── Pin maps ──────────────────────────────────────────────────────────────
    const char *map_names[4] = {
        "sched_map", "mem_map", "syscall_map", "lock_map"};
    char pin_paths[4][128];
    int map_fds[4] = {-1, -1, -1, -1};

    for (int i = 0; i < 4; i++)
    {
        snprintf(pin_paths[i], sizeof(pin_paths[i]),
                 "%s/%s", PIN_BASE_PATH, map_names[i]);
        unlink(pin_paths[i]); // remove stale pin

        struct bpf_map *m = bpf_object__find_map_by_name(obj, map_names[i]);
        if (!m)
        {
            fprintf(stderr, "[collector] map not found: %s\n", map_names[i]);
            continue;
        }
        if (bpf_map__pin(m, pin_paths[i]) != 0)
        {
            fprintf(stderr, "[collector] pin %s failed: %s\n",
                    pin_paths[i], strerror(errno));
        }
        else
        {
            map_fds[i] = bpf_map__fd(m);
            fprintf(stderr, "[collector] pinned %s (fd=%d)\n",
                    map_names[i], map_fds[i]);
        }
    }

    // ── Open CSV ──────────────────────────────────────────────────────────────
    csv_fp = fopen(csv_path, append ? "a" : "w");
    if (!csv_fp)
    {
        fprintf(stderr, "[collector] fopen(%s): %s\n", csv_path, strerror(errno));
        return 1;
    }
    if (!append)
        write_csv_header(csv_fp);

    fprintf(stderr, "[collector] interval=%dms  label=%s  csv=%s\n",
            interval, label, csv_path);
    fprintf(stderr, "[collector] running... Ctrl+C to stop\n");

    // ── Allocate drain buffers ────────────────────────────────────────────────
    static struct agg_key sched_keys[MAX_KEYS];
    static struct sched_val sched_vals[MAX_KEYS];
    static struct agg_key mem_keys[MAX_KEYS];
    static struct mem_val mem_vals[MAX_KEYS];
    static struct agg_key sc_keys[MAX_KEYS];
    static struct syscall_val sc_vals[MAX_KEYS];
    static struct agg_key lock_keys[MAX_KEYS];
    static struct lock_val lock_vals[MAX_KEYS];

    uint64_t total_rows = 0;

    // ── Poll loop ─────────────────────────────────────────────────────────────
    while (running)
    {
        ms_sleep(interval);

        uint64_t ts_ns = now_ns();
        double ts_s = (double)ts_ns / 1e9;

        int ns = map_fds[0] >= 0 ? drain_sched(map_fds[0], sched_keys, sched_vals) : 0;
        int nm = map_fds[1] >= 0 ? drain_mem(map_fds[1], mem_keys, mem_vals) : 0;
        int nsc = map_fds[2] >= 0 ? drain_syscall(map_fds[2], sc_keys, sc_vals) : 0;
        int nl = map_fds[3] >= 0 ? drain_lock(map_fds[3], lock_keys, lock_vals) : 0;

        // Collect all unique (pid,cpu) keys across all four maps
        struct agg_key all_keys[MAX_KEYS * 4];
        int nk = 0;

#define MERGE_KEYS(src_keys, src_n)                           \
    for (int _i = 0; _i < (src_n) && nk < MAX_KEYS * 4; _i++) \
    {                                                         \
        int _found = 0;                                       \
        for (int _j = 0; _j < nk; _j++)                       \
        {                                                     \
            if (all_keys[_j].pid == (src_keys)[_i].pid &&     \
                all_keys[_j].cpu == (src_keys)[_i].cpu)       \
            {                                                 \
                _found = 1;                                   \
                break;                                        \
            }                                                 \
        }                                                     \
        if (!_found)                                          \
            all_keys[nk++] = (src_keys)[_i];                  \
    }

        MERGE_KEYS(sched_keys, ns);
        MERGE_KEYS(mem_keys, nm);
        MERGE_KEYS(sc_keys, nsc);
        MERGE_KEYS(lock_keys, nl);
#undef MERGE_KEYS

        // For each unique key find its value in each map and write one row
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

            write_csv_row(csv_fp, ts_s, ts_ns,
                          &all_keys[k], sv, mv, scv, lv,
                          label, comm);
            total_rows++;
        }

        fflush(csv_fp);

        if (total_rows % 500 == 0 && total_rows > 0)
            fprintf(stderr, "[collector] %llu rows written\n",
                    (unsigned long long)total_rows);
    }

    // ── Cleanup ───────────────────────────────────────────────────────────────
    fclose(csv_fp);
    fprintf(stderr, "[collector] stopped. %llu rows → %s\n",
            (unsigned long long)total_rows, csv_path);

    for (int i = 0; i < 4; i++)
        unlink(pin_paths[i]);

    bpf_object__close(obj);
    return 0;
}