/*
 * pinner.c
 *
 * Loads perf_monitor.bpf.o, attaches all probes, waits.
 * Writes session start/end timestamps to /tmp/perf_session.meta
 * so reader can trim the first N seconds (benchmark warmup) and
 * last N seconds (benchmark teardown) from the collected data.
 *
 * Build:
 *   gcc -O2 -o pinner pinner.c -lbpf -lelf -lz
 *
 * Run:
 *   sudo ./pinner --obj perf_monitor.bpf.o
 */

#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <signal.h>
#include <unistd.h>
#include <time.h>
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

#define META_FILE    "/tmp/perf_session.meta"
#define PIN_BASE     "/sys/fs/bpf"

static volatile int running = 1;
static void on_sig(int s) { (void)s; running = 0; }

static uint64_t now_ns(void)
{
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

int main(int argc, char **argv)
{
    const char *obj_path = "perf_monitor.bpf.o";

    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--obj") && i+1 < argc)
            obj_path = argv[++i];

    signal(SIGINT,  on_sig);
    signal(SIGTERM, on_sig);

    // ── load BPF object ───────────────────────────────────────────────────────
    struct bpf_object *obj = bpf_object__open(obj_path);
    if (libbpf_get_error(obj)) {
        fprintf(stderr, "[pinner] open %s: %s\n", obj_path, strerror(errno));
        return 1;
    }

    // remove stale pins so libbpf creates fresh ones
    const char *map_names[] = { "sched_map", "mem_map", "syscall_map", "lock_map" };
    char pin_path[128];
    for (int i = 0; i < 4; i++) {
        snprintf(pin_path, sizeof(pin_path), "%s/%s", PIN_BASE, map_names[i]);
        unlink(pin_path);
    }

    if (bpf_object__load(obj) != 0) {
        fprintf(stderr, "[pinner] load: %s\n", strerror(errno));
        return 1;
    }
    fprintf(stderr, "[pinner] loaded %s\n", obj_path);

    // confirm auto-pins
    for (int i = 0; i < 4; i++) {
        snprintf(pin_path, sizeof(pin_path), "%s/%s", PIN_BASE, map_names[i]);
        if (access(pin_path, F_OK) == 0)
            fprintf(stderr, "[pinner] auto-pinned: %s\n", pin_path);
        else
            fprintf(stderr, "[pinner] WARNING: pin missing: %s\n", pin_path);
    }

    // ── attach all programs ───────────────────────────────────────────────────
    struct bpf_program *prog;
    int ok = 0, fail = 0;
    bpf_object__for_each_program(prog, obj) {
        struct bpf_link *link = bpf_program__attach(prog);
        if (libbpf_get_error(link)) {
            fprintf(stderr, "[pinner] attach failed: %s (%s)\n",
                    bpf_program__name(prog), strerror(errno));
            fail++;
        } else {
            ok++;
        }
    }
    fprintf(stderr, "[pinner] %d programs attached, %d failed\n", ok, fail);
    if (ok == 0) {
        fprintf(stderr, "[pinner] nothing attached — exiting\n");
        return 1;
    }

    // ── record session start time ─────────────────────────────────────────────
    uint64_t session_start_ns = now_ns();

    // write meta file — reader uses this to trim warmup/teardown
    FILE *meta = fopen(META_FILE, "w");
    if (meta) {
        fprintf(meta, "start_ns=%llu\nend_ns=0\n",
                (unsigned long long)session_start_ns);
        fclose(meta);
    } else {
        fprintf(stderr, "[pinner] WARNING: could not write %s: %s\n",
                META_FILE, strerror(errno));
    }

    fprintf(stderr, "[pinner] session started at %.3f\n",
            (double)session_start_ns / 1e9);
    fprintf(stderr, "[pinner] collecting... Ctrl+C to stop\n");

    // ── wait — BPF runs in kernel, zero userspace overhead ───────────────────
    while (running)
        sleep(1);

    // ── record session end time ───────────────────────────────────────────────
    uint64_t session_end_ns = now_ns();

    // update meta file with end time
    meta = fopen(META_FILE, "w");
    if (meta) {
        fprintf(meta, "start_ns=%llu\nend_ns=%llu\n",
                (unsigned long long)session_start_ns,
                (unsigned long long)session_end_ns);
        fclose(meta);
    }

    double duration_s = (double)(session_end_ns - session_start_ns) / 1e9;
    fprintf(stderr, "[pinner] stopped. session duration: %.1fs\n", duration_s);
    fprintf(stderr, "[pinner] maps still pinned at %s/\n", PIN_BASE);
    fprintf(stderr, "[pinner] now run: sudo ./reader --label <your_label>\n");

    bpf_object__close(obj);
    return 0;
}