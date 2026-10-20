/*
 * Realtime pinner for realtime_monitor.bpf.o.
 *
 * Build from src/realtime:
 *   gcc -O2 -Wall -Wextra -o realtime_pinner pinner.c -lbpf -lelf -lz
 *
 * Run from src/realtime (as root):
 *   sudo ./realtime_pinner --obj ./realtime_monitor.bpf.o
 */
#define _GNU_SOURCE
#include <bpf/bpf.h>
#include <bpf/libbpf.h>
#include <errno.h>
#include <signal.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define META_FILE "/tmp/perf_session.meta"
#define PIN_BASE "/sys/fs/bpf/pbd_test"
#define MAX_PROGRAMS 64

static volatile sig_atomic_t running = 1;

static void on_signal(int signal_number)
{
    (void)signal_number;
    running = 0;
}

static uint64_t monotonic_ns(void)
{
    struct timespec ts;
    if (clock_gettime(CLOCK_MONOTONIC, &ts) != 0)
        return 0;
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static int remove_stale_pins(const char *pin_base)
{
    static const char *const map_names[] = {
        "sched_map", "mem_map", "syscall_map", "lock_map",
    };
    char path[128];
    int failed = 0;

    for (size_t i = 0; i < sizeof(map_names) / sizeof(map_names[0]); i++) {
        int length = snprintf(path, sizeof(path), "%s/%s", pin_base, map_names[i]);
        if (length < 0 || (size_t)length >= sizeof(path)) {
            fprintf(stderr, "[realtime-pinner] pin path is too long\n");
            failed = 1;
            continue;
        }
        if (unlink(path) != 0 && errno != ENOENT) {
            fprintf(stderr, "[realtime-pinner] could not remove stale pin %s: %s\n",
                    path, strerror(errno));
            failed = 1;
        }
    }
    return failed ? -1 : 0;
}

static int write_session_metadata(const char *meta_path, uint64_t start_ns, uint64_t end_ns)
{
    FILE *meta = fopen(meta_path, "w");
    if (!meta) {
        fprintf(stderr, "[realtime-pinner] could not write %s: %s\n",
                meta_path, strerror(errno));
        return -1;
    }
    int result = fprintf(meta, "start_ns=%llu\nend_ns=%llu\n",
                         (unsigned long long)start_ns,
                         (unsigned long long)end_ns);
    if (fclose(meta) != 0 || result < 0) {
        fprintf(stderr, "[realtime-pinner] failed writing %s\n", meta_path);
        return -1;
    }
    return 0;
}

int main(int argc, char **argv)
{
    const char *object_path = getenv("REALTIME_BPF_OBJECT_PATH");
    const char *pin_base = getenv("REALTIME_PIN_BASE");
    const char *meta_file = getenv("REALTIME_META_FILE");
    if (!object_path || !object_path[0])
        object_path = "./realtime_monitor.bpf.o";
    if (!pin_base || !pin_base[0])
        pin_base = PIN_BASE;
    if (!meta_file || !meta_file[0])
        meta_file = META_FILE;
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--obj") == 0 && i + 1 < argc) {
            object_path = argv[++i];
        } else {
            fprintf(stderr, "Usage: %s [--obj realtime_monitor.bpf.o]\n", argv[0]);
            return 2;
        }
    }

    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);

    struct bpf_object_open_opts open_options = {
        .sz = sizeof(open_options),
        .pin_root_path = pin_base,
    };
    struct bpf_object *object = bpf_object__open_file(object_path, &open_options);
    long open_error = libbpf_get_error(object);
    if (open_error) {
        fprintf(stderr, "[realtime-pinner] cannot open %s: %s\n",
                object_path, strerror((int)-open_error));
        return 1;
    }

    if (remove_stale_pins(pin_base) != 0) {
        bpf_object__close(object);
        return 1;
    }
    int load_error = bpf_object__load(object);
    if (load_error) {
        fprintf(stderr, "[realtime-pinner] cannot load %s: %s\n",
                object_path, strerror(-load_error));
        bpf_object__close(object);
        return 1;
    }
    fprintf(stderr, "[realtime-pinner] loaded %s\n", object_path);

    static const char *const required_maps[] = {
        "sched_map", "mem_map", "syscall_map", "lock_map",
    };
    char pin_path[128];
    for (size_t i = 0; i < sizeof(required_maps) / sizeof(required_maps[0]); i++) {
        int length = snprintf(pin_path, sizeof(pin_path), "%s/%s", pin_base, required_maps[i]);
        if (length < 0 || (size_t)length >= sizeof(pin_path) || access(pin_path, F_OK) != 0) {
            fprintf(stderr, "[realtime-pinner] required map was not pinned: %s/%s\n",
                    pin_base, required_maps[i]);
            bpf_object__close(object);
            return 1;
        }
    }

    struct bpf_link *links[MAX_PROGRAMS] = {0};
    size_t link_count = 0;
    size_t failures = 0;
    struct bpf_program *program;
    bpf_object__for_each_program(program, object) {
        if (link_count == MAX_PROGRAMS) {
            fprintf(stderr, "[realtime-pinner] too many programs to attach\n");
            failures++;
            break;
        }
        struct bpf_link *link = bpf_program__attach(program);
        long attach_error = libbpf_get_error(link);
        if (attach_error) {
            fprintf(stderr, "[realtime-pinner] required probe %s failed: %s\n",
                    bpf_program__name(program), strerror((int)-attach_error));
            failures++;
            continue;
        }
        links[link_count++] = link;
    }

    if (failures || link_count == 0) {
        fprintf(stderr, "[realtime-pinner] refusing partial collection (%zu attached, %zu failed)\n",
                link_count, failures);
        for (size_t i = 0; i < link_count; i++)
            bpf_link__destroy(links[i]);
        bpf_object__close(object);
        return 1;
    }
    fprintf(stderr, "[realtime-pinner] all %zu required probes attached\n", link_count);

    uint64_t start_ns = monotonic_ns();
    if (!start_ns || write_session_metadata(meta_file, start_ns, 0) != 0) {
        for (size_t i = 0; i < link_count; i++)
            bpf_link__destroy(links[i]);
        bpf_object__close(object);
        return 1;
    }
    fprintf(stderr, "[realtime-pinner] collecting; press Ctrl+C to stop\n");

    while (running)
        sleep(1);

    uint64_t end_ns = monotonic_ns();
    int metadata_error = write_session_metadata(meta_file, start_ns, end_ns);
    for (size_t i = 0; i < link_count; i++)
        bpf_link__destroy(links[i]);
    bpf_object__close(object);
    fprintf(stderr, "[realtime-pinner] stopped; maps remain pinned under %s/\n", pin_base);
    return metadata_error == 0 ? 0 : 1;
}
