/*
 * pinner.c
 *
 * Loads perf_monitor.bpf.o and attaches all probes, then waits.
 *
 * The four aggregation maps (sched_map, mem_map, syscall_map, lock_map)
 * are declared with LIBBPF_PIN_BY_NAME in the BPF program, so libbpf
 * automatically pins them to /sys/fs/bpf/<mapname> on load.
 * No manual bpf_map__pin() calls needed here.
 *
 * When you Ctrl+C:
 *   - programs detach (links are freed)
 *   - pinned maps remain at /sys/fs/bpf/ because the pin file holds
 *     a reference to the kernel map object independently of this process
 *
 * Then run reader to dump to CSV:
 *   sudo ./reader --label cpu_bound
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
#include <bpf/libbpf.h>
#include <bpf/bpf.h>

static volatile int running = 1;
static void on_sig(int s) { (void)s; running = 0; }

int main(int argc, char **argv)
{
    const char *obj_path = "perf_monitor.bpf.o";

    for (int i = 1; i < argc; i++)
        if (!strcmp(argv[i], "--obj") && i+1 < argc)
            obj_path = argv[++i];

    signal(SIGINT,  on_sig);
    signal(SIGTERM, on_sig);

    // ── load ──────────────────────────────────────────────────────────────────
    // LIBBPF_PIN_BY_NAME in the BPF program causes libbpf to automatically
    // pin sched_map, mem_map, syscall_map, lock_map to /sys/fs/bpf/ here.
    struct bpf_object *obj = bpf_object__open(obj_path);
    if (libbpf_get_error(obj)) {
        fprintf(stderr, "[pinner] open %s: %s\n", obj_path, strerror(errno));
        return 1;
    }

    // Remove stale pins before loading so libbpf can create fresh ones
    const char *map_names[] = { "sched_map", "mem_map", "syscall_map", "lock_map" };
    char pin_path[128];
    for (int i = 0; i < 4; i++) {
        snprintf(pin_path, sizeof(pin_path), "/sys/fs/bpf/%s", map_names[i]);
        unlink(pin_path);
    }

    if (bpf_object__load(obj) != 0) {
        fprintf(stderr, "[pinner] load: %s\n", strerror(errno));
        return 1;
    }
    fprintf(stderr, "[pinner] loaded %s\n", obj_path);

    // Confirm maps were auto-pinned
    for (int i = 0; i < 4; i++) {
        snprintf(pin_path, sizeof(pin_path), "/sys/fs/bpf/%s", map_names[i]);
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

    fprintf(stderr, "[pinner] collecting... Ctrl+C to stop\n");

    // ── wait — zero userspace overhead while BPF runs in kernel ──────────────
    while (running)
        sleep(1);

    bpf_object__close(obj);
    fprintf(stderr, "[pinner] stopped. maps still pinned at /sys/fs/bpf/\n");
    fprintf(stderr, "[pinner] now run: sudo ./reader --label <your_label>\n");
    return 0;
}
