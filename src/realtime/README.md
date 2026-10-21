# Realtime inference pipeline

The realtime pipeline uses a dedicated reduced eBPF collector and the v7.2 SHAP-reduced classifier plus its matching 20-feature attribution MLP. It reconstructs per-`(pid, cpu)` rows from four pinned maps, engineers the notebook features in userspace, and runs both models on the exact saved feature order.

## Build the reduced collector and pinner

Install the host build prerequisites (`clang` with the BPF target, `gcc`, libbpf development headers/libraries, and bpftool), then build from this directory:

```bash
cd src/realtime
clang -O2 -g -target bpf -D__TARGET_ARCH_x86 \
  -I/usr/include/bpf \
  -c realtime_monitor.bpf.c -o realtime_monitor.bpf.o
gcc -O2 -Wall -Wextra -o realtime_pinner pinner.c -lbpf -lelf -lz
```

Use the matching `__TARGET_ARCH_*` define for a non-x86 host. Start a fresh collection session as root:

```bash
sudo ./realtime_pinner --obj ./realtime_monitor.bpf.o
```

Stop any previous collector first. The pinner replaces stale pins under `/sys/fs/bpf`, loads the specified object, attaches every required probe, and exits with an error rather than leaving a partial collector attached. It leaves the four maps pinned when stopped with Ctrl+C.

In a second terminal, run diagnosis from the repository root with the project environment:

```bash
cd /path/to/PerformanceBottleneckDiagnosis
source venv/bin/activate
sudo -E "$(which python)" -m src.realtime.predictor
```

Stop the predictor before starting a new pinner session. The normal predictor shutdown unpins maps; the benchmark intentionally retains them. The pinner does not launch workloads; run bounded workloads separately and size memory workloads conservatively for the host.

## Model artifacts

Run `src/approach3/champion_model_train_v7_2.ipynb` to write `champion_model_v7_shap.pkl`, `feature_cols_v7_shap.json`, and `label_thresholds_v7_shap.json` into `src/realtime/`. Run `src/attributionAnalysis/bottleneck_attrition_analysis_v2.ipynb` to write `baseline_original_best.pt`, `feature_scaler.pkl`, and the matching feature/target lists into `src/attributionAnalysis/pipeline_comparison_artifacts/`. Realtime configuration defaults to these artifacts and fails explicitly when any required file is absent or incompatible.

The realtime reader applies the saved binary decision threshold and rejects any feature-list/model-order mismatch. The attribution network is trained on the same 20 model inputs and the six existing cause labels.

## Map snapshot behavior

- The `/api/live` dashboard reader is read-only. The collector removes scheduler
  entries for each exiting TID and removes memory, syscall, and lock entries for
  a TGID only when its final thread exits. These maps represent active task
  lifetimes, not retained historical records; completed-process counters are
  intentionally retired to release hash-map capacity.
- Exit cleanup scans the relevant hash map only from `sched_process_exit`, not
  on scheduler events or dashboard refreshes. Cleanup errors are reported with
  `bpf_printk`; the collector does not silently treat a partial cleanup as
  complete. The maps have a fixed maximum size, so the exit scan is bounded by
  that map's entry count.
- The inference reader uses bpftool dumps only to enumerate candidate keys, validates every map's BTF/ABI, then consumes each current value with libbpf's atomic lookup-and-delete operation. Updates before a key's atomic consume are included; updates after it recreate the key for the next interval. Candidate keys absent because exit cleanup or another consumer already removed them are skipped.
- Atomic consumption is per key, not one transaction across all four maps. A process can exit between map reads, so an interval may omit its final counters if exit cleanup removes them before that key is consumed. A failure after some keys have been consumed retains those values in the reader process for its next successful interval; a reader-process crash cannot recover its in-memory partial interval.
- Map reading first dumps all four maps to obtain bounded candidate keys, then validates configured pinned-map IDs and BTF layouts before consumption. It opens each map and rechecks the descriptor ID, hash type, key size, and value size immediately before consuming entries. Missing pins, replaced/incompatible maps, malformed keys, and atomic-operation failures stop inference explicitly.
- The dashboard reader remains read-only. It can observe a mixed cross-map view while inference consumes keys; this is not an atomic four-map snapshot.
- The BPF object records only the raw values needed to compute the selected classifier and attribution inputs. Derived ratios and logarithms are computed in userspace to match training.
- The configured polling interval defaults to five seconds and targets cycle start-to-start cadence. If a cycle takes longer, the next starts immediately.
