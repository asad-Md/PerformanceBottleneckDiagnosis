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

- The reader dumps all four pinned maps and then deletes the captured hash-map entries. This is not atomic with kernel-side updates; updates racing with deletion can be lost.
- Map reading launches one `bpftool` dump per map. Missing pins, malformed payloads, and reset failures stop inference explicitly.
- The BPF object records only the raw values needed to compute the selected classifier and attribution inputs. Derived ratios and logarithms are computed in userspace to match training.
- The configured polling interval defaults to five seconds and targets cycle start-to-start cadence. If a cycle takes longer, the next starts immediately.
