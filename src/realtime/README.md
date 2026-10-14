# Realtime inference pipeline

This userspace-only realtime pipeline reads the existing pinned BPF maps, reconstructs one row per `(pid, cpu)` entry in the same shape as the existing reader implementation, calls the shared ML feature engineering function, and emits one prediction per row.

## Notes
- It does not create a new eBPF collector or pinner.
- It consumes the existing pinned maps only.
- It uses the shared feature engineering entry point from the ML package.
- The predictor snapshots all four maps and then deletes the captured hash-map entries. This is not atomic with kernel-side map updates: updates racing with deletion can be lost. The configured polling interval (default 5 seconds) targets cycle start-to-start cadence; if a cycle takes longer than the interval, the next one starts immediately.
- Map reading launches one `bpftool` dump per map and clears captured entries before aggregation/inference. Pinned maps must support deleting entries; reset failures are surfaced and stop the current cycle.


steps for me : 

sudo ./pinner

cd /home/asad/Code/FYP/PerformanceBottleneckDiagnosis
source venv/bin/activate

sudo -E $(which python) -m src.realtime.predictor

stress-ng --cpu 0 --stream 0 --vm 1 --vm-bytes 400M
