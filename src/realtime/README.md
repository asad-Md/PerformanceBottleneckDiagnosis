# Realtime inference pipeline

This userspace-only realtime pipeline reads the existing pinned BPF maps, reconstructs one row per `(pid, cpu)` entry in the same shape as the existing reader implementation, calls the shared ML feature engineering function, and emits one prediction per row.

## Notes
- It does not create a new eBPF collector or pinner.
- It consumes the existing pinned maps only.
- It uses the shared feature engineering entry point from the ML package.
