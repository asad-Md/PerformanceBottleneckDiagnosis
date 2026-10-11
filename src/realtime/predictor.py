from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

try:
    from .config import RealtimeConfig, load_config
    from .realtime_reader import RealtimeReader
    from .utils import unpin_maps
except ImportError:  # pragma: no cover - direct-script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from realtime.config import RealtimeConfig, load_config
    from realtime.realtime_reader import RealtimeReader
    from realtime.utils import unpin_maps


BINARY_CLASS_NAMES = {0: "Not-High", 1: "High"}
BINARY_BOTTLENECK_CLASS = 1


class Predictor:
    DISPLAY_MODE = os.getenv("REALTIME_DISPLAY_MODE", "top_per_class")
    TOP_N_PER_CLASS = int(os.getenv("REALTIME_TOP_N_PER_CLASS", "4"))
    MAX_ROWS_SHOWN = int(os.getenv("REALTIME_MAX_ROWS_SHOWN", "16"))

    def __init__(
        self,
        config: RealtimeConfig | None = None,
        *,
        benchmark_mode: bool = False,
        cleanup_maps_on_exit: bool = True,
    ) -> None:
        self.config = config or load_config()
        self.reader = RealtimeReader(self.config, collect_timings=benchmark_mode)
        self.cleanup_maps_on_exit = cleanup_maps_on_exit
        self.debug_enabled = os.getenv("REALTIME_DEBUG", "").lower() in {"1", "true", "yes", "on"}

        if not self.config.model_path or not os.path.exists(self.config.model_path):
            raise RuntimeError(f"Missing model for realtime inference: {self.config.model_path}")

        self.model = self.reader.model
        self.class_names = BINARY_CLASS_NAMES.copy()
        self.bottleneck_class = BINARY_BOTTLENECK_CLASS

        if hasattr(self.model, "classes_"):
            try:
                model_classes = tuple(int(c) for c in self.model.classes_)
                if set(model_classes) == {0, 1}:
                    self.class_names = {cls: BINARY_CLASS_NAMES[cls] for cls in sorted(model_classes)}
                else:
                    self._log_warning(
                        "Loaded model classes are not binary {0, 1}. Predictions will still be shown, "
                        "but class labels may be incorrect."
                    )
            except Exception:
                self._log_warning("Failed to normalize model classes from model.classes_. Using default binary labels.")

        self._map_paths = [
            self.config.sched_map_path,
            self.config.mem_map_path,
            self.config.syscall_map_path,
            self.config.lock_map_path,
        ]

    def _log_warning(self, message: str) -> None:
        if self.debug_enabled:
            print(f"[predictor][warning] {message}")

    def run_forever(self) -> None:
        try:
            self._run_forever_inner()
        except KeyboardInterrupt:
            print("\n[predictor] Ctrl+C received, shutting down...")
        finally:
            self.cleanup()

    def run_benchmark(
        self,
        cycle_callback: Callable[[dict[str, object]], None],
        *,
        duration_s: float,
        warmup_s: float = 0.0,
    ) -> None:
        if not self.reader.collect_timings:
            raise RuntimeError("Benchmark mode requires stage timing collection")
        if duration_s <= 0 or warmup_s < 0:
            raise ValueError("Benchmark duration must be positive and warm-up duration cannot be negative")
        if self.cleanup_maps_on_exit:
            raise RuntimeError("Benchmark mode must not unpin maps during shutdown")
        self._run_forever_inner(cycle_callback, duration_s=duration_s, warmup_s=warmup_s)

    def cleanup(self) -> None:
        if not self.cleanup_maps_on_exit:
            return
        print("[predictor] unpinning BPF maps...")
        unpin_maps(self._map_paths)

    def _run_forever_inner(
        self,
        cycle_callback: Callable[[dict[str, object]], None] | None = None,
        *,
        duration_s: float | None = None,
        warmup_s: float = 0.0,
    ) -> None:
        benchmark_start_ns = time.perf_counter_ns() if duration_s is not None else None
        warmup_end_ns = benchmark_start_ns + int(warmup_s * 1_000_000_000) if benchmark_start_ns is not None else None
        benchmark_end_ns = (
            benchmark_start_ns + int((warmup_s + duration_s) * 1_000_000_000)
            if benchmark_start_ns is not None and duration_s is not None
            else None
        )
        previous_cycle_start_ns: int | None = None
        cycle_id = 0

        while True:
            if benchmark_end_ns is not None and time.perf_counter_ns() >= benchmark_end_ns:
                break

            cycle_id += 1
            cycle_start_ns = time.perf_counter_ns() if cycle_callback is not None else None
            actual_interval_ms = (
                (cycle_start_ns - previous_cycle_start_ns) / 1_000_000
                if cycle_start_ns is not None and previous_cycle_start_ns is not None
                else None
            )
            try:
                result = self.reader.predict()
            except Exception as exc:
                if cycle_callback is not None and cycle_start_ns is not None:
                    cycle_end_ns = time.perf_counter_ns()
                    timings = self.reader.last_stage_timings_ns
                    cycle_callback({
                        "cycle_id": cycle_id,
                        "phase": "warmup" if warmup_end_ns is not None and cycle_start_ns < warmup_end_ns else "measurement",
                        "cycle_start_ns": cycle_start_ns,
                        "cycle_end_ns": cycle_end_ns,
                        "map_read_ms": self._timing_ms(timings.get("map_read_ns")),
                        "aggregation_ms": self._timing_ms(timings.get("aggregation_ns")),
                        "feature_ms": self._timing_ms(timings.get("feature_ns")),
                        "rf_inference_ms": self._timing_ms(timings.get("rf_inference_ns")),
                        "classifier_inference_ms": self._timing_ms(timings.get("rf_inference_ns")),
                        "model_class": type(self.reader.model).__name__,
                        "mlp_inference_ms": None,
                        "output_ms": None,
                        "detection_latency_ms": None,
                        "total_cycle_ms": (cycle_end_ns - cycle_start_ns) / 1_000_000,
                        "records_processed": None,
                        "high_predictions": None,
                        "actual_cycle_interval_ms": actual_interval_ms,
                        "deadline_missed": None,
                        "cycle_status": "failed",
                        "error": f"{type(exc).__name__}: {exc}",
                    })
                raise

            prediction_end_ns = time.perf_counter_ns() if cycle_callback is not None else None
            output_start_ns = time.perf_counter_ns() if cycle_callback is not None else None
            predictions = result["predictions"]
            probabilities = result["probabilities"]
            rows_df = result["rows"]

            payload: list[dict[str, object]] = []
            for position, (_, row) in enumerate(rows_df.iterrows()):
                pred_class = int(predictions[position])
                confidence = None
                if probabilities is not None:
                    confidence = float(max(probabilities[position]) * 100)

                payload.append({
                    "pid": int(row["pid"]),
                    "cpu": int(row["cpu"]),
                    "comm": str(row.get("comm", "") or "-"),
                    "pred_class": pred_class,
                    "prediction": self.class_names.get(pred_class, str(pred_class)),
                    "is_bottleneck": pred_class == self.bottleneck_class,
                    "confidence": confidence,
                })

            rows_to_show = self._select_display_rows(payload)
            bottleneck_count = sum(1 for item in payload if item["is_bottleneck"])

            print("\n" + "=" * 80)
            print(f"Prediction Time : {datetime.now().strftime('%H:%M:%S')}  |  display mode: {self.DISPLAY_MODE}")
            print("=" * 80)
            print(f"{'PID':<8}{'CPU':<6}{'PROCESS':<22}{'CLASS':<7}{'PREDICTION':<20}{'CONF'}")
            print("-" * 80)

            for item in rows_to_show:
                conf = f"{item['confidence']:.1f}%" if item["confidence"] is not None else "-"
                marker = " <-- BOTTLENECK" if item["is_bottleneck"] else ""
                print(
                    f"{item['pid']:<8}"
                    f"{item['cpu']:<6}"
                    f"{item['comm'][:20]:<22}"
                    f"{item['pred_class']:<7}"
                    f"{item['prediction']:<20}"
                    f"{conf}{marker}"
                )

            print("-" * 80)
            print(f"Processes monitored : {len(payload)}   |   Bottlenecks (class {self.bottleneck_class}) : {bottleneck_count}")
            print("=" * 80)

            if cycle_callback is not None and cycle_start_ns is not None:
                output_end_ns = time.perf_counter_ns()
                cycle_end_ns = time.perf_counter_ns()
                timings = self.reader.last_stage_timings_ns
                cycle_callback({
                    "cycle_id": cycle_id,
                    "phase": "warmup" if warmup_end_ns is not None and cycle_start_ns < warmup_end_ns else "measurement",
                    "cycle_start_ns": cycle_start_ns,
                    "cycle_end_ns": cycle_end_ns,
                    "map_read_ms": self._timing_ms(timings.get("map_read_ns")),
                    "aggregation_ms": self._timing_ms(timings.get("aggregation_ns")),
                    "feature_ms": self._timing_ms(timings.get("feature_ns")),
                    "rf_inference_ms": self._timing_ms(timings.get("rf_inference_ns")),
                    "classifier_inference_ms": self._timing_ms(timings.get("rf_inference_ns")),
                    "model_class": type(self.reader.model).__name__,
                    "mlp_inference_ms": None,
                    "output_ms": (output_end_ns - output_start_ns) / 1_000_000 if output_start_ns is not None else None,
                    "detection_latency_ms": (
                        (prediction_end_ns - cycle_start_ns) / 1_000_000
                        if prediction_end_ns is not None
                        else None
                    ),
                    "total_cycle_ms": (cycle_end_ns - cycle_start_ns) / 1_000_000,
                    "records_processed": len(rows_df),
                    "high_predictions": bottleneck_count,
                    "actual_cycle_interval_ms": actual_interval_ms,
                    "deadline_missed": None,
                    "cycle_status": "completed",
                    "error": None,
                })

            if cycle_start_ns is not None:
                previous_cycle_start_ns = cycle_start_ns

            if benchmark_end_ns is not None:
                remaining_s = (benchmark_end_ns - time.perf_counter_ns()) / 1_000_000_000
                if remaining_s <= 0:
                    break
                time.sleep(min(self.config.polling_interval_s, remaining_s))
            else:
                time.sleep(self.config.polling_interval_s)

    @staticmethod
    def _timing_ms(duration_ns: int | None) -> float | None:
        return duration_ns / 1_000_000 if duration_ns is not None else None

    def _select_display_rows(self, payload: list[dict]) -> list[dict]:
        if self.DISPLAY_MODE == "top_per_class":
            selected: list[dict] = []
            for cls in sorted(self.class_names.keys(), reverse=True):
                class_rows = sorted(
                    (item for item in payload if item["pred_class"] == cls),
                    key=lambda x: x["confidence"] if x["confidence"] is not None else 0,
                    reverse=True,
                )
                selected.extend(class_rows[: self.TOP_N_PER_CLASS])
            return selected[: self.MAX_ROWS_SHOWN]

        ordered = sorted(
            payload,
            key=lambda x: (x["pred_class"], x["confidence"] if x["confidence"] is not None else 0),
            reverse=True,
        )
        return ordered[: self.MAX_ROWS_SHOWN]


def main() -> None:
    Predictor().run_forever()


if __name__ == "__main__":
    main()
