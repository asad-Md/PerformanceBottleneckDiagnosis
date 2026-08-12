from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

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

    def __init__(self, config: RealtimeConfig | None = None) -> None:
        self.config = config or load_config()
        self.reader = RealtimeReader(self.config)
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

    def cleanup(self) -> None:
        print("[predictor] unpinning BPF maps...")
        unpin_maps(self._map_paths)

    def _run_forever_inner(self) -> None:
        while True:
            result = self.reader.predict()
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
            time.sleep(self.config.polling_interval_s)

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
