from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from .config import RealtimeConfig, load_config
    from .realtime_reader import RealtimeReader, CLASS_NAMES, BOTTLENECK_CLASS
    from .utils import unpin_maps
except ImportError:  # pragma: no cover - direct-script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from realtime.config import RealtimeConfig, load_config
    from realtime.realtime_reader import RealtimeReader, CLASS_NAMES, BOTTLENECK_CLASS
    from realtime.utils import unpin_maps


class Predictor:
    # "class_then_confidence" — group by severity class (High->Normal), sort by confidence
    #     within each class. Bottlenecks always dominate the top of the list.
    # "top_per_class"         — take the N highest-confidence rows from EACH class (High,
    #     Medium, Low, Normal), still ordered severity-first. Guarantees every class is
    #     represented even when one class (e.g. Normal) vastly outnumbers the others.
    DISPLAY_MODE = os.getenv("REALTIME_DISPLAY_MODE", "top_per_class")  # or "class_then_confidence"
    TOP_N_PER_CLASS = int(os.getenv("REALTIME_TOP_N_PER_CLASS", "4"))
    MAX_ROWS_SHOWN = int(os.getenv("REALTIME_MAX_ROWS_SHOWN", "16"))

    def __init__(self, config: RealtimeConfig | None = None) -> None:
        self.config = config or load_config()
        self.reader = RealtimeReader(self.config)
        self.debug_enabled = os.getenv("REALTIME_DEBUG", "").lower() in {"1", "true", "yes", "on"}

        if not self.config.model_path or not os.path.exists(self.config.model_path):
            raise RuntimeError(f"Missing model for realtime inference: {self.config.model_path}")

        self.model = self.reader.model

        self._map_paths = [
            self.config.sched_map_path,
            self.config.mem_map_path,
            self.config.syscall_map_path,
            self.config.lock_map_path,
        ]

    def run_forever(self) -> None:
        try:
            self._run_forever_inner()
        except KeyboardInterrupt:
            print("\n[predictor] Ctrl+C received, shutting down...")
        finally:
            self.cleanup()

    def cleanup(self) -> None:
        """Unpin the BPF maps (delete their bpffs pin files) so `sudo ls -l /sys/fs/bpf` comes
        back empty after this process exits. Pinning is just a filesystem reference in bpffs;
        removing the pin file drops it. Runs on Ctrl+C and on any other exit out of run_forever."""
        print("[predictor] unpinning BPF maps...")
        unpin_maps(self._map_paths)

    def _run_forever_inner(self) -> None:
        while True:
            result = self.reader.predict()
            predictions = result["predictions"]
            probabilities = result["probabilities"]
            rows_df = result["rows"]

            payload = []
            # Iterate positionally — predictions/probabilities are plain numpy arrays aligned
            # with rows_df's row order, not with rows_df's pandas index.
            for position, (_, row) in enumerate(rows_df.iterrows()):
                pred_class = int(predictions[position])

                confidence = None
                if probabilities is not None:
                    confidence = float(max(probabilities[position]) * 100)

                payload.append({
                    "pid": int(row["pid"]),
                    "cpu": int(row["cpu"]),
                    "comm": str(row.get("comm", "") or "-"),
                    "pred_class": pred_class,  # raw 0/1/2/3 stall-severity class
                    "prediction": CLASS_NAMES.get(pred_class, str(pred_class)),
                    "is_bottleneck": pred_class == BOTTLENECK_CLASS,
                    "confidence": confidence,
                })

            bottleneck_count = sum(1 for item in payload if item["is_bottleneck"])
            rows_to_show = self._select_display_rows(payload)

            print("\n" + "=" * 80)
            print(f"Prediction Time : {datetime.now().strftime('%H:%M:%S')}  |  display mode: {self.DISPLAY_MODE}")
            print("=" * 80)
            print(f"{'PID':<8}{'CPU':<6}{'PROCESS':<22}{'CLASS':<7}{'PREDICTION':<20}{'CONF'}")
            print("-" * 80)

            for item in rows_to_show:
                conf = (
                    f"{item['confidence']:.1f}%"
                    if item["confidence"] is not None
                    else "-"
                )
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
            print(f"Processes monitored : {len(payload)}   |   Bottlenecks (class {BOTTLENECK_CLASS}) : {bottleneck_count}")
            print("=" * 80)
            time.sleep(self.config.polling_interval_s)

    def _select_display_rows(self, payload: list[dict]) -> list[dict]:
        """Picks which rows to print, per self.DISPLAY_MODE. Both modes always order
        severity class High(3) -> Normal(0) first; they only differ in how much of each
        class gets shown."""
        if self.DISPLAY_MODE == "top_per_class":
            # Best few (by confidence) from EVERY class, so Medium/Low/Normal don't get
            # crowded out by however many thousand Normal-class rows exist.
            selected: list[dict] = []
            for cls in sorted(CLASS_NAMES.keys(), reverse=True):  # 3, 2, 1, 0
                class_rows = sorted(
                    (item for item in payload if item["pred_class"] == cls),
                    key=lambda x: x["confidence"] if x["confidence"] is not None else 0,
                    reverse=True,
                )
                selected.extend(class_rows[: self.TOP_N_PER_CLASS])
            return selected[: self.MAX_ROWS_SHOWN]

        # "class_then_confidence" — one global list, grouped by class, confidence-sorted
        # within each class.
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