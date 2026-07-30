from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from .config import RealtimeConfig, load_config
    from .realtime_reader import RealtimeReader, CLASS_NAMES, BOTTLENECK_CLASS
except ImportError:  # pragma: no cover - direct-script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from realtime.config import RealtimeConfig, load_config
    from realtime.realtime_reader import RealtimeReader, CLASS_NAMES, BOTTLENECK_CLASS


class Predictor:
    def __init__(self, config: RealtimeConfig | None = None) -> None:
        self.config = config or load_config()
        self.reader = RealtimeReader(self.config)
        self.debug_enabled = os.getenv("REALTIME_DEBUG", "").lower() in {"1", "true", "yes", "on"}

        if not self.config.model_path or not os.path.exists(self.config.model_path):
            raise RuntimeError(f"Missing model for realtime inference: {self.config.model_path}")

        self.model = self.reader.model

    def run_forever(self) -> None:
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

            # Sort by confidence (highest first)
            payload.sort(
                key=lambda x: x["confidence"] if x["confidence"] is not None else 0,
                reverse=True,
            )

            bottleneck_count = sum(1 for item in payload if item["is_bottleneck"])

            print("\n" + "=" * 80)
            print(f"Prediction Time : {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 80)
            print(f"{'PID':<8}{'CPU':<6}{'PROCESS':<22}{'CLASS':<7}{'PREDICTION':<20}{'CONF'}")
            print("-" * 80)

            for item in payload[:15]:  # Show top 15 only
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


def main() -> None:
    Predictor().run_forever()


if __name__ == "__main__":
    main()