from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

try:
    from .config import RealtimeConfig, load_config
    from .realtime_reader import RealtimeReader
except ImportError:  # pragma: no cover - direct-script fallback
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from realtime.config import RealtimeConfig, load_config
    from realtime.realtime_reader import RealtimeReader


class Predictor:
    def __init__(self, config: RealtimeConfig | None = None) -> None:
        self.config = config or load_config()
        self.reader = RealtimeReader(self.config)
        self.debug_enabled = os.getenv("REALTIME_DEBUG", "").lower() in {"1", "true", "yes", "on"}

        if not self.config.model_path or not os.path.exists(self.config.model_path):
            raise RuntimeError(f"Missing model for realtime inference: {self.config.model_path}")

        import joblib

        self.model = self.reader.model

    def run_forever(self) -> None:
        while True:
            result = self.reader.read_once()
            feature_frame = result["features"]
            predictions = self.model.predict(feature_frame)

            probabilities = None
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(feature_frame)

            payload = []

            LABELS = {
                "0": "CPU_BOUND",
                "1": "MEMORY_BOUND",
                "2": "IO_BOUND",
                "3": "LOCK_CONTENTION",
            }

            for index, row in result["rows"].iterrows():

                prediction = str(predictions[index])

                confidence = None
                if probabilities is not None:
                    confidence = max(probabilities[index]) * 100

                payload.append({
                    "pid": int(row["pid"]),
                    "cpu": int(row["cpu"]),
                    "comm": str(row.get("comm", "") or "-"),
                    "prediction": LABELS.get(prediction, prediction),
                    "confidence": confidence,
                })

            # Sort by confidence (highest first)
            payload.sort(
                key=lambda x: x["confidence"] if x["confidence"] is not None else 0,
                reverse=True,
            )

            print("\n" + "=" * 72)
            print(f"Prediction Time : {datetime.now().strftime('%H:%M:%S')}")
            print("=" * 72)
            print(f"{'PID':<8}{'CPU':<6}{'PROCESS':<22}{'PREDICTION':<20}{'CONF'}")
            print("-" * 72)

            for item in payload[:15]:          # Show top 15 only
                conf = (
                    f"{item['confidence']:.1f}%"
                    if item["confidence"] is not None
                    else "-"
                )

                print(
                    f"{item['pid']:<8}"
                    f"{item['cpu']:<6}"
                    f"{item['comm'][:20]:<22}"
                    f"{item['prediction']:<20}"
                    f"{conf}"
                )

            print("-" * 72)
            print(f"Processes monitored : {len(payload)}")
            print("=" * 72)
            time.sleep(self.config.polling_interval_s)


def main() -> None:
    Predictor().run_forever()


if __name__ == "__main__":
    main()