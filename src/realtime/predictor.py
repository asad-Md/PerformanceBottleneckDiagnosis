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

        self.model = joblib.load(self.config.model_path)

    def run_forever(self) -> None:
        while True:
            result = self.reader.read_once()
            feature_values = result["features"].to_numpy(dtype=float)
            predictions = list(self.model.predict(feature_values))
            probabilities = None
            if hasattr(self.model, "predict_proba"):
                probabilities = self.model.predict_proba(feature_values)

            payload = []
            for index, row in result["rows"].iterrows():
                item = {
                    "timestamp": datetime.now().isoformat(timespec="seconds"),
                    "pid": int(row["pid"]),
                    "cpu": int(row["cpu"]),
                    "comm": str(row.get("comm", "")),
                    "prediction": str(predictions[index]),
                }
                if probabilities is not None:
                    item["probabilities"] = probabilities[index].tolist()
                payload.append(item)

            if self.debug_enabled:
                print("[realtime][debug] prediction summary: " + json.dumps(payload, indent=2, sort_keys=True))
            else:
                print(json.dumps(payload, indent=2, sort_keys=True))
            time.sleep(self.config.polling_interval_s)


def main() -> None:
    Predictor().run_forever()


if __name__ == "__main__":
    main()
