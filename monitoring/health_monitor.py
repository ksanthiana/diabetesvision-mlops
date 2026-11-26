
import csv
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

API_URL = "https://diabetesvision-mlops.onrender.com/health"
LOG_FILE = Path("monitoring/health_logs.csv")


def check_api():
    start = time.perf_counter()

    try:
        with urllib.request.urlopen(API_URL, timeout=90) as response:
            elapsed = round((time.perf_counter() - start) * 1000, 2)
            data = json.loads(response.read().decode())

            return {
                "timestamp": datetime.now().isoformat(),
                "status": data.get("status", "unknown"),
                "model_loaded": data.get("model_loaded", False),
                "response_time_ms": elapsed
            }

    except Exception as error:
        return {
            "timestamp": datetime.now().isoformat(),
            "status": "offline",
            "model_loaded": False,
            "response_time_ms": "",
            "error": str(error)
        }


def main():
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    file_exists = LOG_FILE.exists()

    with LOG_FILE.open("a", newline="", encoding="utf-8") as file:
        fields = [
            "timestamp", "status", "model_loaded",
            "response_time_ms", "error"
        ]
        writer = csv.DictWriter(file, fieldnames=fields)

        if not file_exists:
            writer.writeheader()

        print("DiabetesVision API monitoring started.")

        try:
            while True:
                result = check_api()
                writer.writerow(result)
                file.flush()
                print(result)
                time.sleep(60)

        except KeyboardInterrupt:
            print("Monitoring stopped.")


if __name__ == "__main__":
    main()
