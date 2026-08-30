import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from flood_ai.data import save_demo_data
from flood_ai.train import train

if __name__ == "__main__":
    path = save_demo_data()
    metrics = train(path)
    print(json.dumps(metrics, indent=2))
