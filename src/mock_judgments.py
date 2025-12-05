
import pandas as pd
import numpy as np
from pathlib import Path

JUDGMENTS_PATH = Path("outputs/results/judgments_template.csv")

def mock_judgments():
    if not JUDGMENTS_PATH.exists():
        print("judgments_template.csv not found.")
        return

    df = pd.read_csv(JUDGMENTS_PATH)
    
    # Randomly assign 0 or 1 to 'relevant' column
    # Let's make it slightly biased towards 0 (irrelevant)
    np.random.seed(42)
    df["relevant"] = np.random.choice([0, 1], size=len(df), p=[0.7, 0.3])
    
    df.to_csv(JUDGMENTS_PATH, index=False)
    print(f"Mocked judgments saved to {JUDGMENTS_PATH}")

if __name__ == "__main__":
    mock_judgments()
