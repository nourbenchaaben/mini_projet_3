
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.evaluate import evaluate_system

OUTPUTS_DIR = Path("outputs")
PLOTS_DIR = OUTPUTS_DIR / "plots"
RESULTS_DIR = OUTPUTS_DIR / "results"
ABLATION_DIR = OUTPUTS_DIR / "ablation"
JUDGMENTS_PATH = RESULTS_DIR / "judgments_template.csv"

def plot_baseline_vs_rocchio():
    print("Plotting Baseline vs Rocchio...")
    
    # Get metrics
    base_path = RESULTS_DIR / "results.csv"
    rocchio_path = RESULTS_DIR / "results_rocchio.csv"
    
    if not base_path.exists() or not rocchio_path.exists() or not JUDGMENTS_PATH.exists():
        print("  Missing files for comparison.")
        return

    m_base = evaluate_system(base_path, JUDGMENTS_PATH)
    m_rocchio = evaluate_system(rocchio_path, JUDGMENTS_PATH)
    
    if not m_base or not m_rocchio:
        print("  Could not compute metrics.")
        return

    # Prepare data
    data = {
        "Metric": ["MAP", "P@5", "P@10", "Recall@10"] * 2,
        "Score": [m_base["MAP"], m_base["P@5"], m_base["P@10"], m_base["Recall@10"],
                  m_rocchio["MAP"], m_rocchio["P@5"], m_rocchio["P@10"], m_rocchio["Recall@10"]],
        "System": ["Baseline"] * 4 + ["Rocchio"] * 4
    }
    df = pd.DataFrame(data)
    
    # Plot
    plt.figure(figsize=(10, 6))
    sns.barplot(data=df, x="Metric", y="Score", hue="System")
    plt.title("Baseline vs Rocchio Performance")
    plt.ylim(0, 1.1)
    plt.savefig(PLOTS_DIR / "baseline_vs_rocchio.png")
    print("  Saved baseline_vs_rocchio.png")
    plt.close()

def plot_ablation():
    print("Plotting Ablation Studies...")
    
    metrics_path = ABLATION_DIR / "ablation_metrics.csv"
    if not metrics_path.exists():
        print("  Missing ablation_metrics.csv. Run src/run_ablation.py first.")
        return
        
    df = pd.read_csv(metrics_path)
    
    # Melt for plotting
    df_melted = df.melt(id_vars="Pipeline", var_name="Metric", value_name="Score")
    
    # Plot MAP comparison
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_melted[df_melted["Metric"] == "MAP"], x="Pipeline", y="Score")
    plt.title("MAP by Pipeline Configuration")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "ablation_map.png")
    print("  Saved ablation_map.png")
    plt.close()

def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    plot_baseline_vs_rocchio()
    plot_ablation()

if __name__ == "__main__":
    main()
