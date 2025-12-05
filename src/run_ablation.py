
import pandas as pd
import numpy as np
import pickle
from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys
import os

# Add src to path to allow imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.evaluate import evaluate_system

# Paths
ABLATION_DIR = Path("outputs/ablation")
RESULTS_DIR = Path("outputs/results")
JUDGMENTS_PATH = RESULTS_DIR / "judgments_template.csv"

# Queries (same as in rank_and_eval.py)
queries = [
    "election results",
    "earthquake Japan",
    "climate summit",
    "trade negotiations",
    "public health emergency",
    "space mission launch",
    "sanctions policy",
    "wildfire response"
]

def rank_query(query, vec, X, k=20):
    """Rank documents for a query using a specific vectorizer and matrix."""
    qv = vec.transform([query])
    sims = cosine_similarity(qv, X).ravel()
    top_idx = np.argsort(sims)[::-1][:k]
    return list(zip(top_idx.tolist(), sims[top_idx].tolist()))

def run_pipeline(pipeline_name):
    print(f"Running pipeline: {pipeline_name}")
    
    # Load model
    vec_path = ABLATION_DIR / f"{pipeline_name}_vectorizer.pkl"
    X_path = ABLATION_DIR / f"{pipeline_name}_X.npz"
    
    if not vec_path.exists() or not X_path.exists():
        print(f"  Missing model files for {pipeline_name}. Run ablation_pipelines.py first.")
        return None

    vec = pickle.load(open(vec_path, "rb"))
    X = sparse.load_npz(X_path)
    
    # Run queries
    rows = []
    for q in queries:
        ranked = rank_query(q, vec, X, k=20)
        for rank, (doc_id, score) in enumerate(ranked, 1):
            rows.append({
                "query": q,
                "doc_id": int(doc_id),
                "score": float(score),
                "rank": rank
            })
            
    # Save results
    df = pd.DataFrame(rows)
    out_path = ABLATION_DIR / f"results_{pipeline_name}.csv"
    df.to_csv(out_path, index=False)
    
    # Evaluate
    if JUDGMENTS_PATH.exists():
        metrics = evaluate_system(out_path, JUDGMENTS_PATH)
        return metrics
    else:
        print("  Judgments file missing, cannot evaluate.")
        return None

def main():
    pipelines = [
        "A_stop_on_stem_off",
        "B_stop_off_stem_off",
        "C_stop_on_stem_on",
        "D_stop_off_stem_on"
    ]
    
    all_metrics = []
    
    for p in pipelines:
        m = run_pipeline(p)
        if m:
            m["Pipeline"] = p
            all_metrics.append(m)
            
    if all_metrics:
        df_metrics = pd.DataFrame(all_metrics)
        print("\n=== Ablation Results ===")
        print(df_metrics.to_string(index=False))
        
        # Save metrics summary
        df_metrics.to_csv(ABLATION_DIR / "ablation_metrics.csv", index=False)
    else:
        print("\nNo metrics computed (check if judgments file exists and models are generated).")

if __name__ == "__main__":
    main()
