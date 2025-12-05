# src/evaluate.py
"""
Role: 
This script evaluates the quality of a document retrieval system. 
It computes standard IR metrics such as Precision@1, Precision@5, and MAP (Mean Average Precision)
by comparing system results (ranked documents) with manually filled relevance judgments.

"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Paths to system results and relevance judgments
RESULTS_PATH = Path("outputs/results/results.csv")  # system ranking
JUDGMENTS_PATH = Path("outputs/results/judgments_template.csv")  # manually filled relevance

# -----------------------------
# Metric computation functions
# -----------------------------

def precision_at_k(relevant_list, k):
    """Compute precision at rank k: fraction of top-k documents that are relevant."""
    return sum(relevant_list[:k]) / k

def average_precision(relevant_list):
    """Compute Average Precision (AP) for a query."""
    score = 0.0
    rel_count = 0
    for i, rel in enumerate(relevant_list):
        if rel == 1:  # if the document is relevant
            rel_count += 1
            score += rel_count / (i + 1)  # precision at this position
    return score / max(rel_count, 1)  # avoid division by zero

def recall_at_k(relevant_list, total_relevant_in_collection, k):
    """Compute recall at rank k."""
    if total_relevant_in_collection == 0:
        return 0.0
    return sum(relevant_list[:k]) / total_relevant_in_collection

# -----------------------------
# Utility functions
# -----------------------------

def safe_read_csv(path):
    """
    Reads a CSV safely:
    - Handles UTF-8 BOM if present
    - Strips whitespace from column names
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    return df

def ensure_rank(df):
    """
    Ensures a 'rank' column exists:
    - If 'score' is available, ranks documents descending by score
    - Else, assigns rank based on file order
    """
    if "rank" in df.columns:
        return df
    if "score" in df.columns:
        df = df.sort_values(["query", "score"], ascending=[True, False]).reset_index(drop=True)
        df["rank"] = df.groupby("query").cumcount() + 1
        # print("Created 'rank' from 'score' (descending).")
    else:
        df = df.reset_index(drop=True)
        df["rank"] = df.groupby("query").cumcount() + 1
        # print("Created 'rank' from file order (no 'score' column).")
    return df

def harmonize_doc_id_types(results_df, jud_df):
    """
    Ensures 'doc_id' has consistent type in both dataframes to allow merging.
    - Converts float doc_ids like 1.0 to int 1
    - Strips whitespace
    """
    for df in (results_df, jud_df):
        if "doc_id" in df.columns:
            df["doc_id"] = df["doc_id"].astype(str).str.strip()
            def tryint(x):
                try:
                    if "." in x:
                        x2 = float(x)
                        if x2.is_integer():
                            return int(x2)
                    return int(x)
                except:
                    return x
            df["doc_id"] = df["doc_id"].apply(tryint)
    return results_df, jud_df

# -----------------------------
# Main evaluation workflow
# -----------------------------

def evaluate_system(results_path, judgments_path):
    # Check input files exist
    if not Path(results_path).exists():
        print(f" Missing results file: {results_path}")
        return None
    if not Path(judgments_path).exists():
        print(f" Missing judgments file: {judgments_path}")
        return None

    # Load data
    results = safe_read_csv(results_path)
    judgments = safe_read_csv(judgments_path)

    # Ensure rank column exists
    results = ensure_rank(results)

    # Harmonize doc_id types between results and judgments
    results, judgments = harmonize_doc_id_types(results, judgments)

    # Merge system results with relevance judgments
    df = results.merge(judgments, on=["query", "doc_id"], how="inner", suffixes=("_sys","_true"))
    
    if df.empty:
        print(f" Warning: No overlapping query/doc_id found between {results_path} and {judgments_path}")
        return None

    # Handle rank column renaming
    if "rank" not in df.columns and "rank_sys" in df.columns:
        df = df.rename(columns={"rank_sys": "rank"})
        
    # Handle relevant column renaming
    if "relevant" not in df.columns and "relevant_true" in df.columns:
        df = df.rename(columns={"relevant_true": "relevant"})

    # Sort merged dataframe by query and rank
    df = df.sort_values(["query", "rank"])

    # Compute evaluation metrics
    p5_list, p10_list, r10_list, ap_list = [], [], [], []
    
    # Calculate total relevant documents per query from the judgments file (assuming it's complete for the top-k pooled)
    # Note: In a real scenario, we need the total relevant in the entire collection. 
    # Here we assume the judgments file contains all known relevant documents for these queries.
    total_rel_per_query = judgments[judgments["relevant"] == 1].groupby("query").size().to_dict()

    for q in df["query"].unique():
        qdf = df[df["query"] == q].sort_values("rank")
        if "relevant" not in qdf.columns:
            print(" judgments file must contain 'relevant' column with 0/1 values.")
            return None
        rel = list(qdf["relevant"].astype(int))

        # Pad fewer than 10 results with zeros for P@10/R@10
        if len(rel) < 10:
            rel = rel + [0] * (10 - len(rel))

        total_rel = total_rel_per_query.get(q, 0)

        p5_list.append(precision_at_k(rel, 5))
        p10_list.append(precision_at_k(rel, 10))
        r10_list.append(recall_at_k(rel, total_rel, 10))
        ap_list.append(average_precision(rel))

    metrics = {
        "P@5": float(np.mean(p5_list)),
        "P@10": float(np.mean(p10_list)),
        "Recall@10": float(np.mean(r10_list)),
        "MAP": float(np.mean(ap_list))
    }
    return metrics

def main():
    print("Evaluating Baseline...")
    metrics_base = evaluate_system(RESULTS_PATH, JUDGMENTS_PATH)
    if metrics_base:
        print("\n=== Baseline Evaluation Results ===")
        for k, v in metrics_base.items():
            print(f"{k}: {v:.4f}")

    # Check for Rocchio results
    ROCCHIO_PATH = Path("outputs/results/results_rocchio.csv")
    if ROCCHIO_PATH.exists():
        print("\nEvaluating Rocchio...")
        metrics_rocchio = evaluate_system(ROCCHIO_PATH, JUDGMENTS_PATH)
        if metrics_rocchio:
            print("\n=== Rocchio Evaluation Results ===")
            for k, v in metrics_rocchio.items():
                print(f"{k}: {v:.4f}")
    else:
        print("\n(No Rocchio results found to evaluate)")

if __name__ == "__main__":
    main()
