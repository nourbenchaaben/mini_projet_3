# src/rank_and_eval.py
"""
Role:
This script performs document ranking for a set of queries using a TF-IDF vectorizer.
It also implements Rocchio relevance feedback to refine queries based on judged relevant documents.
It saves three outputs:
1. results.csv -> baseline ranking (no relevance feedback)
2. judgments_template.csv -> template for manual relevance judgments
3. results_rocchio.csv -> rankings after Rocchio query reformulation

"""

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
import pickle
from src.utils import load_docs
import os

# -----------------------------
# Load precomputed data
# -----------------------------
vec = pickle.load(open("outputs/tfidf_vectorizer.pkl", "rb"))  # TF-IDF vectorizer
X = sparse.load_npz("outputs/X.npz")  # Document-term matrix
docs, titles = load_docs()  # Load documents and optional titles

# -----------------------------
# Queries to rank
# -----------------------------
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

# -----------------------------
# Baseline ranking function
# -----------------------------
def rank_query(query, k=20):
    """
    Rank documents for a given query using cosine similarity.
    Returns top-k document IDs and scores.
    """
    qv = vec.transform([query])  # transform query to TF-IDF vector
    sims = cosine_similarity(qv, X).ravel()  # compute similarity with all docs
    top_idx = np.argsort(sims)[::-1][:k]  # get indices of top-k scores
    return list(zip(top_idx.tolist(), sims[top_idx].tolist()))

# -----------------------------
# Rocchio query reformulation
# -----------------------------
def rocchio_query_rewrite(query, judg_df, alpha=1.0, beta=0.75, gamma=0.15, topk=10):
    """
    Refine query using Rocchio relevance feedback:
    - alpha: weight for original query
    - beta: weight for relevant docs
    - gamma: weight for non-relevant docs
    - topk: only consider top-k results for feedback
    """
    qvec = vec.transform([query]).toarray()  # query vector
    subset = judg_df[(judg_df["query"] == query) & (judg_df["rank"] <= topk)]  # top-k judged docs
    pos_ids = subset[subset["relevant"] == 1]["doc_id"].tolist()  # relevant docs
    neg_ids = subset[subset["relevant"] == 0]["doc_id"].tolist()  # non-relevant docs

    # Create matrices for relevant and non-relevant documents
    Dpos = X[pos_ids].toarray() if pos_ids else np.zeros((0, X.shape[1]))
    Dneg = X[neg_ids].toarray() if neg_ids else np.zeros((0, X.shape[1]))

    # Rocchio formula: new query vector
    qp = alpha * qvec.copy()
    if Dpos.shape[0] > 0:
        qp += beta * Dpos.mean(axis=0)
    if Dneg.shape[0] > 0:
        qp -= gamma * Dneg.mean(axis=0)

    # Compute similarity of all docs with updated query
    sims = (X.dot(qp.T)).ravel()
    top_idx = np.argsort(sims)[::-1][:20]  # top 20 results
    return list(zip(top_idx.tolist(), sims[top_idx].tolist()))

# -----------------------------
# Create results directory
# -----------------------------
os.makedirs("outputs/results", exist_ok=True)

# -----------------------------
# Step 1: Baseline ranking
# -----------------------------
rows = []
for q in queries:
    ranked = rank_query(q, k=20)
    for rank, (doc_id, score) in enumerate(ranked, 1):
        rows.append({
            "query": q,
            "doc_id": int(doc_id),
            "score": float(score),
            "rank": rank,
            "relevant": ""  # empty for manual filling
        })

df = pd.DataFrame(rows)

# Save template for manual judgments ONLY if it doesn't exist
if not os.path.exists("outputs/results/judgments_template.csv"):
    df.to_csv("outputs/results/judgments_template.csv", index=False)
    print("Created outputs/results/judgments_template.csv")
else:
    print("outputs/results/judgments_template.csv already exists. Skipping overwrite.")

df[['query', 'doc_id', 'score', 'rank']].to_csv("outputs/results/results.csv", index=False)

print("Created outputs/results/results.csv")

# -----------------------------
# Step 2: Rocchio ranking
# -----------------------------
# **Load the filled judgments file**
df = pd.read_csv("outputs/results/judgments_template.csv")  # must be filled with 0/1 in 'relevant'

rows_rocchio = []
for q in queries:
    ranked = rocchio_query_rewrite(q, df, topk=10)
    for rank, (doc_id, score) in enumerate(ranked, 1):
        # Safely lookup relevance judgment, default to empty string if not found
        judgment_lookup = df[(df["query"]==q)&(df["doc_id"]==doc_id)]["relevant"].values
        relevant_val = judgment_lookup[0] if len(judgment_lookup) > 0 else ""
        
        rows_rocchio.append({
            "query": q,
            "doc_id": int(doc_id),
            "score": float(score),
            "rank": rank,
            "relevant": relevant_val
        })

rocchio_df = pd.DataFrame(rows_rocchio)
rocchio_df.to_csv("outputs/results/results_rocchio.csv", index=False)
print(" Rocchio results saved to outputs/results/results_rocchio.csv")


# -----------------------------
# Save template for manual judgments
# -----------------------------
# df.to_csv("outputs/results/judgments_template.csv", index=False) # Do not overwrite

# Save baseline system results (without 'relevant')
df[['query', 'doc_id', 'score', 'rank']].to_csv("outputs/results/results.csv", index=False)

print(" Created outputs/results/results.csv")
