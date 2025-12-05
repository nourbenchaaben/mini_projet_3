# src/utils.py
"""
Role:
This utility module provides functions for loading documents and saving data in different formats.
It is a helper module used by other scripts such as:
- ablation_pipelines.py (preprocessing and vectorization)
- build_index.py (creating inverted index)
- rank_and_eval.py (ranking and Rocchio)
- evaluate.py (reading results)

"""

import json
import pandas as pd
from scipy import sparse
import pickle

# -----------------------------
# Load documents from CSV
# -----------------------------
def load_docs(path="data/News_Category_Dataset_v3.json"):
    """
    Load documents and titles from a JSON Lines file.
    Returns:
        docs: list of document texts (from 'short_description')
        titles: list of document titles (from 'headline')
    """
    docs = []
    titles = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            item = json.loads(line)
            docs.append(item.get("short_description", ""))
            titles.append(item.get("headline", ""))
    return docs, titles

# -----------------------------
# Save a Python object as JSON
# -----------------------------
def save_json(obj, path):
    """
    Save a Python object as a JSON file.
    Useful for inverted indices or other intermediate outputs.
    """
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=False)

# -----------------------------
# Save a Python object using pickle
# -----------------------------
def save_pickle(obj, path):
    """
    Serialize and save a Python object to a file using pickle.
    Used for saving vectorizers or other large Python objects.
    """
    with open(path, "wb") as f:
        pickle.dump(obj, f)

# -----------------------------
# Save a sparse matrix in NPZ format
# -----------------------------
def save_sparse_matrix(X, path):
    """
    Save a scipy sparse matrix to disk in .npz format.
    Used for saving document-term matrices efficiently.
    """
    sparse.save_npz(path, X)
