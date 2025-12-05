# src/vectorize.py

"""
Role:
This script converts the text corpus into a TF-IDF representation.
It saves both the vectorizer and the document-term matrix for later use in ranking and retrieval.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from src.utils import load_docs, save_pickle, save_sparse_matrix
import os

# -----------------------------
# Load documents
# -----------------------------
docs, _ = load_docs()  # load text corpus (titles ignored here)

# Ensure outputs folder exists
os.makedirs("outputs", exist_ok=True)

# -----------------------------
# Create TF-IDF vectorizer
# -----------------------------
vec = TfidfVectorizer(stop_words="english", min_df=3)  # remove stopwords, ignore terms in <3 docs

# Fit the vectorizer and transform the documents into a sparse matrix
X = vec.fit_transform(docs)

# -----------------------------
# Save outputs for later use
# -----------------------------
save_pickle(vec, "outputs/tfidf_vectorizer.pkl")  # save the trained vectorizer
save_sparse_matrix(X, "outputs/X.npz")           # save the document-term matrix

print("Vectorizer + matrix saved. Vocab size:", len(vec.vocabulary_))
