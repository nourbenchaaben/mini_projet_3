"""
build_index.py
Builds a global TF-IDF vectorizer (without ablation variations) on the corpus.
Saves TF-IDF vectorizer (tfidf_vectorizer.pkl) and matrix (X.npz) for general ranking.
"""


import nltk  # for tokenization
from collections import defaultdict, Counter  # for building inverted index
from src.utils import load_docs, save_json  # helper functions

# Download tokenizer if not already present
nltk.download("punkt", quiet=True)

# ------------------------------
# Load all documents
# ------------------------------
docs, _ = load_docs()  # load document texts (titles ignored here)

# ------------------------------
# Initialize inverted index
# ------------------------------
# defaultdict(dict) creates a dictionary where each key is a term
# and its value is another dictionary {doc_id: term_frequency}
inv = defaultdict(dict)

# ------------------------------
# Build the inverted index
# ------------------------------
for doc_id, doc in enumerate(docs):
    # Tokenize the document, lowercase, and keep only alphanumeric tokens
    tokens = [t.lower() for t in nltk.word_tokenize(doc) if t.isalnum()]
    
    # Count term frequency in this document
    counts = Counter(tokens)
    
    # Add each term and its frequency to the inverted index
    for term, tf in counts.items():
        inv[term][str(doc_id)] = tf  # doc_id as string

# ------------------------------
# Save inverted index to JSON
# ------------------------------
save_json(inv, "outputs/inverted_index.json")

print("Saved outputs/inverted_index.json")
