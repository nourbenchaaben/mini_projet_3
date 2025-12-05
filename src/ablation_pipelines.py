
"""
ablation_pipelines.py
Role: Implements multiple text preprocessing pipelines to test the effect of stopwords removal and stemming on retrieval.
Converts documents to TF-IDF vectors for each pipeline.
Saves both the vectorizer and the document-term matrix for later evaluation.
"""
# Import the TF-IDF vectorizer from scikit-learn
from sklearn.feature_extraction.text import TfidfVectorizer
# Import helper functions to load documents and save outputs
from src.utils import load_docs, save_pickle, save_sparse_matrix
# Import a stemming algorithm from NLTK
from nltk.stem import PorterStemmer
import os  # for creating directories

# ------------------------------
# Function to preprocess documents
# ------------------------------
def preprocess_docs(docs, do_stem=False):
    """
    Preprocess a list of documents:
    - Tokenize each document
    - Convert tokens to lowercase
    - Remove non-alphanumeric tokens
    - Apply stemming if do_stem=True
    Returns a list of preprocessed documents (as strings)
    """
    import nltk
    nltk.download("punkt", quiet=True)  # download tokenizer if not present
    nltk.download("punkt_tab", quiet=True)
    ps = PorterStemmer()  # initialize stemmer
    new = []  # store processed documents
    for d in docs:
        # tokenize and lowercase
        tokens = [t.lower() for t in nltk.word_tokenize(d) if t.isalnum()]
        if do_stem:
            # apply stemming
            tokens = [ps.stem(t) for t in tokens]
        # join tokens back into a string
        new.append(" ".join(tokens))
    return new

# ------------------------------
# Load all documents
# ------------------------------
docs, _ = load_docs()  # load document texts and titles (titles ignored here)

# ------------------------------
# Define ablation pipelines
# ------------------------------
# Each pipeline varies stopword removal and stemming settings
pipelines = [
    {"name":"A_stop_on_stem_off", "stop":"english", "stem":False},  # stopwords ON, no stemming
    {"name":"B_stop_off_stem_off", "stop":None, "stem":False},      # stopwords OFF, no stemming
    {"name":"C_stop_on_stem_on", "stop":"english", "stem":True},    # stopwords ON, stemming ON
    {"name":"D_stop_off_stem_on", "stop":None, "stem":True},        # stopwords OFF, stemming ON
]

# ------------------------------
# Create output directory if missing
# ------------------------------
os.makedirs("outputs/ablation", exist_ok=True)

# ------------------------------
# Run each pipeline
# ------------------------------
for p in pipelines:
    # Preprocess the documents according to the pipeline settings
    proc_docs = preprocess_docs(docs, do_stem=p["stem"])
    
    # Create TF-IDF vectorizer with the pipeline's stopword setting
    # min_df=3 removes terms that appear in fewer than 3 documents
    vec = TfidfVectorizer(stop_words=p["stop"], min_df=3)
    
    # Fit the vectorizer on the preprocessed documents and transform them into a document-term matrix
    X = vec.fit_transform(proc_docs)
    
    # Save the vectorizer object to a pickle file for later use
    save_pickle(vec, f"outputs/ablation/{p['name']}_vectorizer.pkl")
    
    # Save the sparse document-term matrix in .npz format
    save_sparse_matrix(X, f"outputs/ablation/{p['name']}_X.npz")
    
    # Print confirmation and vocabulary size
    print("Saved pipeline", p["name"], "vocab:", len(vec.vocabulary_))
