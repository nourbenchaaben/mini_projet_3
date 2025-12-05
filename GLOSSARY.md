# Project Glossary - Technical Terms & Vocabulary

A comprehensive guide to all technical terms, acronyms, and concepts used in this Information Retrieval project.

---

## A

### **Ablation Study**
A systematic experiment that removes or changes components to understand their individual impact on system performance.

### **Alpha (α)**
Weight parameter in Rocchio algorithm for the original query (typically 1.0).

### **Average Precision (AP)**
Metric measuring the quality of ranked results by averaging precision at each relevant document position.

---

## B

### **Baseline**
The initial system performance before applying improvements (like Rocchio feedback).

### **Beta (β)**
Weight parameter in Rocchio algorithm for relevant documents (typically 0.75).

### **Binary Relevance**
Documents marked as either relevant (1) or non-relevant (0) to a query.

### **Bag of Words**
Text representation that ignores word order and only counts word occurrences.

---

## C

### **Corpus**
The complete collection of documents being searched (our dataset has 209,527 news articles).

### **Cosine Similarity**
Measure of similarity between two vectors based on the angle between them (range: 0-1).

### **CSV (Comma-Separated Values)**
File format for storing tabular data (used for results and metrics).

---

## D

### **Document**
A single text item in the collection (in our case, a news article).

### **Document-Term Matrix**
A matrix where rows represent documents and columns represent terms, with TF-IDF scores as values.

### **Discriminative Term**
A word that helps distinguish one document from others (has high IDF).

---

## E

### **Evaluation Metrics**
Quantitative measures to assess system performance (P@K, Recall, MAP, etc.).

---

## F

### **Feature**
A term/word in the vocabulary used for document representation.

### **Feature Vector**
Numerical representation of a document or query as a list of TF-IDF scores.

---

## G

### **Gamma (γ)**
Weight parameter in Rocchio algorithm for non-relevant documents (typically 0.15).

### **Ground Truth**
Manually verified correct answers (relevance judgments) used for evaluation.

---

## H

### **Hyperparameter**
Configuration setting that affects system behavior (e.g., max_features, alpha, beta).

---

## I

### **IDF (Inverse Document Frequency)**
Measure of how rare a term is across all documents; formula: `log(N / df)` where N = total docs, df = docs containing term.

### **Indexing**
Process of organizing documents for efficient retrieval.

### **Information Retrieval (IR)**
The science of searching for and retrieving relevant information from large collections.

### **Inverted Index**
Data structure mapping each term to the list of documents containing it.

---

## J

### **JSON (JavaScript Object Notation)**
Text format for storing structured data (our dataset format).

### **JSON Lines**
File format where each line is a separate JSON object.

### **Judgment**
Human assessment of whether a document is relevant to a query.

---

## K

### **K (in P@K, Recall@K)**
The number of top results considered (e.g., K=5 means top 5 results).

---

## L

### **Lemmatization**
Reducing words to their dictionary form (more sophisticated than stemming).

### **Lowercase Normalization**
Converting all text to lowercase for consistent matching.

---

## M

### **MAP (Mean Average Precision)**
Average of AP scores across all queries; overall system quality metric.

### **Matrix Sparsity**
Percentage of zero values in a matrix (our matrix is 99.92% sparse).

### **Max Features**
Maximum number of terms to include in vocabulary (we use 10,000).

---

## N

### **N-gram**
Sequence of N consecutive words (1-gram = single word, 2-gram = two words).

### **NLTK (Natural Language Toolkit)**
Python library for natural language processing tasks.

### **Normalization**
Process of standardizing text (lowercase, removing punctuation, etc.).

---

## O

### **Offline Evaluation**
Testing system performance using pre-collected relevance judgments.

---

## P

### **Pipeline**
Sequence of processing steps (tokenization → stopword removal → stemming → vectorization).

### **Porter Stemmer**
Algorithm for reducing words to their root form (e.g., "running" → "run").

### **Precision**
Fraction of retrieved documents that are relevant; formula: `relevant_retrieved / total_retrieved`.

### **Precision@K (P@K)**
Precision calculated only for the top K results.

### **Preprocessing**
Preparing raw text for analysis (tokenization, stemming, stopword removal).

---

## Q

### **Query**
User's search request (e.g., "climate change").

### **Query Expansion**
Adding related terms to a query to improve retrieval (Rocchio does this).

### **Query Vector**
TF-IDF representation of a search query.

---

## R

### **Ranking**
Ordering documents by relevance score (highest first).

### **Recall**
Fraction of relevant documents that were retrieved; formula: `relevant_retrieved / total_relevant`.

### **Recall@K**
Recall calculated only for the top K results.

### **Relevance Feedback**
Using user judgments to improve query and results (Rocchio algorithm).

### **Relevant Document**
A document that satisfies the user's information need.

### **Retrieval**
Process of finding documents matching a query.

### **Rocchio Algorithm**
Query refinement technique that adjusts query based on relevant/non-relevant documents.

---

## S

### **Scikit-learn (sklearn)**
Python library for machine learning and text processing.

### **Score**
Numerical value indicating document-query similarity (0-1 for cosine similarity).

### **SciPy**
Python library for scientific computing (we use it for sparse matrices).

### **Sparse Matrix**
Matrix with mostly zero values, stored efficiently.

### **Stemming**
Reducing words to their root form by removing suffixes.

### **Stopwords**
Common words with little meaning (e.g., "the", "is", "and") often removed during preprocessing.

---

## T

### **Term**
A word or token in the vocabulary.

### **Term Frequency (TF)**
Number of times a term appears in a document.

### **TF-IDF (Term Frequency-Inverse Document Frequency)**
Weighting scheme that balances term frequency with term rarity.

### **TfidfVectorizer**
Scikit-learn tool that converts text to TF-IDF vectors.

### **Token**
Individual word or symbol after text splitting.

### **Tokenization**
Splitting text into individual words/tokens.

### **Top-K Results**
The K highest-ranked documents (e.g., top 10 results).

---

## U

### **Unigram**
Single word (1-gram).

---

## V

### **Vector**
Numerical array representing a document or query.

### **Vectorization**
Converting text into numerical vectors.

### **Vocabulary**
Set of all unique terms used in the system.

---

## W

### **Weight**
Numerical value indicating term importance (TF-IDF score).

---

## Common Acronyms

| Acronym | Full Form | Meaning |
|---------|-----------|---------|
| **AP** | Average Precision | Quality of ranking for one query |
| **CSV** | Comma-Separated Values | Tabular data format |
| **IDF** | Inverse Document Frequency | Rarity measure |
| **IR** | Information Retrieval | Document search field |
| **JSON** | JavaScript Object Notation | Data format |
| **MAP** | Mean Average Precision | Overall system quality |
| **NLTK** | Natural Language Toolkit | NLP library |
| **NPZ** | NumPy Zipped | Compressed array format |
| **P@K** | Precision at K | Precision in top K |
| **PKL** | Pickle | Python serialization |
| **TF** | Term Frequency | Word count measure |
| **TF-IDF** | Term Frequency-Inverse Document Frequency | Weighting scheme |

---

## File Extensions Used

| Extension | Description | Example |
|-----------|-------------|---------|
| `.py` | Python source code | `utils.py` |
| `.json` | JSON data file | `inverted_index.json` |
| `.csv` | Comma-separated values | `results.csv` |
| `.pkl` | Pickled Python object | `tfidf_vectorizer.pkl` |
| `.npz` | Compressed NumPy array | `X.npz` |
| `.md` | Markdown documentation | `README.md` |
| `.ipynb` | Jupyter Notebook | `IR_System_Demo.ipynb` |
| `.png` | Image file | `ablation_map.png` |

---

## Mathematical Notation

| Symbol | Meaning |
|--------|---------|
| **α** | Alpha - original query weight |
| **β** | Beta - relevant documents weight |
| **γ** | Gamma - non-relevant documents weight |
| **K** | Number of top results |
| **N** | Total number of documents |
| **d** | A document |
| **q** | A query |
| **t** | A term |
| **‖v‖** | Magnitude/length of vector v |
| **v·w** | Dot product of vectors v and w |
| **Σ** | Summation |
| **log** | Logarithm (base 10 or natural) |

---

## Common Variable Names in Code

| Variable | Meaning |
|----------|---------|
| `docs` | List of document texts |
| `titles` | List of document titles |
| `X` | Document-term matrix |
| `vec` | TF-IDF vectorizer object |
| `query` | Search query string |
| `qvec` | Query vector |
| `sims` | Similarity scores |
| `top_idx` | Indices of top results |
| `df` | DataFrame (pandas) |
| `inv` | Inverted index |
| `k` | Number of results |

---

## Pipeline Component Names

| Component | Description |
|-----------|-------------|
| **A_stop_on_stem_off** | Stopwords removed, no stemming |
| **B_stop_off_stem_off** | No stopwords removed, no stemming |
| **C_stop_on_stem_on** | Stopwords removed, with stemming |
| **D_stop_off_stem_on** | No stopwords removed, with stemming |

---

## Output Directory Structure

```
outputs/
├── inverted_index.json      # Term → document mapping
├── tfidf_vectorizer.pkl      # Trained vectorizer
├── X.npz                     # Document-term matrix
├── results/
│   ├── results.csv           # Baseline rankings
│   ├── judgments_template.csv # Relevance judgments
│   └── results_rocchio.csv   # Rocchio rankings
├── ablation/
│   ├── ablation_metrics.csv  # Performance comparison
│   └── tfidf_vectorizer_*.pkl # Pipeline-specific vectorizers
└── plots/
    ├── baseline_vs_rocchio.png
    └── ablation_map.png
```

---

## Quick Reference: Key Formulas

```
TF-IDF(t,d) = TF(t,d) × IDF(t)

IDF(t) = log(N / df_t)

Cosine Similarity = (A·B) / (‖A‖ × ‖B‖)

Precision@K = relevant_in_top_K / K

Recall@K = relevant_in_top_K / total_relevant

Rocchio = α×Q_original + β×avg(D+) - γ×avg(D-)
```

---

## Common Confusions Clarified

### **TF vs TF-IDF**
- **TF**: Just counts word frequency
- **TF-IDF**: Balances frequency with rarity (better for IR)

### **Precision vs Recall**
- **Precision**: "Of what I found, how much is relevant?"
- **Recall**: "Of what's relevant, how much did I find?"

### **Stemming vs Lemmatization**
- **Stemming**: Crude chopping (running → run)
- **Lemmatization**: Dictionary-based (better → good)

### **Baseline vs Rocchio**
- **Baseline**: Original query results
- **Rocchio**: Improved results after user feedback

### **Sparse vs Dense Matrix**
- **Sparse**: Mostly zeros (99.92% in our case)
- **Dense**: Most values are non-zero

---

*This glossary covers all technical terminology used in the TF-IDF Information Retrieval System project.*
