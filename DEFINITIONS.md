# Key Definitions and Concepts

## Information Retrieval (IR) System
A system designed to find and rank documents relevant to a user's query from a large collection. Our system uses TF-IDF vectorization and cosine similarity to retrieve news articles.

---

## Core Concepts

### TF-IDF (Term Frequency-Inverse Document Frequency)
A numerical statistic that reflects how important a word is to a document in a collection.

- **TF (Term Frequency)**: How often a term appears in a document
  - Formula: `TF(t,d) = count(t in d) / total terms in d`
  - Higher TF = term is more important to that document

- **IDF (Inverse Document Frequency)**: How rare a term is across all documents
  - Formula: `IDF(t) = log(total documents / documents containing t)`
  - Higher IDF = term is more discriminative/unique

- **TF-IDF Score**: `TF × IDF`
  - Balances frequency with uniqueness
  - Common words (like "the") get low scores
  - Rare, meaningful words get high scores

### Cosine Similarity
A measure of similarity between two vectors (documents or queries).

- **Formula**: `cos(θ) = (A · B) / (||A|| × ||B||)`
- **Range**: 0 (completely different) to 1 (identical)
- **Usage**: Compare query vector with document vectors to find matches
- **Example**: Score of 0.78 means 78% similarity

### Inverted Index
A data structure mapping terms to the documents containing them.

- **Structure**: `{term: {doc_id: frequency}}`
- **Purpose**: Fast document lookup by keywords
- **Example**: `{"climate": {1: 3, 5: 2, 12: 1}}`

---

## Preprocessing Techniques

### Stopword Removal
Removing common words that don't carry much meaning.

- **Examples**: "the", "is", "and", "a"
- **Purpose**: Reduce noise and focus on content words
- **Trade-off**: May lose some context

### Stemming (Porter Stemmer)
Reducing words to their root form.

- **Examples**: 
  - "running", "runs", "ran" → "run"
  - "connection", "connected" → "connect"
- **Purpose**: Match different forms of the same word
- **Benefit**: Improves recall by normalizing variations

---

## Evaluation Metrics

### Precision@K (P@K)
Percentage of relevant documents in the top K results.

- **Formula**: `P@K = (relevant docs in top K) / K`
- **Example**: P@5 = 0.40 means 2 out of 5 top results are relevant
- **Measures**: Accuracy of top results

### Recall@K
Percentage of all relevant documents found in top K results.

- **Formula**: `Recall@K = (relevant docs in top K) / (total relevant docs)`
- **Example**: Recall@10 = 1.0 means all relevant docs were found in top 10
- **Measures**: Completeness of retrieval

### Average Precision (AP)
Average of precision values at each relevant document position.

- **Formula**: `AP = Σ(P@k × rel(k)) / total relevant docs`
- **Range**: 0 to 1 (higher is better)
- **Measures**: Quality of ranking

### Mean Average Precision (MAP)
Average of AP scores across all queries.

- **Formula**: `MAP = Σ(AP for each query) / number of queries`
- **Usage**: Overall system performance metric
- **Our best**: 0.410 (Pipeline D)

---

## Advanced Techniques

### Rocchio Relevance Feedback
A query refinement technique that adjusts the query based on user feedback.

- **Formula**: `Q_new = α×Q_original + β×avg(relevant docs) - γ×avg(non-relevant docs)`
- **Parameters**:
  - α = 1.0 (original query weight)
  - β = 0.75 (boost from relevant documents)
  - γ = 0.15 (penalty from non-relevant documents)
- **Effect**: Moves query vector toward relevant documents
- **Result**: Improved retrieval scores (often 50%+ increase)

### Ablation Study
Systematic testing of different system configurations to identify optimal settings.

- **Purpose**: Understand impact of each component
- **Our study**: Tested 4 preprocessing combinations
  - Stopwords: ON/OFF
  - Stemming: ON/OFF
- **Findings**: Stemming without stopword removal performed best

---

## Document-Term Matrix

### Sparse Matrix
A matrix where most values are zero.

- **Our matrix**: 209,527 × 10,000 (documents × terms)
- **Sparsity**: 99.92% zeros
- **Storage**: Only non-zero values stored (efficient)
- **Reason**: Most documents don't contain most terms

### Vocabulary Size
Total number of unique terms in the system.

- **Our size**: 10,000 terms (limited by `max_features`)
- **Selection**: Most frequent/important terms
- **Trade-off**: Smaller = faster, larger = more precise

---

## System Components

### Vectorizer
Converts text documents into numerical vectors.

- **Input**: Raw text strings
- **Output**: TF-IDF weighted vectors
- **Configuration**:
  - `max_features=10000`: Limit vocabulary size
  - `stop_words='english'`: Remove common English words
  - `lowercase=True`: Normalize case
  - `ngram_range=(1,1)`: Use single words only

### Query Processing
Steps to handle a user query:

1. **Tokenization**: Split query into words
2. **Vectorization**: Convert to TF-IDF vector using trained vectorizer
3. **Similarity Calculation**: Compare with all document vectors
4. **Ranking**: Sort by similarity score (descending)
5. **Return**: Top K results

---

## Performance Indicators

### Matrix Sparsity
Percentage of zero values in the document-term matrix.

- **Formula**: `(1 - non_zeros / total_elements) × 100`
- **Our value**: 99.92%
- **Meaning**: Very efficient storage

### IDF Score Interpretation
- **High IDF (>10)**: Very rare, discriminative terms
- **Medium IDF (5-10)**: Moderately common terms
- **Low IDF (<5)**: Common terms with less discriminative power

### Similarity Score Interpretation
- **0.7-1.0**: Excellent match
- **0.5-0.7**: Good match
- **0.3-0.5**: Fair match
- **<0.3**: Poor match

---

## Pipeline Configurations (Ablation Study)

### Pipeline A: Stopwords ON, Stemming OFF
- Removes common words, keeps original word forms
- Best for: Precision at top results

### Pipeline B: Stopwords OFF, Stemming OFF
- Keeps all words in original form
- Best for: Recall (finding all relevant documents)

### Pipeline C: Stopwords ON, Stemming ON
- Removes common words, normalizes word forms
- Balanced approach

### Pipeline D: Stopwords OFF, Stemming ON ⭐
- Keeps all words, normalizes forms
- **Best overall performance** (MAP = 0.410)
- Recommended configuration

---

## Dataset Information

### News Category Dataset v3
- **Size**: 209,527 news articles
- **Format**: JSON Lines (one article per line)
- **Fields**: headline, short_description, category
- **Average length**: 19.7 words per article
- **Source**: News aggregation platform

---

## Key Formulas Summary

| Concept | Formula |
|---------|---------|
| TF-IDF | `TF(t,d) × IDF(t)` |
| Cosine Similarity | `(A·B) / (‖A‖×‖B‖)` |
| Precision@K | `relevant_in_top_K / K` |
| Recall@K | `relevant_in_top_K / total_relevant` |
| Rocchio | `α×Q + β×avg(D+) - γ×avg(D-)` |

---

## Presentation Tips

### When explaining TF-IDF:
"TF-IDF helps us identify important words. If a word appears often in one document but rarely in others, it's probably important for that specific document."

### When explaining Cosine Similarity:
"Think of documents as points in space. Cosine similarity measures the angle between them - smaller angle means more similar documents."

### When explaining Rocchio:
"It's like asking 'give me more like this, but less like that' - the system learns from your feedback to improve results."

### When explaining the Ablation Study:
"We tested different preprocessing approaches to find the best configuration. Keeping all words but normalizing their forms (stemming) gave us the best results."

---

## Success Metrics

Our system achieved:
- ✅ **50% Precision@10**: Half of top 10 results are relevant
- ✅ **100% Recall@10**: Found all relevant documents in top 10
- ✅ **0.71 Average Precision**: Strong ranking quality
- ✅ **0.41 MAP**: Competitive overall performance
- ✅ **50%+ score improvement** with Rocchio feedback

---

*This glossary covers all key concepts used in the TF-IDF Information Retrieval System project.*
