# TF-IDF Information Retrieval System

A comprehensive information retrieval system implementing TF-IDF vectorization, document ranking, Rocchio relevance feedback, and ablation studies for evaluating different preprocessing configurations.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Usage](#usage)
- [Output Files](#output-files)
- [Queries](#queries)
- [Evaluation Metrics](#evaluation-metrics)

## 🎯 Overview

This project implements a complete information retrieval pipeline using TF-IDF (Term Frequency-Inverse Document Frequency) for document ranking. It includes:

- **Inverted Index Construction**: Fast document lookup by terms
- **TF-IDF Vectorization**: Convert documents to numerical representations
- **Cosine Similarity Ranking**: Rank documents by relevance to queries
- **Rocchio Relevance Feedback**: Refine queries based on user judgments
- **Ablation Studies**: Compare different preprocessing configurations
- **Evaluation Metrics**: MAP, P@5, P@10, Recall@10

## ✨ Features

- 🔍 **Document Ranking**: Rank documents using TF-IDF and cosine similarity
- 🔄 **Rocchio Feedback**: Improve query results through relevance feedback
- 📊 **Ablation Studies**: Test impact of stopword removal and stemming
- 📈 **Visualization**: Generate comparison plots for different systems
- 📉 **Evaluation**: Comprehensive IR metrics (MAP, Precision, Recall)

## 🔧 Prerequisites

- **Python**: 3.7 or higher
- **Dataset**: `News_Category_Dataset_v3.json` (place in `data/` directory)

## 📦 Installation

### 1. Clone or Navigate to the Project Directory

```bash
cd c:\Users\Houssein\Desktop\repos\mini_projet_3
```

### 2. Install Required Dependencies

```bash
pip install numpy pandas scikit-learn scipy nltk matplotlib seaborn
```

### 3. Download NLTK Data

The scripts will automatically download required NLTK data (punkt tokenizer, stopwords) on first run.

## 📁 Project Structure

```
mini_projet_3/
├── data/
│   └── News_Category_Dataset_v3.json    # Input dataset (JSON Lines format)
├── src/
│   ├── utils.py                         # Helper functions for loading/saving data
│   ├── build_index.py                   # Build inverted index
│   ├── vectorize.py                     # Create TF-IDF vectorizer and matrix
│   ├── rank_and_eval.py                 # Baseline and Rocchio ranking
│   ├── evaluate.py                      # Compute IR metrics
│   ├── mock_judgments.py                # Generate mock relevance judgments
│   ├── ablation_pipelines.py            # Create preprocessing variations
│   ├── run_ablation.py                  # Run ablation study
│   └── plot_results.py                  # Generate visualization plots
├── outputs/
│   ├── inverted_index.json              # Term → document mapping
│   ├── tfidf_vectorizer.pkl             # Trained TF-IDF vectorizer
│   ├── X.npz                            # Document-term matrix
│   ├── results/
│   │   ├── results.csv                  # Baseline ranking results
│   │   ├── judgments_template.csv       # Template for manual judgments
│   │   └── results_rocchio.csv          # Rocchio feedback results
│   ├── ablation/                        # Ablation study outputs
│   └── plots/                           # Generated visualizations
├── CODEBASE_OVERVIEW.md                 # Detailed code documentation
└── README.md                            # This file
```

## 🚀 Usage

### Interactive Notebook (Recommended for Exploration)

For an interactive demonstration of all features, use the Jupyter notebook:

```bash
jupyter notebook IR_System_Demo.ipynb
```

The notebook includes:
- 📊 Dataset exploration and visualization
- 🔍 Interactive query interface
- 🔄 Rocchio feedback examples
- 📈 Evaluation metrics demonstrations
- 📉 Ablation study visualizations
- 🔤 Vocabulary analysis

**Note**: Install Jupyter if needed: `pip install jupyter`

### Command-Line Pipeline

### Complete Pipeline (Recommended)

Run the entire pipeline in order:

```bash
# Step 1: Build inverted index
python -m src.build_index

# Step 2: Create TF-IDF vectorizer and matrix
python -m src.vectorize

# Step 3: Generate baseline rankings and judgment template
python -m src.rank_and_eval

# Step 4: (Optional) Fill judgments_template.csv with relevance judgments (0 or 1)
# Edit: outputs/results/judgments_template.csv

# Step 5: Generate mock judgments (if you don't want to manually judge)
python -m src.mock_judgments

# Step 6: Re-run to generate Rocchio results
python -m src.rank_and_eval

# Step 7: Evaluate baseline and Rocchio systems
python -m src.evaluate

# Step 8: Run ablation studies
python -m src.ablation_pipelines
python -m src.run_ablation

# Step 9: Generate visualization plots
python -m src.plot_results
```

### Individual Components

#### 1. Build Inverted Index

```bash
python -m src.build_index
```

Creates an inverted index mapping terms to documents with term frequencies.

**Output**: `outputs/inverted_index.json`

#### 2. Create TF-IDF Vectorizer

```bash
python -m src.vectorize
```

Trains a TF-IDF vectorizer on the corpus and creates the document-term matrix.

**Outputs**:
- `outputs/tfidf_vectorizer.pkl` - Trained vectorizer
- `outputs/X.npz` - Document-term matrix

#### 3. Rank Documents

```bash
python -m src.rank_and_eval
```

Ranks documents for predefined queries using:
1. **Baseline**: Cosine similarity with TF-IDF
2. **Rocchio**: Query refinement based on relevance feedback

**Outputs**:
- `outputs/results/results.csv` - Baseline rankings
- `outputs/results/judgments_template.csv` - Template for manual judgments
- `outputs/results/results_rocchio.csv` - Rocchio rankings

#### 4. Generate Mock Judgments (Optional)

```bash
python -m src.mock_judgments
```

Generates random relevance judgments for testing purposes.

**Note**: For real evaluation, manually fill `judgments_template.csv` with 0 (not relevant) or 1 (relevant).

#### 5. Evaluate Systems

```bash
python -m src.evaluate
```

Computes IR metrics for baseline and Rocchio systems.

**Metrics**:
- **MAP** (Mean Average Precision)
- **P@5** (Precision at 5)
- **P@10** (Precision at 10)
- **Recall@10** (Recall at 10)

#### 6. Run Ablation Studies

```bash
# Create preprocessing variations
python -m src.ablation_pipelines

# Run queries and evaluate
python -m src.run_ablation
```

Tests 4 preprocessing configurations:
1. No stopwords, No stemming
2. Stopwords removed, No stemming
3. No stopwords, Porter stemming
4. Stopwords removed, Porter stemming

**Output**: `outputs/ablation/ablation_metrics.csv`

#### 7. Generate Plots

```bash
python -m src.plot_results
```

Creates visualization plots comparing:
- Baseline vs. Rocchio performance
- Ablation study results

**Output**: `outputs/plots/`

## 📄 Output Files

### Results Directory (`outputs/results/`)

- **`results.csv`**: Baseline ranking results (query, doc_id, score, rank)
- **`judgments_template.csv`**: Template for manual relevance judgments
- **`results_rocchio.csv`**: Rankings after Rocchio feedback

### Ablation Directory (`outputs/ablation/`)

- **`ablation_metrics.csv`**: Performance metrics for each preprocessing configuration
- **Vectorizers and matrices**: `tfidf_vectorizer_*.pkl`, `X_*.npz`

### Plots Directory (`outputs/plots/`)

- Comparison charts for baseline vs. Rocchio
- Ablation study visualizations

## 🔎 Queries

The system uses the following predefined queries:

1. "election results"
2. "earthquake Japan"
3. "climate summit"
4. "trade negotiations"
5. "public health emergency"
6. "space mission launch"
7. "sanctions policy"
8. "wildfire response"

To modify queries, edit the `queries` list in:
- `src/rank_and_eval.py`
- `src/run_ablation.py`

## 📊 Evaluation Metrics

### Precision at K (P@K)
Proportion of relevant documents in the top K results.

### Recall at K
Proportion of all relevant documents found in the top K results.

### Average Precision (AP)
Average of precision values at each relevant document position.

### Mean Average Precision (MAP)
Mean of AP across all queries.

## 🛠️ Troubleshooting

### Missing Dataset
Ensure `News_Category_Dataset_v3.json` is in the `data/` directory.

### NLTK Download Errors
Run manually:
```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
```

### Import Errors
Make sure you're running scripts as modules from the project root:
```bash
python -m src.script_name
```

### Rocchio Feedback
- The script will work even if `judgments_template.csv` has empty relevance values
- For best results, fill the `relevant` column with 0 (not relevant) or 1 (relevant)
- Documents returned by Rocchio that weren't in the baseline results will have empty relevance values


## 📝 Notes

- The first run of `rank_and_eval.py` creates `judgments_template.csv`
- Fill this template with relevance judgments (0 or 1) before running Rocchio
- Use `mock_judgments.py` for testing without manual judgments
- All scripts should be run from the project root directory

## 📚 Additional Documentation

For detailed information about each module, see [CODEBASE_OVERVIEW.md](CODEBASE_OVERVIEW.md).

---

**Author**: Houssein  
**Last Updated**: December 2025
