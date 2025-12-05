# Codebase Overview

This document explains the role of each file in the `src/` directory.

## Core Components

### Data Loading & Utilities
- **[utils.py](src/utils.py)**: Contains helper functions to load documents from the JSON dataset (`load_docs`) and save/load data in various formats (JSON, Pickle, NPZ).

### Indexing & Vectorization
- **[build_index.py](src/build_index.py)**: Builds an inverted index from the document corpus and saves it as a JSON file. This is the first step in the retrieval pipeline.
- **[vectorize.py](src/vectorize.py)**: Converts the text corpus into a TF-IDF matrix. It saves the trained vectorizer and the document-term matrix for later use.

### Ranking & Evaluation
- **[rank_and_eval.py](src/rank_and_eval.py)**: The main script for ranking documents. It performs:
    1.  Baseline ranking using cosine similarity.
    2.  Rocchio relevance feedback to refine queries.
    3.  Generation of results files (`results.csv`, `results_rocchio.csv`).
- **[evaluate.py](src/evaluate.py)**: Computes Information Retrieval metrics (MAP, P@5, P@10, Recall@10) by comparing system rankings against relevance judgments.
- **[mock_judgments.py](src/mock_judgments.py)**: Generates random relevance judgments (0 or 1) for testing purposes when real manual judgments are not available.

## Ablation Studies
- **[ablation_pipelines.py](src/ablation_pipelines.py)**: Implements multiple preprocessing pipelines (varying stopword removal and stemming) to test their impact on retrieval performance. It saves vectorizers and matrices for each configuration.
- **[run_ablation.py](src/run_ablation.py)**: Runs queries through all ablation pipelines, evaluates them, and saves the metrics to `ablation_metrics.csv`.

## Visualization
- **[plot_results.py](src/plot_results.py)**: Generates plots to visualize the performance of different systems (Baseline vs. Rocchio) and ablation studies.
