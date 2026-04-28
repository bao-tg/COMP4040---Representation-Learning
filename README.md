# COMP4040---Representation-Learning

Typically, the README file is first-read file in the whole coding project.

# Setup and usage

## 1. Install

```bash
pip install -r requirements.txt
```

## 2. Prepare data

```bash
python scripts/download_amazon_reviews.py --category All_Beauty --size 10000 --output_dir data/raw --seed 42
```


# Folder structure

The folder structure below is recommended by Claude:
+ Typically, there must be one data folder for preprocessing step
+ One notebooks folder for experimenting, debugging, plotting
+ src (Source code of every important function)
+ experiemnts (Used for saving experiments' outpue)

```bash
representation-learning/
│
├── data/
│   ├── raw/              ← original downloaded Parquet files from HuggingFace
│   ├── processed/        ← cleaned, deduplicated, truncated dataset (1.5M rows)
│   ├── embeddings/       ← one .npy file per encoder (tfidf.npy, w2v.npy, sbert.npy, bge.npy)
│   └── indexes/          ← FAISS index files (.index) per encoder
│
├── notebooks/
│   ├── 01_eda.ipynb           ← rating distribution, review length, category counts
│   ├── 02_preprocess.ipynb    ← cleaning pipeline, saves to data/processed/
│   ├── 03_encode.ipynb        ← runs all 5 encoders, saves to data/embeddings/
│   ├── 04_clustering.ipynb    ← Experiment 1: k-means, NMI, ARI, UMAP plots
│   ├── 05_retrieval.ipynb     ← Experiment 2: FAISS indexing, Precision@k, MRR
│   └── 06_anomaly.ipynb       ← Experiment 3: IsolationForest + rating inconsistency
│
├── src/
│   ├── encoders.py     ← unified Encoder interface (TF-IDF, W2V, GloVe, SBERT, BGE)
│   ├── preprocess.py   ← cleaning functions (strip HTML, dedup, truncate)
│   ├── cluster.py      ← MiniBatchKMeans wrapper + evaluation (NMI, ARI, Silhouette)
│   ├── retrieval.py    ← FAISS index builder + search + Precision@k / MRR
│   ├── anomaly.py      ← IsolationForest + rating-text inconsistency scorer
│   ├── metrics.py      ← all evaluation metrics in one place
│   ├── visualize.py    ← UMAP 2D plots, metric bar charts
│   ├── utils.py        ← logging, seed setting, Parquet I/O helpers
│   ├── config.py       ← all hyperparams in one place (k, batch_size, model names)
│   └── demo.py         ← live retrieval demo script for the presentation
│
├── experiments/
│   ├── clustering/     ← saved metrics JSON per encoder (nmi, ari, silhouette)
│   ├── retrieval/      ← Precision@k and MRR tables per encoder
│   ├── anomaly/        ← flagged review CSVs, anomaly score distributions
│   └── figures/        ← UMAP PNGs, metric comparison charts
│
├── reports/
│   ├── report.pdf      ← final written report
│   ├── slides.pptx     ← presentation deck
│   └── figures/        ← high-res versions of all paper figures
│
├── README.md           ← setup instructions, how to reproduce all experiments
├── requirements.txt    ← pinned dependencies
├── .env.example        ← HuggingFace token if needed
└── Makefile            ← shortcuts: make encode, make cluster, make demo
```

# It's time to learn git

Some keywords, ask LLMs to learn (Learn it quick, and practice them in your own repositories):
+ Remote vs Local in Git. How to push code into Remote?
+ Why we use git add ., and git commit?
+ What is a Conflict? what does resolving a conflict mean?
+ What is a Pull Request? How to open a Pull Request?

# Our gitworkflow

Typically, we don't want to directly push "dirty code" into the `main` branch, we implement in our own branch, and open the Pull Request (or PR), and the reviewer (Bao will be the reviewer) will check, if it's validate, your code will be merged into the main.

We will have some conventions for managing the source code:
+ Each PR should have 1 commit only (Learn `git reset`, and `git log` for this)
+ Before openning the PR, you should test all of your code, and resolve all conflict (learn `git fetch` and `git rebase` for this!!!)
