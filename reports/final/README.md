# Final Report Artifacts

This directory is the commit-safe result package for the full `server_full` run.
It contains the aggregate metrics and report-ready figures needed for the project report, while excluding raw/generated files that are too large or may expose review text or local machine paths.

Included:
- `final_analysis_summary.md`, `.csv`, `.json`: final encoder comparison table with runtime, artifact size, clustering, retrieval, anomaly, UMAP, and linear probe metrics.
- `metrics/`: sanitized task-level metrics for clustering, retrieval, anomaly detection, linear probe, UMAP, efficiency, and input validation.
- `metrics/confusion_matrices/`: aggregate linear probe confusion matrices.
- `figures/`: UMAP overview grids and per-encoder anomaly score/residual plots.

Excluded on purpose:
- large embeddings under `data/embeddings/`
- raw processed data and pretrained vector files
- runtime logs containing local/server paths
- anomaly row CSVs containing review text fields
- UMAP coordinate arrays and clustering assignment arrays

Use this folder for Git/report evidence. Keep `data/`, `experiments/`, and `logs/` ignored unless a specific reviewer asks for a separate artifact archive.
