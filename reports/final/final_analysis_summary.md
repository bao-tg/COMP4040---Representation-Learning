# Final Analysis Summary

Validation status: OK

| Encoder | Rows | Dim | File Size | Device | Embed Time | NMI | ARI | Silhouette | MRR | P@5 | P@10 | P@50 | Iso Anom. | Inconsist. | Acc. | Macro F1 | Weighted F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| TFIDF | 1946618 | 300 | 2.18 GB | cpu | 5:08.61 | 0.226 | 0.107 | 0.001 | 0.691 | 0.527 | 0.515 | 0.482 | 19467 | 19466 | 0.447 | 0.435 | 0.435 |
| W2V | 1946618 | 300 | 2.18 GB | cpu | 2:30.93 | 0.014 | 0.010 | 0.006 | 0.722 | 0.580 | 0.569 | 0.546 | 19550 | 19466 | 0.432 | 0.420 | 0.419 |
| GLOVE | 1946618 | 300 | 2.18 GB | cpu | 1:56.52 | 0.030 | 0.023 | 0.036 | 0.742 | 0.597 | 0.590 | 0.568 | 19476 | 19466 | 0.433 | 0.408 | 0.408 |
| SBERT | 1946618 | 384 | 2.78 GB | cuda | 3:01:27 | 0.410 | 0.348 | 0.016 | 0.819 | 0.718 | 0.712 | 0.697 | 19467 | 19466 | 0.456 | 0.443 | 0.443 |
| BGE | 1946618 | 1024 | 7.43 GB | cuda | 65:56:44 | 0.342 | 0.294 | 0.032 | 0.828 | 0.730 | 0.725 | 0.713 | 19467 | 19466 | 0.590 | 0.584 | 0.584 |

Notes:
- Paths, raw samples, log files, generated arrays, and anomaly row CSVs are intentionally excluded from this commit-safe package.
- Full precision values are kept in `final_analysis_summary.csv` and `final_analysis_summary.json`.
