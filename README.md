# Olivetti Faces: bagging and SAMME

Vaibhav Krishna · P26CS0012 · split seed 260012

Repository: https://github.com/TowardsCybersecurity/ML-Assign---2

## Setup

Use Python 3.13.

```sh
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

Run all commands from this folder. The dataset downloads through scikit-learn on first use. It is not included in the repository; see `DATA_NOTICE.md`.

## Reproduction

```sh
bash scripts/reproduce.sh
```

This runs the following stages and overwrites experiment outputs. Preserve existing results before rerunning if you want to compare them.

```sh
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
.venv/bin/python -m src.data
.venv/bin/python -u -m src.experiment select
.venv/bin/python -u -m src.experiment evaluate
.venv/bin/python scripts/case_attribution.py
.venv/bin/python -u -m src.experiment resample
.venv/bin/python -m src.figures
```

Selection uses training and validation data only. All stages load `artifacts/split.json`. Models train on clean raw pixels. The 30 refits use identical within-identity resamples across models, with fixed model randomness. Validation and test images never enter training.

## Files

- `src/`: data loading, scratch ensembles, experiments, and figure generation.
- `configs/`: seeds, parameter grids, and repository URL.
- `artifacts/`: saved split, configurations, final logs, predictions, training traces, and all 30 refit outputs.
- `figures/`: generated plots (PDF) and numerical tables (CSV).
- `scripts/`: reproduction and masking case analysis.
- `requirements.txt`: pinned Python environment.

The report is maintained separately in Overleaf and is not included here. Figure PDFs and CSV tables can be regenerated with `.venv/bin/python -m src.figures` without retraining.

## Limitations and attribution

Validation selected unrestricted tree depth, so the depth-64 comparison cannot provide a strictly deeper setting for Q3(b). The report discloses that these two settings are equivalent.

