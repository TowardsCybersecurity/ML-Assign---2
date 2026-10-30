#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PYTHON="${PYTHON:-.venv/bin/python}"
"$PYTHON" -m src.data
"$PYTHON" -u -m src.experiment select
"$PYTHON" -u -m src.experiment evaluate
"$PYTHON" scripts/case_attribution.py
"$PYTHON" -u -m src.experiment resample
"$PYTHON" -m src.figures
