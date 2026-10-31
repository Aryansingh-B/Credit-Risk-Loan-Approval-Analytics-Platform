"""
run_pipeline.py
---------------
ONE command from raw CSV to scored output (PRD FR-12 / reproducibility NFR):

    python src/run_pipeline.py

Steps: load + validate DB -> run the 10 SQL queries -> tune/train/calibrate
-> score every customer and loan -> regenerate the model card.
Fixed seeds throughout, so re-running reproduces the reported metrics.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

SRC = Path(__file__).resolve().parent
STEPS = [
    ("1/5  Load CSVs into the governed database", "data_load.py"),
    ("2/5  Run the 10 analytical SQL queries", "sql_analytics.py"),
    ("3/5  Tune, compare, calibrate and persist the model", "train.py"),
    ("4/5  Score every customer and disbursed loan", "risk_score.py"),
    ("5/5  Regenerate the model card from the real metrics", "report.py"),
]


def main() -> None:
    start = time.time()
    for label, script in STEPS:
        print(f"\n{'=' * 78}\n{label}\n{'=' * 78}", flush=True)
        t = time.time()
        subprocess.run([sys.executable, str(SRC / script)], check=True, cwd=SRC.parent)
        print(f"-- done in {time.time() - t:.1f}s", flush=True)
    print(f"\nPipeline complete in {time.time() - start:.0f}s. "
          f"Launch the dashboard with:  streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()
