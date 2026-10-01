#!/usr/bin/env python3
"""Run the offline RAG evaluation and write JSON + Markdown reports.

Builds the hybrid index from ``data/corpus``, runs the pipeline over the
gold Q&A set in extractive (no-API-key) mode, writes reports to ``reports/``,
and exits nonzero if any metric falls below its ``eval_config.json`` threshold.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rag_eval.eval import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
