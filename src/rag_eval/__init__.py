"""rag-eval-harness: hybrid RAG pipeline with a built-in evaluation harness."""

from pathlib import Path

__version__ = "0.1.0"

#: Absolute path of the project root (the directory containing ``data/``,
#: ``eval_config.json`` and ``reports/``). Used to resolve data paths
#: regardless of the current working directory.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

__all__ = ["__version__", "PROJECT_ROOT"]
