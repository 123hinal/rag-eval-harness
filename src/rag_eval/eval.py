"""Offline evaluation harness for the RAG pipeline.

Metrics (all in ``[0, 1]``, higher is better):

* ``context_precision_at_k`` — fraction of the top-k retrieved documents
  that are in the gold relevant set.
* ``context_recall`` — fraction of the gold relevant documents retrieved.
* ``faithfulness`` — fraction of answer sentences whose content words are
  substantially (≥50%) supported by at least one retrieved chunk. A
  transparent lexical heuristic: no model calls, fully offline.
* ``answer_relevancy`` — fraction of the question's content words that
  appear in the answer.

The gold set's ``reference_answer`` fields are not scored by these
retrieval-grounded metrics; they are included for qualitative review and
for future LLM-judge extensions.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path

from . import PROJECT_ROOT
from .ingest import chunk_documents, load_corpus
from .pipeline import ExtractiveGenerator, RAGPipeline
from .retrieval import HybridRetriever, get_embedding_backend

#: Common English words excluded from content-word overlap computations.
STOPWORDS = frozenset(
    """
    a an the and or but if then else when while of at by for with about into
    through during before after above below to from up down in out on off over
    under again further once here there all any both each few more most other
    some such no nor not only own same so than too very can will just don
    should now is are was were be been being have has had having do does did
    doing would could ought i you he she it we they them his her its our their
    this that these those am me my your him us what which who whom whose where
    why how as
    """.split()
)


def content_words(text: str) -> set[str]:
    """Lowercase alphanumeric tokens minus stopwords (length > 1)."""
    return {
        w
        for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in STOPWORDS and len(w) > 1
    }


def split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on ``.!?`` boundaries."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def context_precision_at_k(
    retrieved_doc_ids: list[str], relevant_doc_ids: set[str], k: int
) -> float:
    """Fraction of the top-k retrieved docs that are relevant."""
    if k <= 0:
        return 0.0
    top_k = retrieved_doc_ids[:k]
    if not top_k:
        return 0.0
    return sum(1 for doc_id in top_k if doc_id in relevant_doc_ids) / len(top_k)


def context_recall(
    retrieved_doc_ids: list[str], relevant_doc_ids: set[str]
) -> float:
    """Fraction of relevant docs that were retrieved (at any rank)."""
    if not relevant_doc_ids:
        return 1.0
    return len(set(retrieved_doc_ids) & relevant_doc_ids) / len(relevant_doc_ids)


def faithfulness(answer: str, context_texts: list[str]) -> float:
    """Fraction of answer sentences supported by the retrieved context.

    A sentence counts as supported when at least half of its content words
    occur in a single retrieved chunk.
    """
    claims = [s for s in split_sentences(answer) if content_words(s)]
    if not claims:
        return 0.0
    chunk_word_sets = [content_words(text) for text in context_texts]
    supported = 0
    for claim in claims:
        words = content_words(claim)
        if any(len(words & chunk_words) / len(words) >= 0.5 for chunk_words in chunk_word_sets):
            supported += 1
    return supported / len(claims)


def answer_relevancy(question: str, answer: str) -> float:
    """Fraction of the question's content words covered by the answer."""
    question_words = content_words(question)
    if not question_words:
        return 0.0
    return len(question_words & content_words(answer)) / len(question_words)


# ------------------------------------------------------------------ data types


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    reference_answer: str
    relevant_doc_ids: list[str]


@dataclass
class QuestionResult:
    id: str
    question: str
    context_precision_at_k: float
    context_recall: float
    faithfulness: float
    answer_relevancy: float
    retrieved_doc_ids: list[str]


@dataclass
class EvalReport:
    top_k: int
    num_questions: int
    averages: dict[str, float]
    thresholds: dict[str, float]
    passed: bool
    failures: list[str] = field(default_factory=list)
    results: list[QuestionResult] = field(default_factory=list)


def load_gold(path: str | Path) -> list[EvalCase]:
    """Load gold Q&A cases from a JSON file."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [EvalCase(**case) for case in raw]


METRIC_NAMES = (
    "context_precision_at_k",
    "context_recall",
    "faithfulness",
    "answer_relevancy",
)


def evaluate(
    pipeline: RAGPipeline, cases: list[EvalCase], top_k: int = 3
) -> list[QuestionResult]:
    """Run the pipeline over every gold case and score each metric."""
    results: list[QuestionResult] = []
    for case in cases:
        response = pipeline.query(case.question, top_k=top_k)
        relevant = set(case.relevant_doc_ids)
        retrieved_doc_ids = [r.chunk.doc_id for r in response.results]
        context_texts = [r.chunk.text for r in response.results]
        results.append(
            QuestionResult(
                id=case.id,
                question=case.question,
                context_precision_at_k=context_precision_at_k(
                    retrieved_doc_ids, relevant, top_k
                ),
                context_recall=context_recall(retrieved_doc_ids, relevant),
                faithfulness=faithfulness(response.answer, context_texts),
                answer_relevancy=answer_relevancy(case.question, response.answer),
                retrieved_doc_ids=retrieved_doc_ids,
            )
        )
    return results


def summarize(
    results: list[QuestionResult], top_k: int, thresholds: dict[str, float]
) -> EvalReport:
    """Aggregate per-question results and check them against thresholds."""
    averages = {
        name: (
            sum(getattr(r, name) for r in results) / len(results) if results else 0.0
        )
        for name in METRIC_NAMES
    }
    failures = [
        f"{name}: {averages[name]:.3f} < threshold {thresholds[name]:.3f}"
        for name in METRIC_NAMES
        if name in thresholds and averages[name] < thresholds[name]
    ]
    return EvalReport(
        top_k=top_k,
        num_questions=len(results),
        averages=averages,
        thresholds=dict(thresholds),
        passed=not failures,
        failures=failures,
        results=results,
    )


def write_json_report(report: EvalReport, path: str | Path) -> Path:
    """Write the full report (per-question results + aggregates) as JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(asdict(report), indent=2), encoding="utf-8")
    return path


def write_markdown_report(report: EvalReport, path: str | Path) -> Path:
    """Write a human-readable Markdown summary of the report."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# RAG Evaluation Report",
        "",
        f"- Questions evaluated: **{report.num_questions}**",
        f"- Retrieval top-k: **{report.top_k}**",
        f"- Overall: **{'PASS' if report.passed else 'FAIL'}**",
        "",
        "## Aggregate metrics",
        "",
        "| Metric | Average | Threshold | Status |",
        "| --- | --- | --- | --- |",
    ]
    for name in METRIC_NAMES:
        avg = report.averages[name]
        threshold = report.thresholds.get(name)
        status = (
            "✅" if threshold is None or avg >= threshold else "❌"
        )
        thr = f"{threshold:.2f}" if threshold is not None else "n/a"
        lines.append(f"| `{name}` | {avg:.3f} | {thr} | {status} |")
    if report.failures:
        lines += ["", "## Threshold failures", ""]
        lines += [f"- {failure}" for failure in report.failures]
    lines += ["", "## Per-question results", ""]
    lines += [
        "| ID | P@k | Recall | Faithful. | Relevancy | Retrieved docs |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    for r in report.results:
        docs = ", ".join(r.retrieved_doc_ids)
        lines.append(
            f"| {r.id} | {r.context_precision_at_k:.2f} | {r.context_recall:.2f} "
            f"| {r.faithfulness:.2f} | {r.answer_relevancy:.2f} | {docs} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def run_evaluation(
    corpus_dir: str | Path | None = None,
    gold_path: str | Path | None = None,
    config_path: str | Path | None = None,
    reports_dir: str | Path | None = None,
) -> tuple[EvalReport, int]:
    """Build the index, evaluate in offline (extractive) mode, write reports.

    Returns ``(report, exit_code)`` where the exit code is ``0`` when every
    metric meets its configured threshold, ``1`` otherwise.
    """
    corpus_dir = Path(corpus_dir or PROJECT_ROOT / "data" / "corpus")
    gold_path = Path(gold_path or PROJECT_ROOT / "data" / "gold.json")
    config_path = Path(config_path or PROJECT_ROOT / "eval_config.json")
    reports_dir = Path(reports_dir or PROJECT_ROOT / "reports")

    config = json.loads(config_path.read_text(encoding="utf-8"))
    top_k = int(config.get("top_k", 3))
    thresholds = dict(config.get("thresholds", {}))

    documents = load_corpus(corpus_dir)
    chunks = chunk_documents(documents)
    retriever = HybridRetriever().build(chunks, backend=get_embedding_backend())
    # Force the extractive generator: eval must run fully offline, no API key.
    pipeline = RAGPipeline(retriever, generator=ExtractiveGenerator())

    cases = load_gold(gold_path)
    results = evaluate(pipeline, cases, top_k=top_k)
    report = summarize(results, top_k=top_k, thresholds=thresholds)

    write_json_report(report, reports_dir / "eval_report.json")
    write_markdown_report(report, reports_dir / "eval_report.md")
    return report, 0 if report.passed else 1


def print_summary(report: EvalReport) -> None:
    """Print a compact console summary of the report."""
    print(f"\nEvaluated {report.num_questions} questions (top_k={report.top_k})")
    print(f"{'metric':<24}{'average':>10}{'threshold':>12}  status")
    print("-" * 54)
    for name in METRIC_NAMES:
        avg = report.averages[name]
        threshold = report.thresholds.get(name)
        ok = threshold is None or avg >= threshold
        thr = f"{threshold:.2f}" if threshold is not None else "n/a"
        print(f"{name:<24}{avg:>10.3f}{thr:>12}  {'PASS' if ok else 'FAIL'}")
    if report.failures:
        print("\nThreshold failures:")
        for failure in report.failures:
            print(f"  - {failure}")
    print(f"\nOverall: {'PASS' if report.passed else 'FAIL'}")
    print("Reports written to reports/eval_report.json and reports/eval_report.md")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: run the full offline evaluation."""
    del argv  # no CLI flags yet; configuration lives in eval_config.json
    report, exit_code = run_evaluation()
    print_summary(report)
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
