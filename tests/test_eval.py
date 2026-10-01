"""Tests for the evaluation metrics."""

import pytest

from rag_eval.eval import (
    answer_relevancy,
    context_precision_at_k,
    context_recall,
    faithfulness,
)

CHUNK = (
    "FleetOps commits to 99.95% monthly API uptime. If uptime falls below "
    "99.95% in a month, affected customers receive a 10% service credit."
)


def test_context_precision_perfect():
    assert context_precision_at_k(["a", "b"], {"a", "b"}, 2) == pytest.approx(1.0)


def test_context_precision_partial():
    assert context_precision_at_k(["a", "x", "b"], {"a", "b"}, 3) == pytest.approx(2 / 3)


def test_context_precision_empty():
    assert context_precision_at_k([], {"a"}, 3) == 0.0
    assert context_precision_at_k(["a"], {"a"}, 0) == 0.0


def test_context_recall():
    assert context_recall(["a", "x"], {"a", "b"}) == pytest.approx(0.5)
    assert context_recall(["a", "b"], {"a", "b"}) == pytest.approx(1.0)
    assert context_recall([], {"a"}) == pytest.approx(0.0)


def test_faithfulness_fully_supported():
    answer = CHUNK  # verbatim chunk text: every claim is supported
    assert faithfulness(answer, [CHUNK]) == pytest.approx(1.0)


def test_faithfulness_detects_unsupported_claim():
    answer = CHUNK + " The moon is made of green cheese and tastes delicious."
    score = faithfulness(answer, [CHUNK])
    assert 0.0 < score < 1.0


def test_faithfulness_empty_answer():
    assert faithfulness("", [CHUNK]) == 0.0
    assert faithfulness("   ", [CHUNK]) == 0.0


def test_answer_relevancy_high_overlap():
    question = "What uptime guarantee does FleetOps commit to?"
    answer = "FleetOps commits to 99.95% monthly API uptime guarantee."
    assert answer_relevancy(question, answer) > 0.5


def test_answer_relevancy_no_overlap():
    assert answer_relevancy("What is the refund policy?", "The moon is cheese.") == 0.0


def test_answer_relevancy_empty_question():
    assert answer_relevancy("", "some answer") == 0.0
