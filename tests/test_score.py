import copy
import json
from pathlib import Path

import pytest

from lex_eval.cases import Case
from lex_eval.score import aggregate, quotes_of, score_case, verdict_of

FIXTURES = Path(__file__).parent / "fixtures"
ANSWER = json.loads((FIXTURES / "answer-code-civil-art1.json").read_text(encoding="utf-8"))
VERIFY = json.loads((FIXTURES / "verify-code-civil-art1.json").read_text(encoding="utf-8"))

CODE_CIVIL = Case("lu-code-civil-art1", "Que dit l'article 1er du Code civil ?", "answer", "answer_lu_article",
                  "/lu-legilux/code-civil", "art_1er")


def verifications():
    return {q.permalink: VERIFY for q in quotes_of(ANSWER)}


def test_a_real_answer_is_scored_as_cited_and_verified():
    result = score_case(CODE_CIVIL, ANSWER, 1.2, replay=ANSWER, verifications=verifications())
    assert result.verdict == "answer"
    assert result.verdict_ok
    assert result.citation_ok
    assert result.quotes == 1 and result.quotes_verified == 1
    assert result.non_template_strings == 0
    assert result.replay_identical and result.replay_same_plan


def test_a_quote_whose_digest_differs_does_not_verify():
    tampered = copy.deepcopy(VERIFY)
    tampered["result"]["value"]["article_text_sha256"] = "0" * 64
    checks = {q.permalink: tampered for q in quotes_of(ANSWER)}
    assert score_case(CODE_CIVIL, ANSWER, 1.0, verifications=checks).quotes_verified == 0


def test_a_quote_of_another_act_is_not_the_expected_citation():
    other = Case("x", "q", "answer", "answer_lu_article", "/lu-legilux/code-travail")
    assert score_case(other, ANSWER, 1.0).citation_ok is False


def test_a_rate_limited_ask_is_an_error_never_a_verdict():
    limited = {"refusal": {"code": "rate_limited"}, "result": None}
    result = score_case(CODE_CIVIL, limited, 0.1)
    assert verdict_of(limited) == "rate_limited"
    assert result.error == "rate_limited" and not result.verdict_ok


def test_a_changed_plan_on_replay_is_not_identical():
    other = copy.deepcopy(ANSWER)
    other["result"]["value"]["trace"]["plan_sha256"] = "f" * 64
    result = score_case(CODE_CIVIL, ANSWER, 1.0, replay=other)
    assert result.replay_identical is True, "the same verdict"
    assert result.replay_same_plan is False, "another plan"


def test_a_refusal_of_advice_is_the_refuse_verdict():
    refused = {"refusal": {"code": "advice_boundary", "helpful_payload": {}}, "result": None}
    case = Case("r", "q", "refuse", "refuse_advice")
    result = score_case(case, refused, 1.0, replay=refused)
    assert result.verdict == "refuse" and result.verdict_ok
    assert result.replay_identical is True and result.replay_same_plan is None


def test_metrics_are_reported_per_question_type():
    good = score_case(CODE_CIVIL, ANSWER, 1.0, verifications=verifications())
    bad_case = Case("eu", "q", "answer", "answer_eu_article", "/eu-eurlex/32016R0679")
    bad = score_case(bad_case, {"result": {"value": {"verdict": "clarify"}}}, 2.0)
    metrics = aggregate([good, bad])
    assert metrics["verdict_accuracy"] == pytest.approx(0.5)
    assert metrics["verdict_accuracy.answer_lu_article"] == 1.0
    assert metrics["verdict_accuracy.answer_eu_article"] == 0.0
    assert metrics["citation_accuracy"] == pytest.approx(0.5)
    assert metrics["quote_verified_rate"] == 1.0
