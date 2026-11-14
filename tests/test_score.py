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


def _case(**kw):
    from lex_eval.cases import Case
    base = dict(id="c", question="q", expect_verdict="answer", question_type="t", expect_work="/lu-legilux/loi-1997-07-27-n10",
                expect_anchor="art_44", suite="insurance")
    base.update(kw)
    return Case(**base)


def test_each_result_has_its_kind_and_a_wrong_article_is_told_apart_from_not_found():
    from lex_eval.score import Quote, outcome_of
    right = [Quote("/lu-legilux/loi-1997-07-27-n10/2025-11-25--x/fra/art_44", None, "art_44")]
    wrong = [Quote("/lu-legilux/loi-1997-07-27-n10/2025-11-25--x/fra/art_70", None, "art_70")]
    case = _case()
    assert outcome_of(case, "answer", right, True, None) == "right_article"
    assert outcome_of(case, "answer", wrong, False, None) == "wrong_article"
    assert outcome_of(case, "clarify", [], False, None) == "not_found"
    assert outcome_of(case, "point", [], False, None) == "not_found"
    assert outcome_of(case, "refuse", [], False, None) == "wrongly_refused"
    assert outcome_of(case, "answer", right, True, "http 500") == "error"
    advice = _case(expect_verdict="refuse", expect_work=None, expect_anchor=None)
    assert outcome_of(advice, "refuse", [], None, None) == "right_behaviour"
    assert outcome_of(advice, "answer", right, None, None) == "wrong_behaviour"


def test_one_wrong_article_on_the_insurance_suite_fails_the_run_and_so_does_a_regression(tmp_path):
    from lex_eval.run import gates, regressions
    from lex_eval.score import CaseResult
    metrics = {"quote_verified_rate": 1.0, "non_template_strings": 0, "errors": 0, "verdict_accuracy": 0.9,
               "outcome.wrong_article.insurance": 1, "outcome.wrong_article.eu": 0}
    assert any("wrong_article on insurance" in item for item in gates(metrics, 0.8, gate_suite="insurance"))
    assert gates({**metrics, "outcome.wrong_article.insurance": 0}, 0.8, gate_suite="insurance") == []
    baseline = tmp_path / "results.jsonl"
    baseline.write_text('{"id": "a", "outcome": "right_article"}\n{"id": "b", "outcome": "not_found"}\n', encoding="utf-8")

    def result(id_, outcome):
        return CaseResult(id=id_, question_type="t", expected="answer", verdict="answer", verdict_ok=True, citation_expected=True,
                          citation_ok=True, quotes=1, quotes_verified=1, non_template_strings=0, replay_identical=None,
                          replay_same_plan=None, seconds=1.0, outcome=outcome)
    assert regressions([result("a", "not_found"), result("b", "not_found")], baseline) == ["a"], "only what passed before"
    assert regressions([result("a", "right_article")], baseline) == []


def test_the_cost_is_an_estimate_from_the_prompt_and_the_plan_and_nothing_when_no_plan_was_made():
    from lex_eval.run import cost_estimate
    assert cost_estimate("q", None) == 0.0
    cost = cost_estimate("Quel est le délai de prescription ?", {"steps": [{"operation_id": "search"}]})
    assert 0.0005 < cost < 0.002, cost


def test_the_shipped_v2_cases_have_the_insurance_suite_and_an_article_for_each_concept_question():
    from lex_eval.cases import load_cases
    cases = load_cases("cases/lex-assistant-v2.jsonl")
    insurance = [c for c in cases if c.suite == "insurance"]
    assert len(cases) >= 55 and len(insurance) >= 20
    assert all(c.expect_anchor for c in cases if c.question_type.startswith("concept_"))
