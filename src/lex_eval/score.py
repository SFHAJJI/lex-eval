"""Scoring, as pure functions over the API's JSON: what the assistant answered, whether it cited the expected act and
article, whether each quotation verifies against the published text, and the aggregate metrics, per question type and
never only as one average."""

from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from typing import Any

from .cases import Case


@dataclass(frozen=True)
class Quote:
    permalink: str
    text_sha256: str | None
    anchor: str | None


@dataclass
class CaseResult:
    id: str
    question_type: str
    expected: str
    verdict: str | None
    verdict_ok: bool
    citation_expected: bool
    citation_ok: bool | None
    quotes: int
    quotes_verified: int
    non_template_strings: int
    replay_identical: bool | None
    replay_same_plan: bool | None
    seconds: float
    error: str | None = None
    plan_sha256: str | None = None
    notes: list[str] = field(default_factory=list)
    suite: str = "general"
    # What kind of result it is (OUTCOMES): the headline is "wrong_article", whose target is zero.
    outcome: str = "right_behaviour"
    chosen_work: str | None = None
    chosen_article: str | None = None

    def row(self) -> dict[str, Any]:
        return asdict(self)


def _walk(node: Any) -> Iterator[dict[str, Any]]:
    if isinstance(node, dict):
        yield node
        for value in node.values():
            yield from _walk(value)
    elif isinstance(node, list):
        for value in node:
            yield from _walk(value)


def answer_value(envelope: dict[str, Any]) -> dict[str, Any] | None:
    result = envelope.get("result") or {}
    value = result.get("value")
    return value if isinstance(value, dict) else None


REFUSAL_VERDICTS = {"advice_boundary": "refuse"}


def verdict_of(envelope: dict[str, Any]) -> str | None:
    """The assistant's verdict: the answer dossier's, or a presentation card's (localization_unavailable,
    assistant_v3_unavailable), or the envelope's refusal code."""
    value = answer_value(envelope)
    if value is not None:
        if isinstance(value.get("verdict"), str):
            return value["verdict"]
        if isinstance(value.get("presentation_result"), str):
            return value["presentation_result"]
    refusal = envelope.get("refusal")
    if isinstance(refusal, dict) and isinstance(refusal.get("code"), str):
        # A refuse travels as the advice_boundary refusal, its answer dossier in the payload.
        return REFUSAL_VERDICTS.get(refusal["code"], refusal["code"])
    return None


def quotes_of(envelope: dict[str, Any]) -> list[Quote]:
    value = answer_value(envelope) or {}
    found = []
    for node in _walk(value.get("parts")):
        if node.get("kind") == "quote" and isinstance(node.get("article_permalink"), str):
            found.append(Quote(node["article_permalink"], node.get("article_text_sha256"), node.get("anchor")))
    return found


def trace_of(envelope: dict[str, Any]) -> dict[str, Any]:
    value = answer_value(envelope) or {}
    trace = value.get("trace")
    return trace if isinstance(trace, dict) else {}


def cites(quote: Quote, work: str, anchor: str | None) -> bool:
    if not quote.permalink.startswith(work.rstrip("/") + "/"):
        return False
    return anchor is None or quote.anchor == anchor


def quote_verifies(quote: Quote, verify_envelope: dict[str, Any]) -> bool:
    """A quotation verifies when the site's own verify operation, asked for the quotation's permalink, finds the
    published text's digest and it is the digest the answer carried."""
    value = answer_value(verify_envelope) or {}
    if value.get("verdict") != "digest_matches":
        return False
    served = value.get("article_text_sha256")
    return quote.text_sha256 is None or served is None or served == quote.text_sha256


def score_case(
    case: Case,
    first: dict[str, Any],
    seconds: float,
    *,
    replay: dict[str, Any] | None = None,
    verifications: dict[str, dict[str, Any]] | None = None,
    status: int = 200,
) -> CaseResult:
    verdict = verdict_of(first)
    error = None
    if status != 200:
        error = f"http {status}"
    elif verdict == "rate_limited":
        error = "rate_limited"
    quotes = quotes_of(first)
    verifications = verifications or {}
    verified = sum(
        1 for quote in quotes if quote.permalink in verifications and quote_verifies(quote, verifications[quote.permalink])
    )
    citation_expected = case.expect_work is not None
    citation_ok = (
        any(cites(quote, case.expect_work, case.expect_anchor) for quote in quotes) if citation_expected else None
    )
    value = answer_value(first) or {}
    gates = value.get("gates") or {}
    plan = trace_of(first).get("plan_sha256")
    replay_identical = replay_same_plan = None
    if replay is not None:
        # The same verdict twice; and, when the planner ran, the same plan (temperature 0 does not promise it).
        replay_identical = verdict == verdict_of(replay)
        replay_same_plan = None if plan is None else plan == trace_of(replay).get("plan_sha256")
    chosen_work, chosen_article = chosen_of(first)
    return CaseResult(
        suite=case.suite,
        outcome=outcome_of(case, verdict, quotes, citation_ok, error),
        chosen_work=chosen_work,
        chosen_article=chosen_article,
        id=case.id,
        question_type=case.question_type,
        expected=case.expect_verdict,
        verdict=verdict,
        verdict_ok=error is None and verdict == case.expect_verdict,
        citation_expected=citation_expected,
        citation_ok=citation_ok,
        quotes=len(quotes),
        quotes_verified=verified,
        non_template_strings=int(gates.get("non_template_strings") or 0),
        replay_identical=replay_identical,
        replay_same_plan=replay_same_plan,
        seconds=seconds,
        error=error,
        plan_sha256=plan,
    )


# The kinds of result a case can have. For a case that expects an article: the right article, a wrong article (an answer
# that quotes another article or act: the failure that matters most, target zero), not found (no answer: a precision asked,
# a pointer, no quotation) or wrongly refused (refused as advice, or no language); and an error. For any other case: the
# right behaviour or the wrong one.
OUTCOMES = ("right_article", "wrong_article", "not_found", "wrongly_refused", "right_behaviour", "wrong_behaviour", "error")
ANSWERS = frozenset({"answer", "answer_with_enrichment"})


def outcome_of(case: Case, verdict: str | None, quotes: list[Quote], citation_ok: bool | None, error: str | None) -> str:
    if error is not None:
        return "error"
    if case.expect_work is not None:
        if verdict in ANSWERS and citation_ok:
            return "right_article"
        if verdict in ANSWERS and quotes:
            return "wrong_article"
        if verdict in {"refuse", "localization_unavailable"}:
            return "wrongly_refused"
        return "not_found"
    return "right_behaviour" if verdict == case.expect_verdict else "wrong_behaviour"


def chosen_of(envelope: dict[str, Any]) -> tuple[str | None, str | None]:
    """The act and article the answer chose: the first quotation's, as its permalink and anchor name them."""
    quotes = quotes_of(envelope)
    if not quotes:
        return None, None
    parts = quotes[0].permalink.split("/")
    work = "/".join(parts[:3]) if len(parts) > 2 else None
    return work, quotes[0].anchor


def search_step(envelope: dict[str, Any]) -> dict[str, Any] | None:
    """The parameters of the plan's first search step, as the trace records them after binding."""
    for step in trace_of(envelope).get("steps") or []:
        if step.get("operation_id") == "search" and isinstance(step.get("parameters"), dict):
            return step["parameters"]
    return None


def trace_record(case: Case, result: "CaseResult", envelope: dict[str, Any], search: dict[str, Any] | None) -> dict[str, Any]:
    """One question's trace for the run's artifacts: the plan, each operation run, the search's first hits and its best-hit
    decision (the same search asked again by the harness), and the article chosen."""
    trace = trace_of(envelope)
    steps = [{key: step.get(key) for key in ("step_id", "operation_id", "parameters", "outcome", "verdict", "refusal_code")}
             for step in trace.get("steps") or []]
    hits = None
    best = None
    if search is not None:
        value = answer_value(search) or {}
        hits = [{"work": hit.get("work_key") or hit.get("celex"), "article": hit.get("publisher_id"), "lane": hit.get("lane")}
                for hit in (value.get("hits") or [])[:3]]
        best_hit = value.get("best_hit") or {}
        best = {"decision": best_hit.get("decision"), "hit": best_hit.get("hit")}
    return {
        "id": case.id, "question": case.question, "suite": case.suite, "expected": {"verdict": case.expect_verdict,
        "work": case.expect_work, "article": case.expect_anchor}, "verdict": result.verdict, "outcome": result.outcome,
        "planner_id": trace.get("planner_id"), "plan": trace.get("plan"), "steps": steps, "search_hits": hits,
        "best_hit": best, "chosen": {"work": result.chosen_work, "article": result.chosen_article}, "seconds": result.seconds,
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(q * (len(ordered) - 1))))
    return ordered[index]


def aggregate(results: list[CaseResult]) -> dict[str, float]:
    """The run's metrics. Rates are over the cases they apply to; per question type, the verdict accuracy, so a good
    average can never hide a weak subset."""
    metrics: dict[str, float] = {}
    n = len(results)
    metrics["cases"] = n
    metrics["errors"] = sum(1 for r in results if r.error)
    metrics["verdict_accuracy"] = sum(r.verdict_ok for r in results) / n if n else 0.0
    cited = [r for r in results if r.citation_expected]
    metrics["citation_accuracy"] = sum(bool(r.citation_ok) for r in cited) / len(cited) if cited else 0.0
    quotes = sum(r.quotes for r in results)
    metrics["quotes"] = quotes
    metrics["quote_verified_rate"] = sum(r.quotes_verified for r in results) / quotes if quotes else 1.0
    metrics["non_template_strings"] = sum(r.non_template_strings for r in results)
    replayed = [r for r in results if r.replay_identical is not None]
    metrics["replay_same_verdict_rate"] = sum(bool(r.replay_identical) for r in replayed) / len(replayed) if replayed else 0.0
    planned = [r for r in results if r.replay_same_plan is not None]
    metrics["replay_same_plan_rate"] = sum(bool(r.replay_same_plan) for r in planned) / len(planned) if planned else 0.0
    seconds = [r.seconds for r in results if not r.error]
    metrics["latency_p50_s"] = statistics.median(seconds) if seconds else 0.0
    metrics["latency_p95_s"] = percentile(seconds, 0.95)
    by_type: dict[str, list[CaseResult]] = defaultdict(list)
    for r in results:
        by_type[r.question_type].append(r)
    for question_type, group in sorted(by_type.items()):
        metrics[f"verdict_accuracy.{question_type}"] = sum(r.verdict_ok for r in group) / len(group)
    # Per kind of result, overall and per suite: "wrong_article" is the headline, and its target is zero.
    for outcome in OUTCOMES:
        metrics[f"outcome.{outcome}"] = sum(1 for r in results if r.outcome == outcome)
    for suite in sorted({r.suite for r in results}):
        group = [r for r in results if r.suite == suite]
        for outcome in OUTCOMES:
            metrics[f"outcome.{outcome}.{suite}"] = sum(1 for r in group if r.outcome == outcome)
    return metrics
