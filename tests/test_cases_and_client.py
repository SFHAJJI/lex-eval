import json
from pathlib import Path

import httpx
import pytest
import respx

from lex_eval.cases import Case, load_cases
from lex_eval.client import LexClient

ROOT = Path(__file__).parent.parent


def test_the_shipped_cases_load_and_are_unique():
    cases = load_cases(ROOT / "cases" / "lex-assistant-v1.jsonl")
    assert len(cases) >= 10
    assert len({c.id for c in cases}) == len(cases)
    assert {c.question_type for c in cases} >= {"answer_lu_article", "answer_eu_article", "refuse_advice", "point_not_held"}


def test_a_citation_belongs_only_to_an_answer():
    with pytest.raises(ValueError):
        Case("x", "q", "refuse", "refuse_advice", "/lu-legilux/code-civil")


def test_an_unknown_verdict_is_refused():
    with pytest.raises(ValueError):
        Case("x", "q", "maybe", "t")


@respx.mock
def test_the_client_posts_the_operation_and_retries_once_after_429():
    route = respx.post("https://law.soufien.lu/api/v3/ask").mock(side_effect=[
        httpx.Response(429, json={"code": "rate_limited"}),
        httpx.Response(200, json={"verdict": "answer"}),
    ])
    slept = []
    client = LexClient(min_interval=0, sleep=slept.append)
    timed = client.ask("Que dit l'article 1er du Code civil ?")
    assert timed.status == 200 and timed.body == {"verdict": "answer"}
    assert route.call_count == 2
    assert 60 in slept
    sent = json.loads(route.calls[0].request.content)
    assert sent == {"operation_id": "ask", "parameters": {"question": "Que dit l'article 1er du Code civil ?"}}


def test_the_client_paces_requests():
    now = [0.0]
    slept = []

    def sleep(seconds):
        slept.append(seconds)
        now[0] += seconds

    transport = httpx.MockTransport(lambda request: httpx.Response(200, json={}))
    client = LexClient(min_interval=3.5, transport=transport, sleep=sleep, clock=lambda: now[0])
    client.coverage()
    client.coverage()
    assert slept and slept[0] == pytest.approx(3.5)
