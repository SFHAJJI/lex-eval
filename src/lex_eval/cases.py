"""The test questions: one JSON object per line, each with the verdict the assistant should give and, for an answer,
the act and article its quotation must come from."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

VERDICTS = frozenset({"answer", "clarify", "point", "refuse", "split", "localization_unavailable"})


@dataclass(frozen=True)
class Case:
    id: str
    question: str
    expect_verdict: str
    question_type: str
    expect_work: str | None = None
    expect_anchor: str | None = None
    note: str = ""
    # The subset a case belongs to, for the gates that read one subset ("insurance": the 1997 insurance-contract law).
    suite: str = "general"

    def __post_init__(self) -> None:
        if self.expect_verdict not in VERDICTS:
            raise ValueError(f"{self.id}: unknown verdict {self.expect_verdict!r}")
        if self.expect_anchor is not None and self.expect_work is None:
            raise ValueError(f"{self.id}: an expected article needs its act (expect_work)")
        if self.expect_work is not None and self.expect_verdict != "answer":
            raise ValueError(f"{self.id}: only an answer names the act and article it quotes")


def load_cases(path: str | Path) -> list[Case]:
    cases: list[Case] = []
    seen: set[str] = set()
    for number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        # The held-out file (heldout-v1.jsonl, fixed by its SHA-256 before any run) names the article "expect_article" and
        # its subset "set", and gives no question type: read as the same fields, never by rewriting the file.
        if "expect_article" in raw:
            raw["expect_anchor"] = raw.pop("expect_article")
        if "set" in raw:
            raw["suite"] = raw.pop("set")
        raw.setdefault("question_type", "heldout")
        case = Case(**raw)
        if case.id in seen:
            raise ValueError(f"line {number}: duplicate case id {case.id}")
        seen.add(case.id)
        cases.append(case)
    if not cases:
        raise ValueError(f"{path} holds no case")
    return cases


def dataset_sha256(path: str | Path) -> str:
    """The digest of the case file's bytes: the dataset version every run records."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
