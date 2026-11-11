"""A small client for Lex's public API (POST /api/v3/{operation}), paced below the site's per-client limits."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Self

import httpx

DEFAULT_BASE_URL = "https://law.soufien.lu"


@dataclass(frozen=True)
class Timed:
    body: dict[str, Any]
    status: int
    seconds: float


class LexClient:
    """Calls one operation at a time. `min_interval` spaces requests so a run stays under the assistant's 20 asks a minute
    and the site's general limit; the server answers 429 otherwise, which this client retries once after a pause."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        *,
        min_interval: float = 3.5,
        timeout: float = 30.0,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
        clock=time.monotonic,
    ) -> None:
        self._http = httpx.Client(
            base_url=base_url,
            timeout=timeout,
            transport=transport,
            headers={"content-type": "application/json", "user-agent": "lex-eval/0.1"},
        )
        self._min_interval = min_interval
        self._sleep = sleep
        self._clock = clock
        self._last: float | None = None

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _pace(self) -> None:
        if self._last is not None:
            wait = self._min_interval - (self._clock() - self._last)
            if wait > 0:
                self._sleep(wait)
        self._last = self._clock()

    def call(self, operation: str, parameters: dict[str, Any]) -> Timed:
        payload = {"operation_id": operation, "parameters": parameters}
        for attempt in range(2):
            self._pace()
            started = self._clock()
            response = self._http.post(f"/api/v3/{operation}", json=payload)
            seconds = self._clock() - started
            if response.status_code == 429 and attempt == 0:
                self._sleep(60)
                continue
            return Timed(body=response.json(), status=response.status_code, seconds=seconds)
        raise RuntimeError("unreachable")

    def ask(self, question: str) -> Timed:
        return self.call("ask", {"question": question})

    def verify(self, permalink: str) -> Timed:
        return self.call("verify", {"identifier": permalink})

    def coverage(self) -> Timed:
        return self.call("coverage", {})
