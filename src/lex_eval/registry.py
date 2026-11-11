"""The deployed assistant as an MLflow model, so the registry records which release, planner and prompt passed which
evaluation. The model holds no weights: Lex's planner runs behind the site's API (Azure OpenAI, EU data zone), and this
wrapper calls it, which is what the registry versions."""

from __future__ import annotations

from typing import Any

import mlflow.pyfunc

from .client import LexClient
from .score import verdict_of


class LexAssistantModel(mlflow.pyfunc.PythonModel):
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url

    def predict(self, context: Any, model_input: Any, params: dict[str, Any] | None = None) -> list[str | None]:
        questions = list(model_input["question"]) if hasattr(model_input, "__getitem__") else list(model_input)
        with LexClient(self.base_url) as client:
            return [verdict_of(client.ask(question).body) for question in questions]
