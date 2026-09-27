"""Shared System One request handling for HTTP and CLI clients."""

from __future__ import annotations

import math
from typing import Protocol

from actlogit.schema import (
    ChoiceAnswer,
    ChoiceQuestion,
    DecisionRequest,
    DecisionResponse,
    NoulAnswer,
    NoulQuestion,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
)


class Predictor(Protocol):
    @property
    def model_id(self) -> str: ...

    def predict_many(self, requests: list[DecisionRequest]) -> list[DecisionResponse]: ...


class UnknownModelError(ValueError):
    pass


def predict(engine: Predictor, request: SystemOneRequest) -> SystemOneResponse:
    if request.model not in {"actlogit", "default", engine.model_id}:
        raise UnknownModelError(
            "model must name this server's configured model, 'actlogit', or 'default'"
        )
    questions = list(request.questions.values())
    results = engine.predict_many([question.to_decision(request.state) for question in questions])
    answers = {}
    for (key, question), result in zip(request.questions.items(), results, strict=True):
        probabilities = result.probabilities
        if isinstance(question, NoulQuestion):
            answers[key] = NoulAnswer(noul=probabilities["true"])
        elif isinstance(question, ChoiceQuestion):
            answers[key] = ChoiceAnswer(
                choice=result.choice, probabilities=probabilities, confidence=result.confidence
            )
        else:
            answers[key] = ScoreAnswer(
                score=math.fsum(
                    index * probabilities[str(index)] for index in range(len(question.criteria))
                ),
                legend={str(index): value for index, value in enumerate(question.criteria)},
                probabilities=probabilities,
                confidence=result.confidence,
            )
    return SystemOneResponse(model=engine.model_id, answers=answers)
