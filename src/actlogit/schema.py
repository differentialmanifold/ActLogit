from __future__ import annotations

import json
import math
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Choice(StrictModel):
    id: Annotated[str, Field(min_length=1)]
    description: Annotated[str, Field(min_length=1)]
    payload: JsonValue = None


class DecisionRequest(StrictModel):
    """Normalized engine input shared by all question types; not a public API."""

    state: JsonValue
    instructions: JsonValue = "Choose the best action for the supplied state."
    choices: Annotated[list[Choice], Field(min_length=1)]

    @model_validator(mode="after")
    def unique_choices(self) -> DecisionRequest:
        ids = [choice.id for choice in self.choices]
        if len(ids) != len(set(ids)):
            raise ValueError("choice IDs must be unique within each decision")
        return self


class DecisionResponse(StrictModel):
    choice: str
    probabilities: dict[str, float]
    confidence: float  # Maximum conditional choice probability, not a calibrated success rate.
    token: str
    model: str


def _description(value: JsonValue, fallback: str) -> str:
    if value is None:
        return fallback
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


class ChoiceQuestion(StrictModel):
    type: Literal["choice"] = "choice"
    criteria: Annotated[dict[str, JsonValue], Field(min_length=1, max_length=255)]
    instructions: JsonValue = "Choose the best option."

    @model_validator(mode="after")
    def usable_criteria(self) -> ChoiceQuestion:
        if any(not key or value == "" for key, value in self.criteria.items()):
            raise ValueError("criteria IDs and string descriptions must be nonempty")
        return self

    def to_decision(self, state: JsonValue) -> DecisionRequest:
        return DecisionRequest(
            state=state,
            instructions=self.instructions,
            choices=[
                Choice(id=key, description=_description(value, key))
                for key, value in self.criteria.items()
            ],
        )


class NoulCriteria(StrictModel):
    true: JsonValue = "The answer to the question is yes."
    false: JsonValue = "The answer to the question is no."


class NoulQuestion(StrictModel):
    type: Literal["noul"] = "noul"
    instructions: JsonValue = "Is the supplied statement true?"
    criteria: NoulCriteria | None = None

    def to_decision(self, state: JsonValue) -> DecisionRequest:
        criteria = self.criteria or NoulCriteria()
        return DecisionRequest(
            state=state,
            instructions=self.instructions,
            choices=[
                Choice(id="true", description=_description(criteria.true, "Yes")),
                Choice(id="false", description=_description(criteria.false, "No")),
            ],
        )


class ScoreQuestion(StrictModel):
    type: Literal["score"] = "score"
    instructions: JsonValue = "Rate the supplied state against the listed levels."
    criteria: Annotated[list[JsonValue], Field(min_length=2, max_length=10)]

    @model_validator(mode="after")
    def described_levels(self) -> ScoreQuestion:
        if any(level is None or level == "" for level in self.criteria):
            raise ValueError("score levels must have nonempty descriptions")
        return self

    def to_decision(self, state: JsonValue) -> DecisionRequest:
        return DecisionRequest(
            state=state,
            instructions=self.instructions,
            choices=[
                Choice(id=str(index), description=_description(value, str(index)))
                for index, value in enumerate(self.criteria)
            ],
        )


Question = Annotated[ChoiceQuestion | NoulQuestion | ScoreQuestion, Field(discriminator="type")]


class SystemOneRequest(StrictModel):
    model: str = "actlogit"
    state: JsonValue
    questions: Annotated[dict[str, Question], Field(min_length=1, max_length=32)]


class ChoiceAnswer(StrictModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class NoulAnswer(StrictModel):
    type: Literal["noul"] = "noul"
    noul: float


class ScoreAnswer(StrictModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, JsonValue]
    probabilities: dict[str, float]
    confidence: float


Answer = Annotated[ChoiceAnswer | NoulAnswer | ScoreAnswer, Field(discriminator="type")]


class Usage(StrictModel):
    input_tokens: int | None = None
    output_tokens: int | None = None


class SystemOneResponse(StrictModel):
    model: str
    answers: dict[str, Answer]
    usage: Usage = Field(default_factory=Usage)


class TrainingRecord(StrictModel):
    state: JsonValue
    question: Question
    target: dict[str, Annotated[float, Field(ge=0)]]
    weight: Annotated[float, Field(gt=0)] = 1.0
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def decision(self) -> DecisionRequest:
        return self.question.to_decision(self.state)

    @model_validator(mode="after")
    def complete_target(self) -> TrainingRecord:
        ids = {choice.id for choice in self.decision.choices}
        if set(self.target) != ids:
            raise ValueError(
                "target keys must exactly match the question outcomes, including zeros"
            )
        try:
            total = math.fsum(self.target.values())
        except OverflowError as exc:
            raise ValueError("target weights overflow; rescale them") from exc
        if not math.isfinite(total) or total <= 0:
            raise ValueError("target must have a finite, positive total weight")
        return self

    def distribution(self) -> list[float]:
        total = math.fsum(self.target.values())
        return [self.target[choice.id] / total for choice in self.decision.choices]
