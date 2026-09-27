import json

import pytest

from actlogit.data import load_records
from actlogit.schema import DecisionRequest, NoulQuestion, ScoreQuestion, TrainingRecord


def test_target_keys_align_by_action_id_not_dictionary_order(decision, choice_question):
    record = TrainingRecord(
        state=decision.state,
        question=choice_question,
        target={"account": 10, "technical": 20, "billing": 70},
    )
    assert record.distribution() == [0.7, 0.2, 0.1]


@pytest.mark.parametrize(
    "target",
    [
        {"billing": 1},
        {"billing": 0, "technical": 0, "account": 0},
        {"billing": -1, "technical": 2, "account": 0},
        {"billing": float("nan"), "technical": 1, "account": 0},
        {"billing": float("inf"), "technical": 1, "account": 0},
    ],
)
def test_malformed_distribution_fails(decision, choice_question, target):
    with pytest.raises(ValueError):
        TrainingRecord(state=decision.state, question=choice_question, target=target)


def test_duplicate_choice_ids_fail(decision):
    data = decision.model_dump()
    data["choices"].append(data["choices"][0])
    with pytest.raises(ValueError, match="unique"):
        DecisionRequest.model_validate(data)


def test_jsonl_reports_line_number(tmp_path, decision, choice_question):
    path = tmp_path / "bad.jsonl"
    record = TrainingRecord(
        state=decision.state,
        question=choice_question,
        target={"billing": 1, "technical": 0, "account": 0},
    )
    path.write_text(record.model_dump_json() + "\n" + json.dumps({"bad": 1}) + "\n")
    with pytest.raises(ValueError, match="bad.jsonl:2"):
        load_records(path)


def test_empty_data_fails(tmp_path):
    path = tmp_path / "empty.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError, match="empty"):
        load_records(path)


def test_noul_and_score_target_order_and_roundtrip():
    records = [
        TrainingRecord(state="refund", question=NoulQuestion(), target={"false": 8, "true": 2}),
        TrainingRecord(
            state="bug",
            question=ScoreQuestion(criteria=["low", {"description": "medium"}, "high"]),
            target={"2": 6, "0": 1, "1": 3},
        ),
    ]
    assert records[0].distribution() == [0.2, 0.8]
    assert records[1].distribution() == [0.1, 0.3, 0.6]
    for record in records:
        restored = TrainingRecord.model_validate_json(record.model_dump_json())
        assert restored.decision == record.question.to_decision(record.state)
        assert restored.distribution() == record.distribution()
        assert "decision" not in record.model_dump()


@pytest.mark.parametrize("target", [{"true": 1}, {"yes": 1, "no": 0}, {"true": 0, "false": 0}])
def test_noul_requires_complete_nonzero_targets(target):
    with pytest.raises(ValueError):
        TrainingRecord(state="x", question=NoulQuestion(), target=target)


def test_score_requires_a_distribution_not_a_scalar():
    with pytest.raises(ValueError):
        TrainingRecord(state="x", question=ScoreQuestion(criteria=["low", "high"]), target=0.7)
