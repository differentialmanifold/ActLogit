import pytest
from fastapi.testclient import TestClient

from actlogit.schema import DecisionResponse, SystemOneRequest
from actlogit.server import create_app


class FakeEngine:
    model_id = "test-local-model"

    def __init__(self):
        self.requests = []

    def predict_many(self, requests):
        self.requests.extend(requests)
        results = []
        for request in requests:
            ids = [choice.id for choice in request.choices]
            values = {
                ("true", "false"): [0.12, 0.88],
                ("0", "1", "2"): [0.1, 0.3, 0.6],
            }.get(tuple(ids), [1 / len(ids)] * len(ids))
            index = max(range(len(ids)), key=values.__getitem__)
            results.append(
                DecisionResponse(
                    choice=ids[index],
                    probabilities=dict(zip(ids, values, strict=True)),
                    confidence=values[index],
                    token="ABC"[index],
                    model=self.model_id,
                )
            )
        return results


@pytest.fixture
def mixed_request():
    return {
        "state": {"ticket": "refund"},
        "questions": {
            "team": {
                "type": "choice",
                "criteria": {"billing": None, "technical": {"description": "Bugs"}},
            },
            "refund": {
                "type": "noul",
                "instructions": "Does the customer request a refund?",
                "criteria": {"true": {"meaning": "Refund requested"}, "false": "No refund"},
            },
            "urgency": {
                "type": "score",
                "instructions": {"question": "How urgent?"},
                "criteria": ["Routine", {"description": "Today"}, "Immediately"],
            },
        },
    }


def test_mixed_questions_return_typed_answers(mixed_request):
    engine = FakeEngine()
    client = TestClient(create_app(engine))
    response = client.post("/v1/systemone", json=mixed_request)
    assert response.status_code == 200
    result = response.json()
    assert result["model"] == engine.model_id
    assert result["usage"] == {"input_tokens": None, "output_tokens": None}
    assert result["answers"] == {
        "team": {
            "type": "choice",
            "choice": "billing",
            "probabilities": {"billing": 0.5, "technical": 0.5},
            "confidence": 0.5,
        },
        "refund": {"type": "noul", "noul": 0.12},
        "urgency": {
            "type": "score",
            "score": pytest.approx(1.5),
            "probabilities": {"0": 0.1, "1": 0.3, "2": 0.6},
            "legend": {"0": "Routine", "1": {"description": "Today"}, "2": "Immediately"},
            "confidence": 0.6,
        },
    }
    assert len(engine.requests) == 3
    assert all(request.state == mixed_request["state"] for request in engine.requests)
    assert engine.requests[0].choices[0].description == "billing"
    assert engine.requests[1].choices[0].description == '{"meaning": "Refund requested"}'


def test_noul_without_criteria_and_one_choice():
    client = TestClient(create_app(FakeEngine()))
    response = client.post(
        "/v1/systemone",
        json={
            "state": "x",
            "questions": {
                "yes": {"type": "noul", "instructions": "Is this a refund?"},
                "only": {"type": "choice", "criteria": {"one": "Only option"}},
            },
        },
    )
    assert response.status_code == 200
    assert response.json()["answers"]["yes"] == {"type": "noul", "noul": 0.12}
    assert response.json()["answers"]["only"]["probabilities"] == {"one": 1.0}


@pytest.mark.parametrize(
    "question",
    [
        {"type": "unknown"},
        {"criteria": {"yes": "Yes"}},
        {"type": "score", "criteria": {"low": "Low", "high": "High"}},
        {"type": "score", "criteria": ["Only one level"]},
        {"type": "score", "criteria": ["level"] * 11},
        {"type": "score", "criteria": [None, "high"]},
        {"type": "score", "criteria": ["", "high"]},
        {"type": "noul", "criteria": {"yes": "Yes", "no": "No"}},
        {"type": "noul", "criteria": ["yes", "no"]},
        {"type": "choice", "criteria": {}},
        {"type": "choice", "criteria": {"": "Empty ID"}},
        {"type": "choice", "criteria": {"yes": ""}},
        {"type": "choice", "criteria": {str(i): None for i in range(256)}},
    ],
)
def test_rejects_invalid_questions_before_inference(question):
    engine = FakeEngine()
    response = TestClient(create_app(engine)).post(
        "/v1/systemone", json={"state": "x", "questions": {"q": question}}
    )
    assert response.status_code == 422
    assert not engine.requests


def test_model_errors_capacity_and_question_limit(mixed_request):
    client = TestClient(create_app(FakeEngine()))
    assert (
        client.post("/v1/systemone", json={**mixed_request, "model": "unknown"}).status_code == 400
    )
    for model in ("default", "actlogit", "test-local-model"):
        assert (
            client.post("/v1/systemone", json={**mixed_request, "model": model}).status_code == 200
        )
    request = {"state": "x", "questions": {str(i): {"type": "noul"} for i in range(33)}}
    assert client.post("/v1/systemone", json=request).status_code == 422

    class LimitedEngine(FakeEngine):
        def predict_many(self, requests):
            raise ValueError("3 choices exceed label capacity 2")

    response = TestClient(create_app(LimitedEngine())).post("/v1/systemone", json=mixed_request)
    assert response.status_code == 422
    assert "capacity" in response.json()["detail"]


def test_only_systemone_is_exposed_and_schema_lists_all_types():
    client = TestClient(create_app(FakeEngine()))
    assert client.post("/v1/decide", json={}).status_code == 404
    assert client.get("/health").status_code == 200
    spec = client.get("/openapi.json").json()
    assert set(spec["paths"]) == {"/health", "/v1/systemone"}
    schemas = spec["components"]["schemas"]
    question = schemas["SystemOneRequest"]["properties"]["questions"]["additionalProperties"]
    assert set(question["discriminator"]["mapping"]) == {"choice", "noul", "score"}


def test_real_transformers_mixed_inference_matches_engine(config, mixed_request):
    from actlogit.engine import DecisionEngine

    engine = DecisionEngine.load(config)
    request = SystemOneRequest.model_validate(mixed_request)
    expected = engine.predict_many(
        [q.to_decision(request.state) for q in request.questions.values()]
    )
    response = TestClient(create_app(engine)).post("/v1/systemone", json=mixed_request)
    assert response.status_code == 200
    answers = response.json()["answers"]
    assert answers["team"]["probabilities"] == pytest.approx(expected[0].probabilities)
    assert answers["refund"]["noul"] == pytest.approx(expected[1].probabilities["true"])
    assert answers["urgency"]["score"] == pytest.approx(
        sum(i * expected[2].probabilities[str(i)] for i in range(3))
    )


def test_cli_and_http_share_the_same_contract(tmp_path, monkeypatch, capsys, mixed_request):
    import json
    import sys

    from actlogit import backend, cli

    config = tmp_path / "config.toml"
    config.write_text('[model]\nname_or_path="test-local-model"\n')
    request_file = tmp_path / "request.json"
    request_file.write_text(json.dumps(mixed_request))
    monkeypatch.setattr(backend, "load_engine", lambda _: FakeEngine())
    monkeypatch.setattr(
        sys, "argv", ["actlogit", "predict", "--config", str(config), "--input", str(request_file)]
    )
    cli.main()
    cli_result = json.loads(capsys.readouterr().out)
    http_result = (
        TestClient(create_app(FakeEngine())).post("/v1/systemone", json=mixed_request).json()
    )
    assert cli_result == http_result
