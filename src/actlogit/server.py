from __future__ import annotations

from threading import Lock

from fastapi import FastAPI, HTTPException

from actlogit.schema import SystemOneRequest, SystemOneResponse
from actlogit.systemone import Predictor, UnknownModelError, predict


def create_app(engine: Predictor) -> FastAPI:
    app = FastAPI(title="ActLogit", version="0.1.0")
    lock = Lock()

    @app.post("/v1/systemone", response_model=SystemOneResponse)
    def system_one(request: SystemOneRequest) -> SystemOneResponse:
        try:
            with lock:
                return predict(engine, request)
        except UnknownModelError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": engine.model_id}

    return app
