"""Shared request-boundary tests for the module 15 inference service."""

import sys
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

MODULE_DIR = Path(__file__).resolve().parents[1] / "labs" / "15_production"
sys.path.insert(0, str(MODULE_DIR))

from app.server import RequestBodyLimitMiddleware  # noqa: E402


def _body_echo_app(*, limit: int) -> tuple[FastAPI, dict[str, bool]]:
    app = FastAPI()
    called = {"predict": False, "other": False}

    @app.post("/predict")
    async def predict(request: Request):
        called["predict"] = True
        return {"size": len(await request.body())}

    @app.post("/other")
    async def other(request: Request):
        called["other"] = True
        return {"size": len(await request.body())}

    app.add_middleware(RequestBodyLimitMiddleware, max_body_bytes=limit, path="/predict")
    return app, called


def test_request_body_limit_rejects_from_content_length_before_endpoint_runs():
    app, called = _body_echo_app(limit=4)

    with TestClient(app) as client:
        response = client.post("/predict", content=b"12345")

    assert response.status_code == 413
    assert response.json() == {"detail": "request body too large"}
    assert called["predict"] is False


def test_request_body_limit_counts_streamed_chunks_without_content_length():
    app, called = _body_echo_app(limit=4)

    def chunks():
        yield b"12"
        yield b"345"

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            content=chunks(),
            headers={"transfer-encoding": "chunked"},
        )

    assert response.status_code == 413
    assert response.json() == {"detail": "request body too large"}
    assert called["predict"] is True


def test_request_body_limit_does_not_affect_other_routes():
    app, called = _body_echo_app(limit=4)

    with TestClient(app) as client:
        response = client.post("/other", content=b"12345")

    assert response.status_code == 200
    assert response.json() == {"size": 5}
    assert called["other"] is True
