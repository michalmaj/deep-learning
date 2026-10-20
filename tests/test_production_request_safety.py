"""Shared request-boundary tests for the module 15 inference service."""

import asyncio
import io
import sys
import threading
from pathlib import Path

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient
from PIL import Image

MODULE_DIR = Path(__file__).resolve().parents[1] / "labs" / "15_production"
sys.path.insert(0, str(MODULE_DIR))

from app.model import (  # noqa: E402
    ImageTooLargeError,
    InvalidImageError,
    decode_image_bytes,
)
from app.server import RequestBodyLimitMiddleware, run_prediction  # noqa: E402


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


def _png_bytes(size: tuple[int, int], mode: str = "L") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size).save(buffer, format="PNG")
    return buffer.getvalue()


def test_decode_image_bytes_returns_an_independent_rgb_image():
    image = decode_image_bytes(_png_bytes((3, 2)))

    assert image.mode == "RGB"
    assert image.size == (3, 2)
    assert image.getpixel((0, 0)) == (0, 0, 0)


def test_decode_image_bytes_rejects_invalid_image_data():
    with pytest.raises(InvalidImageError, match="could not decode image"):
        decode_image_bytes(b"not an image")


def test_decode_image_bytes_rejects_excessive_decoded_pixel_count():
    with pytest.raises(ImageTooLargeError, match="5-pixel limit"):
        decode_image_bytes(_png_bytes((3, 2)), max_pixels=5)


def test_run_prediction_offloads_synchronous_inference(monkeypatch):
    event_loop_thread = threading.get_ident()
    inference_thread = None

    def fake_predict(loaded, data):
        nonlocal inference_thread
        inference_thread = threading.get_ident()
        return {"loaded": loaded, "size": len(data)}

    monkeypatch.setattr("app.server.model_module.predict", fake_predict)

    result = asyncio.run(run_prediction("model", b"123"))

    assert result == {"loaded": "model", "size": 3}
    assert inference_thread is not None
    assert inference_thread != event_loop_thread


@pytest.mark.parametrize(
    ("error", "expected_status", "expected_detail"),
    [
        (InvalidImageError("bad input"), 400, "could not decode image"),
        (ImageTooLargeError("too many pixels"), 413, "decoded image too large"),
    ],
)
def test_run_prediction_maps_only_known_input_errors(
    monkeypatch, error, expected_status, expected_detail
):
    def fail(loaded, data):
        raise error

    monkeypatch.setattr("app.server.model_module.predict", fail)

    with pytest.raises(HTTPException) as exc_info:
        asyncio.run(run_prediction("model", b"data"))

    assert exc_info.value.status_code == expected_status
    assert exc_info.value.detail == expected_detail


def test_run_prediction_does_not_mask_internal_model_failures(monkeypatch):
    def fail(loaded, data):
        raise RuntimeError("internal inference failure")

    monkeypatch.setattr("app.server.model_module.predict", fail)

    with pytest.raises(RuntimeError, match="internal inference failure"):
        asyncio.run(run_prediction("model", b"data"))
