"""FastAPI inference service.

`/health`, `/ready`, `/model-info` are given (scaffold). `/predict` is
your task -- see README.md section 3.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool
from starlette.responses import JSONResponse
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from . import model as model_module
from .schemas import HealthResponse, ModelInfoResponse, PredictionResponse, ReadyResponse

MAX_UPLOAD_BYTES = 5_000_000
MAX_MULTIPART_OVERHEAD_BYTES = 64 * 1024
MAX_REQUEST_BYTES = MAX_UPLOAD_BYTES + MAX_MULTIPART_OVERHEAD_BYTES
ARTIFACT_DIR = os.environ.get("MODULE15_ARTIFACT_DIR", "artifacts")


class RequestBodyTooLarge(Exception):
    """Internal signal raised before multipart parsing can finish."""


class RequestBodyLimitMiddleware:
    """Reject an oversized request while ASGI body chunks are still arriving.

    Limiting ``UploadFile.read()`` inside an endpoint is too late to protect
    the service: Starlette has already parsed the multipart body and spooled
    the uploaded file by then. This middleware wraps ``receive`` so the limit
    applies before the request reaches the multipart parser.
    """

    def __init__(
        self,
        app: ASGIApp,
        *,
        max_body_bytes: int,
        path: str = "/predict",
    ) -> None:
        self.app = app
        self.max_body_bytes = max_body_bytes
        self.path = path

    async def _reject(self, scope: Scope, receive: Receive, send: Send) -> None:
        response = JSONResponse(status_code=413, content={"detail": "request body too large"})
        await response(scope, receive, send)

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if (
            scope["type"] != "http"
            or scope.get("method") != "POST"
            or scope.get("path") != self.path
        ):
            await self.app(scope, receive, send)
            return

        headers = {key.lower(): value for key, value in scope.get("headers", [])}
        content_length = headers.get(b"content-length")
        if content_length is not None:
            try:
                if int(content_length) > self.max_body_bytes:
                    await self._reject(scope, receive, send)
                    return
            except ValueError:
                # The HTTP server normally rejects malformed Content-Length.
                # If one reaches ASGI, the streaming counter below remains the
                # authoritative limit.
                pass

        received = 0
        response_started = False

        async def receive_limited() -> Message:
            nonlocal received
            message = await receive()
            if message["type"] == "http.request":
                received += len(message.get("body", b""))
                if received > self.max_body_bytes:
                    raise RequestBodyTooLarge
            return message

        async def track_response(message: Message) -> None:
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await send(message)

        try:
            await self.app(scope, receive_limited, track_response)
        except RequestBodyTooLarge:
            if response_started:
                raise
            await self._reject(scope, receive, send)


@asynccontextmanager
async def lifespan(application: FastAPI):
    # Model is loaded ONCE here, at process start -- never inside a request
    # handler, and never retrained here.
    application.state.loaded = model_module.load_artifact(ARTIFACT_DIR)
    try:
        yield
    finally:
        application.state.loaded = None


app = FastAPI(title="Module 15 inference service", lifespan=lifespan)
app.add_middleware(
    RequestBodyLimitMiddleware,
    max_body_bytes=MAX_REQUEST_BYTES,
    path="/predict",
)
app.state.loaded = None


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get("/ready", response_model=ReadyResponse)
def ready():
    return {"ready": app.state.loaded is not None}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    loaded = app.state.loaded
    if loaded is None:
        raise HTTPException(status_code=503, detail="model not ready")
    return {
        "model_version": loaded.model_version,
        "class_names": loaded.class_names,
        "input_size": loaded.input_size,
    }


async def read_upload_limited(file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES) -> bytes:
    """Read at most one byte beyond the limit so oversized uploads stay bounded."""
    data = await file.read(max_bytes + 1)
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="file too large")
    return data


async def run_prediction(loaded, data: bytes) -> dict:
    """Run CPU inference off the event loop and map only input errors to 4xx."""
    try:
        return await run_in_threadpool(model_module.predict, loaded, data)
    except model_module.ImageTooLargeError:
        raise HTTPException(status_code=413, detail="decoded image too large") from None
    except model_module.InvalidImageError:
        raise HTTPException(status_code=400, detail="could not decode image") from None


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    # TODO:
    # 1. if app.state.loaded is None: raise HTTPException(503, "model not ready")
    # 2. data = await read_upload_limited(file) -- this helper rejects empty files
    #    (400) and files over MAX_UPLOAD_BYTES (413) without reading an unbounded body
    # 3. return await run_prediction(app.state.loaded, data). The helper runs
    #    synchronous CPU inference outside the event loop, maps known input
    #    errors to safe 400/413 responses, and deliberately leaves unexpected
    #    server/model failures as HTTP 500 instead of mislabeling them as 400.
    raise NotImplementedError("TODO: /predict endpoint")
