"""FastAPI inference service.

`/health`, `/ready`, `/model-info` are given (scaffold). `/predict` is
your task -- see README.md section 3.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile

from . import model as model_module
from .schemas import HealthResponse, ModelInfoResponse, PredictionResponse, ReadyResponse

MAX_UPLOAD_BYTES = 5_000_000
ARTIFACT_DIR = os.environ.get("MODULE15_ARTIFACT_DIR", "artifacts")


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


async def read_upload_limited(
    file: UploadFile, max_bytes: int = MAX_UPLOAD_BYTES
) -> bytes:
    """Read at most one byte beyond the limit so oversized uploads stay bounded."""
    data = await file.read(max_bytes + 1)
    if not data:
        raise HTTPException(status_code=400, detail="empty file")
    if len(data) > max_bytes:
        raise HTTPException(status_code=413, detail="file too large")
    return data


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    # TODO:
    # 1. if app.state.loaded is None: raise HTTPException(503, "model not ready")
    # 2. data = await read_upload_limited(file) -- this helper rejects empty files
    #    (400) and files over MAX_UPLOAD_BYTES (413) without reading an unbounded body
    # 3. call model_module.predict(app.state.loaded, data) inside a try/except;
    #    on ANY exception, raise HTTPException(400, "could not decode or process
    #    image") -- NEVER let the raw exception/traceback reach the client
    # 4. return the result dict
    raise NotImplementedError("TODO: /predict endpoint")
