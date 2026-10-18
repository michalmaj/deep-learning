"""FastAPI inference service.

`/health`, `/ready`, `/model-info` are given (scaffold). `/predict` is
your task -- see README.md section 3.
"""

import os

from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: F401 - uzywane w TODO ponizej

from . import model as model_module
from .schemas import HealthResponse, ModelInfoResponse, PredictionResponse, ReadyResponse

MAX_UPLOAD_BYTES = 5_000_000
ARTIFACT_DIR = os.environ.get("MODULE15_ARTIFACT_DIR", "artifacts")

app = FastAPI(title="Module 15 inference service")
app.state.loaded = None


@app.on_event("startup")
def _startup():
    # Model is loaded ONCE here, at process start -- never inside a request
    # handler, and never retrained here.
    app.state.loaded = model_module.load_artifact(ARTIFACT_DIR)


@app.get("/health", response_model=HealthResponse)
def health():
    return {"status": "ok"}


@app.get("/ready", response_model=ReadyResponse)
def ready():
    return {"ready": app.state.loaded is not None}


@app.get("/model-info", response_model=ModelInfoResponse)
def model_info():
    loaded = app.state.loaded
    return {
        "model_version": loaded.model_version,
        "class_names": loaded.class_names,
        "input_size": loaded.input_size,
    }


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(file: UploadFile = File(...)):
    # TODO:
    # 1. if app.state.loaded is None: raise HTTPException(503, "model not ready")
    # 2. data = await file.read()
    # 3. validate size: reject empty files (400) AND files over MAX_UPLOAD_BYTES (413)
    # 4. call model_module.predict(app.state.loaded, data) inside a try/except;
    #    on ANY exception, raise HTTPException(400, "could not decode or process
    #    image") -- NEVER let the raw exception/traceback reach the client
    # 5. return the result dict
    raise NotImplementedError("TODO: /predict endpoint")
