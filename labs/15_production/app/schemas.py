"""Response schemas for the module 15 inference service."""

from pydantic import BaseModel


class PredictionResponse(BaseModel):
    predicted_class: str
    confidence: float
    model_version: str


class HealthResponse(BaseModel):
    status: str


class ReadyResponse(BaseModel):
    ready: bool


class ModelInfoResponse(BaseModel):
    model_version: str
    class_names: list[str]
    input_size: list[int]
