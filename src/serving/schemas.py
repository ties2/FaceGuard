# validation data

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, field_validator


# Schemas for Model Version
class ModelVersionCreate(BaseModel):
    version: str
    framework: str
    model_format: str
    artifact_uri: str
    status: str = "staging"


class ModelVersionResponse(ModelVersionCreate):
    id: int
    model_id: int
    created_at: datetime

    # allo to pydantic that read data directly from SQLAlchemy
    model_config = ConfigDict(from_attributes=True)


# Schemas for Model Registry
class ModelCreate(BaseModel):
    name: str
    description: str | None = None


class ModelResponse(ModelCreate):
    id: int
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class PredictionRequest(BaseModel):
    features: list[float]

    @field_validator("features")
    @classmethod
    def check_empty_feature(cls, v):
        if not v:
            raise ValueError("Feature cannot be empty")
        return v


class PredictionResponse(BaseModel):
    model_name: str
    version: str
    prediction: Any
