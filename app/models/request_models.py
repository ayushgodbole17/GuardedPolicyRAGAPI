# app/models/request_models.py
from pydantic import BaseModel, Field
from typing import Optional, Literal

class IngestTextRequest(BaseModel):
    document_name: str = Field(..., min_length=1)
    text: str = Field(..., min_length=1)

class AskRequest(BaseModel):
    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(default=None, ge=1, le=20)

