import time
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from vadsa.auth import require_api_key
from vadsa.errors import ErrorResponse

router = APIRouter()


class ModelCard(BaseModel):
    id: str
    object: Literal["model"] = "model"
    created: int
    owned_by: Literal["vadsa"] = "vadsa"
    # what Eneo reads to list this as a transcription model
    model_type: Literal["transcription"] = "transcription"


class ModelList(BaseModel):
    object: Literal["list"] = "list"
    data: list[ModelCard]


@router.get(
    "/v1/models",
    dependencies=[Depends(require_api_key)],
    responses={401: {"model": ErrorResponse}},
)
def list_models(request: Request) -> ModelList:
    name = request.app.state.settings.served_model_name
    return ModelList(data=[ModelCard(id=name, created=int(time.time()))])
