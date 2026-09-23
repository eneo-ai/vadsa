from fastapi import APIRouter, Request, Response

router = APIRouter()


@router.get(
    "/health",
    response_class=Response,
    responses={200: {"description": "The model is loaded."}, 503: {"description": "Loading."}},
)
def health(request: Request) -> Response:
    """Like vLLM: an empty 200 once the model is loaded. No auth."""
    return Response(status_code=200 if request.app.state.scheduler.ready else 503)
