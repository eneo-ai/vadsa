"""Bearer keys for everything under /v1, HTTP and WebSocket."""

import hmac
from typing import Annotated

from fastapi import Depends, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.security.utils import get_authorization_scheme_param

from vadsa.errors import ApiError

_bearer = HTTPBearer(auto_error=False)


def key_accepted(token: str | None, keys: frozenset[str]) -> bool:
    """No keys means no auth; the settings allow that only in development."""
    if not keys:
        return True
    return token is not None and any(
        hmac.compare_digest(token.encode(), key.encode()) for key in keys
    )


def bearer_token(authorization: str | None) -> str | None:
    scheme, token = get_authorization_scheme_param(authorization)
    return token if scheme.lower() == "bearer" else None


def require_api_key(
    request: Request,
    credentials: Annotated[HTTPAuthorizationCredentials | None, Depends(_bearer)],
) -> None:
    token = credentials.credentials if credentials else None
    if not key_accepted(token, request.app.state.settings.api_keys):
        raise ApiError(401, "Invalid API key.", code="invalid_api_key")
