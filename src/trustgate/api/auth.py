"""Role-based API Key Authentication for TrustGate REST API."""

import os
from collections.abc import Callable
from enum import StrEnum
from typing import Annotated

from fastapi import Depends, HTTPException, Security, status
from fastapi.security import APIKeyHeader
from pydantic import BaseModel

API_KEY_HEADER_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_HEADER_NAME, auto_error=False)


class Role(StrEnum):
    """User access roles."""

    ADMIN = "admin"
    OPERATOR = "operator"
    VIEWER = "viewer"


class AuthenticatedUser(BaseModel):
    """Authenticated user/service context."""

    api_key: str
    role: Role
    description: str


# Role permission hierarchy: higher numbers inherit lower-level permissions
ROLE_HIERARCHY: dict[Role, int] = {
    Role.VIEWER: 1,
    Role.OPERATOR: 2,
    Role.ADMIN: 3,
}

# Built-in developer API keys for out-of-the-box operation
DEFAULT_API_KEYS: dict[str, tuple[Role, str]] = {
    "tg-admin-secret-key": (Role.ADMIN, "Default Administrator"),
    "tg-operator-secret-key": (Role.OPERATOR, "Default Operator"),
    "tg-viewer-secret-key": (Role.VIEWER, "Default Viewer / Read-only"),
}


def get_api_key_registry() -> dict[str, tuple[Role, str]]:
    """Retrieve active API keys merged from defaults and environment."""
    registry = dict(DEFAULT_API_KEYS)
    env_keys = os.getenv("TRUSTGATE_API_KEYS", "")
    if env_keys:
        for entry in env_keys.split(";"):
            entry = entry.strip()
            if not entry:
                continue
            parts = entry.split(":")
            if len(parts) >= 2:
                key = parts[0].strip()
                role_str = parts[1].strip().lower()
                desc = parts[2].strip() if len(parts) > 2 else "Configured API Key"
                try:
                    registry[key] = (Role(role_str), desc)
                except ValueError:
                    pass
    return registry


def get_current_user(
    api_key: Annotated[str | None, Security(api_key_header)],
) -> AuthenticatedUser:
    """Validate API key from header and return authenticated user."""
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing 'X-API-Key' authentication header.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    registry = get_api_key_registry()
    if api_key not in registry:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API Key provided.",
            headers={"WWW-Authenticate": "ApiKey"},
        )

    role, desc = registry[api_key]
    return AuthenticatedUser(api_key=api_key, role=role, description=desc)


def require_role(min_role: Role) -> Callable[[AuthenticatedUser], AuthenticatedUser]:
    """Enforce minimum required role hierarchy on an endpoint."""

    def role_checker(
        current_user: Annotated[AuthenticatedUser, Depends(get_current_user)],
    ) -> AuthenticatedUser:
        user_rank = ROLE_HIERARCHY.get(current_user.role, 0)
        required_rank = ROLE_HIERARCHY.get(min_role, 0)
        if user_rank < required_rank:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Operation requires minimum role '{min_role.value}', but caller has '{current_user.role.value}'.",
            )
        return current_user

    return role_checker
