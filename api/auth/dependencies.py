from typing import Annotated
import uuid
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from api.auth.jwt_utils import verify_token

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> uuid.UUID:
    """
    Dependency to validate JWT token and extract user UUID.
    Use this in protected routes to ensure the user is authenticated.

    Usage:
        @router.get("/protected")
        async def protected_route(user_id: str = Depends(get_current_user)):
            return {"user_id": user_id}
    """
    token = credentials.credentials
    user_id = verify_token(token)

    if user_id is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return uuid.UUID(user_id)


UserId = Annotated[uuid.UUID, Depends(get_current_user)]
