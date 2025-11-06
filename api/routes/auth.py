from fastapi import APIRouter
from pydantic import BaseModel
from api.auth.jwt_utils import create_access_token, verify_token

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    user_id: str


class TokenValidationRequest(BaseModel):
    token: str


class TokenValidationResponse(BaseModel):
    valid: bool
    user_id: str | None


@router.post("/login")
async def login() -> LoginResponse:
    """
    Dummy login endpoint that generates a JWT with a unique UUID.
    No credentials required - each request creates a new user session.
    """
    # Generate a new UUID and create token
    token = create_access_token()
    print(f"Generated token: {token}")
    # Extract the UUID from the token for the response
    user_id = verify_token(token)

    return LoginResponse(access_token=token, token_type="bearer", user_id=user_id)
