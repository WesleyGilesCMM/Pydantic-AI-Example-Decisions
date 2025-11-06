import os
import jwt
import uuid
from datetime import datetime, timedelta
from typing import Optional

from api.utils import utc_now

# Secret key for JWT - in production, this should be in environment variables
SECRET_KEY = os.getenv("SECRET_KEY")
ALGORITHM = os.getenv("CRYPTO_ALGORITHM")
ACCESS_TOKEN_EXPIRE_MINUTES = 60 * 24 * 365  # 1 year


def create_access_token(user_uuid: Optional[str] = None) -> str:
    """
    Create a JWT access token with a unique UUID.

    Args:
        user_uuid: Optional UUID string. If not provided, a new UUID will be generated.

    Returns:
        Encoded JWT token string
    """
    if user_uuid is None:
        user_uuid = str(uuid.uuid4())

    expire = utc_now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)

    to_encode = {"sub": user_uuid, "exp": expire.timestamp(), "iat": utc_now().timestamp()}
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt


def verify_token(token: str) -> Optional[str]:
    """
    Verify a JWT token and return the user UUID.

    Args:
        token: JWT token string to verify

    Returns:
        User UUID if token is valid, None otherwise
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_uuid: str = payload.get("sub")
        return user_uuid
    except jwt.InvalidTokenError:
        return None
