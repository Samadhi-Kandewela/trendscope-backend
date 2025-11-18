# backend/app/utils/auth.py
from datetime import datetime, timedelta
from functools import wraps
from typing import Optional

import jwt
from flask import current_app, request, jsonify

from ..extensions import bcrypt
from ..models.user import User


def hash_password(password: str) -> str:
    return bcrypt.generate_password_hash(password).decode("utf-8")


def verify_password(password: str, password_hash: str) -> bool:
    return bcrypt.check_password_hash(password_hash, password)


def generate_token(user: User, expires_in: int = 60 * 60 * 24) -> str:
    """
    Generates a JWT for the given user.
    Default expiry: 24 hours.
    """
    payload = {
        "sub": user.id,
        "email": user.email,
        "exp": datetime.utcnow() + timedelta(seconds=expires_in),
        "iat": datetime.utcnow(),
    }
    secret = current_app.config["JWT_SECRET_KEY"]
    return jwt.encode(payload, secret, algorithm="HS256")


def decode_token(token: str) -> Optional[dict]:
    secret = current_app.config["JWT_SECRET_KEY"]
    try:
        return jwt.decode(token, secret, algorithms=["HS256"])
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None


def login_required(fn):
    """
    Decorator for protected routes.
    Expects header: Authorization: Bearer <token>
    """

    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get("Authorization", "")
        if not auth_header.startswith("Bearer "):
            return jsonify({"error": "Authorization header missing"}), 401

        token = auth_header.split(" ", 1)[1]
        payload = decode_token(token)
        if not payload:
            return jsonify({"error": "Invalid or expired token"}), 401

        user = User.query.get(payload["sub"])
        if not user:
            return jsonify({"error": "User not found"}), 401

        # attach user to request context
        request.current_user = user
        return fn(*args, **kwargs)

    return wrapper
