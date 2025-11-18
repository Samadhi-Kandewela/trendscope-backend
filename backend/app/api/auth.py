# backend/app/api/auth.py
from flask import Blueprint, request, jsonify

from ..models.user import User
from ..extensions import db
from ..utils.auth import (
    hash_password,
    verify_password,
    generate_token,
    login_required,
)

auth_bp = Blueprint("auth", __name__)


@auth_bp.post("/signup")
def signup():
    """
    POST /api/auth/signup
    Body: { "full_name": "...", "email": "...", "password": "..." }
    """
    data = request.get_json() or {}
    full_name = (data.get("full_name") or "").strip()
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not full_name or not email or not password:
        return jsonify({"error": "full_name, email and password are required"}), 400

    if len(password) < 8:
        return jsonify({"error": "Password must be at least 8 characters"}), 400

    # Check if user exists
    existing = User.query.filter_by(email=email).first()
    if existing:
        return jsonify({"error": "Email is already registered"}), 409

    user = User(
        full_name=full_name,
        email=email,
        password_hash=hash_password(password),
    )
    db.session.add(user)
    db.session.commit()

    token = generate_token(user)

    return jsonify({"user": user.to_dict(), "token": token}), 201


@auth_bp.post("/login")
def login():
    """
    POST /api/auth/login
    Body: { "email": "...", "password": "..." }
    """
    data = request.get_json() or {}
    email = (data.get("email") or "").strip().lower()
    password = data.get("password") or ""

    if not email or not password:
        return jsonify({"error": "email and password are required"}), 400

    user = User.query.filter_by(email=email).first()
    if not user or not verify_password(password, user.password_hash):
        return jsonify({"error": "Invalid email or password"}), 401

    token = generate_token(user)

    return jsonify({"user": user.to_dict(), "token": token}), 200


@auth_bp.get("/me")
@login_required
def me():
    """
    GET /api/auth/me
    Requires Authorization: Bearer <token>
    """
    user = getattr(request, "current_user", None)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    return jsonify({"user": user.to_dict()}), 200


# ─────────────────────────────────────────
# NEW: Update profile (change full_name)
# ─────────────────────────────────────────
@auth_bp.put("/profile")
@login_required
def update_profile():
    """
    PUT /api/auth/profile
    Body: { "full_name": "New Name" }
    Requires Authorization: Bearer <token>
    """
    user = getattr(request, "current_user", None)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}
    full_name = (data.get("full_name") or "").strip()

    if not full_name:
        return jsonify({"error": "full_name is required"}), 400

    user.full_name = full_name
    db.session.commit()

    return jsonify({"user": user.to_dict()}), 200


# ─────────────────────────────────────────
# NEW: Change password
# ─────────────────────────────────────────
@auth_bp.post("/change-password")
@login_required
def change_password():
    """
    POST /api/auth/change-password
    Body: {
      "current_password": "...",
      "new_password": "...",
      "confirm_password": "..."
    }
    Requires Authorization: Bearer <token>
    """
    user = getattr(request, "current_user", None)
    if not user:
        return jsonify({"error": "Unauthorized"}), 401

    data = request.get_json() or {}

    current_password = data.get("current_password") or ""
    new_password = data.get("new_password") or ""
    confirm_password = data.get("confirm_password") or ""

    if not current_password or not new_password or not confirm_password:
        return jsonify({"error": "All password fields are required"}), 400

    if new_password != confirm_password:
        return jsonify({"error": "New password and confirm password do not match"}), 400

    if len(new_password) < 8:
        return jsonify({"error": "New password must be at least 8 characters"}), 400

    # Verify current password
    if not verify_password(current_password, user.password_hash):
        return jsonify({"error": "Current password is incorrect"}), 401

    # Update password hash
    user.password_hash = hash_password(new_password)
    db.session.commit()

    return jsonify({"message": "Password updated successfully"}), 200
