"""Shared runtime building blocks for Toori."""

from .app import create_app
from .service import RuntimeContainer

__all__ = ["create_app", "RuntimeContainer"]
