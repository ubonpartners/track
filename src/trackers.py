"""Compatibility shim (repo_cleanup.md stage 4c; delete in stage 7).

The tracker factory lives in src/tracker/factory.py.
"""
from src.tracker.factory import (  # noqa: F401
    create_tracker,
)
