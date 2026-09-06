"""Compatibility shim (repo_cleanup.md stage 4c; delete in stage 7).

Object and friends live in src/core/objects.py.
"""
from src.core.objects import (  # noqa: F401
    object_interpolate,
    object_class_remap,
    Object,
)
