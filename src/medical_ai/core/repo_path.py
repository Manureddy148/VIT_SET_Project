"""Resolve repository root for imports (e.g. root-level `imaging_extension.py`)."""

from __future__ import annotations

import sys
from pathlib import Path


def repo_root() -> Path:
    # src/medical_ai/core/repo_path.py -> parents[3] == repo root
    return Path(__file__).resolve().parents[3]


def ensure_repo_on_path() -> Path:
    root = repo_root()
    s = str(root)
    if s not in sys.path:
        sys.path.insert(0, s)
    return root
