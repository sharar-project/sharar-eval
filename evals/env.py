"""Settings shared with Hibana: HIBANA_X, else the former SHARAR_X (Hibana was called Sharar), and its directories under
the former name while only those exist."""
from __future__ import annotations

import os
from pathlib import Path

PREFIXES = ("HIBANA_", "SHARAR_")


def setting(name: str, env=None) -> str | None:
    """HIBANA_<name>, else SHARAR_<name>, else None (an empty value counts as unset)."""
    env = os.environ if env is None else env
    return next((v for p in PREFIXES if (v := env.get(p + name))), None)


def former(new: Path, old: Path) -> Path:
    """`new`, or `old`, the same directory under the former name, while only that one exists (as Hibana resolves it)."""
    return old if old.exists() and not new.exists() else new
