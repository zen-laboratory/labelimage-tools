from __future__ import annotations

from importlib import import_module
from typing import Any


def optional_import(
    module_name: str,
    *,
    extra: str,
    feature: str,
    package_name: str | None = None,
) -> Any:
    """Import an optional dependency or raise a clear installation hint."""
    display_name = package_name or module_name
    try:
        return import_module(module_name)
    except ImportError as exc:
        install_hint = f"`pip install labelimage-tools[{extra}]`"
        if extra != "all":
            install_hint += " or `pip install labelimage-tools[all]`"
        raise ImportError(
            f"{feature} requires the optional dependency `{display_name}`. "
            f"Install it with {install_hint} "
            f"or install `{display_name}` directly."
        ) from exc
