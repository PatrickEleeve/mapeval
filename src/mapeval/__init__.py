"""MAPEval package wrapper."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


_PACKAGE_DIR = Path(__file__).resolve().parent
_LEGACY_MODULE_DIR = _PACKAGE_DIR.parent
if str(_LEGACY_MODULE_DIR) not in __path__:
    __path__.append(str(_LEGACY_MODULE_DIR))


try:
    __version__ = version("mapeval")
except PackageNotFoundError:
    __version__ = "0.2.0"
