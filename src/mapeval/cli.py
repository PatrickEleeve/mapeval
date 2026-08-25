"""Stable package entrypoint for the existing CLI."""

from __future__ import annotations

import sys
from importlib import import_module


def main() -> None:
    sys.argv[0] = "mapeval"
    import_module("mapeval.main").main()
