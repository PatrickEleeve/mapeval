"""Stable package entrypoint for the existing CLI."""

from __future__ import annotations

from importlib import import_module
import sys


def main() -> None:
    sys.argv[0] = "mapeval"
    import_module("mapeval.main").main()
