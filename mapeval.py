"""Repository-local launcher for `python -m mapeval` without installation."""

from __future__ import annotations

import sys
from pathlib import Path


_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _ROOT / "src"
_PACKAGE_DIR = _SRC_DIR / "mapeval"

# Expose a package-like namespace so `import mapeval.cli` still works when the
# repository root shadows the installed distribution on sys.path.
__path__ = [str(_PACKAGE_DIR), str(_SRC_DIR)]


def main() -> None:
    src_path = str(_SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    from mapeval.cli import main as cli_main

    cli_main()


if __name__ == "__main__":
    main()
