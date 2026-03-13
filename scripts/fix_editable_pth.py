"""Repair editable-install `.pth` files on macOS.

Python's site loader skips hidden `.pth` files. Some local editable installs on
macOS end up with a hidden `__editable__.*.pth`, which breaks console scripts
outside the repository root.
"""

from __future__ import annotations

import os
import site
import stat
from pathlib import Path


def repair_editable_pth_files() -> tuple[int, int]:
    cleared = 0
    fallback_written = 0
    for site_dir in site.getsitepackages():
        site_path = Path(site_dir)
        for pth_file in site_path.glob("__editable__.*.pth"):
            file_stat = pth_file.stat()
            hidden_flag = getattr(stat, "UF_HIDDEN", 0)
            if hidden_flag and getattr(file_stat, "st_flags", 0) & hidden_flag:
                os.chflags(pth_file, file_stat.st_flags & ~hidden_flag)
                cleared += 1

            # Write a non-hidden fallback .pth so console scripts still work
            # even if macOS re-applies the hidden flag to __editable__ files.
            fallback = site_path / "mapeval-editable-fallback.pth"
            fallback.write_text(pth_file.read_text())
            fallback_stat = fallback.stat()
            if hidden_flag and getattr(fallback_stat, "st_flags", 0) & hidden_flag:
                os.chflags(fallback, fallback_stat.st_flags & ~hidden_flag)
            fallback_written += 1

    return cleared, fallback_written


def main() -> None:
    cleared, fallback_written = repair_editable_pth_files()
    print(
        "Editable install repair:"
        f" cleared_hidden={cleared}"
        f" fallback_written={fallback_written}"
    )


if __name__ == "__main__":
    main()
