#!/usr/bin/env python3
"""Intro gate: check the bootcamper's `intro` project (Part 1) exists.

Checks, in order:

1. Root `projects.toml` registers `[projects.intro]` with `path = "intro"`.
2. `intro/warg.toml` exists and parses as TOML.
3. `intro/README.md` exists and isn't empty.

The contents beyond that are the bootcamper's business; a lead reads them
during review. Nothing here inspects or prints their contact details.

Stdlib-only; safe to run locally from the repository root:

    python3 .github/scripts/check_intro.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import tomllib


def load_toml(path: Path) -> dict | None:
    """Parse a TOML file, returning None (not raising) on any failure."""
    try:
        with path.open("rb") as file:
            return tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError):
        return None


def main() -> int:
    root = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path.cwd()
    failures = 0

    def report(ok: bool, requirement: str, hint: str) -> None:
        nonlocal failures
        if ok:
            print(f"PASS: {requirement}")
        else:
            failures += 1
            print(f"FAIL: {requirement}")
            print(f"      hint: {hint}")

    # 1. Registry entry.
    registry = load_toml(root / "projects.toml")
    entry = (registry or {}).get("projects", {}).get("intro")
    report(
        isinstance(entry, dict) and entry.get("path") == "intro",
        'projects.toml registers [projects.intro] with path = "intro"',
        "add the [projects.intro] entry shown in the root README",
    )

    # 2. Manifest.
    report(
        load_toml(root / "intro" / "warg.toml") is not None,
        "intro/warg.toml exists and parses as TOML",
        "create intro/warg.toml from the template in the root README",
    )

    # 3. README. Never read out or print its contents.
    readme_path = root / "intro" / "README.md"
    report(
        readme_path.is_file() and readme_path.stat().st_size > 0,
        "intro/README.md exists and is not empty",
        "create intro/README.md with your name, Waterloo email, and GitHub username",
    )

    if failures:
        print(f"\nintro check failed: {failures} requirement(s) not met.")
        return 1
    print("\nintro check passed: all requirements met.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
