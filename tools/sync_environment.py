"""Pre-commit hook: keep environment.win-64.lock.yml in sync with the project env.

The env name is read from the lock file's `name:` line. Runs only when the lock
file is in the commit (`sync-environment-lock` hook in .pre-commit-config.yaml),
so local package drift does not block unrelated commits. Strips conda's trailing
`prefix:` line (a machine-local install path that `conda env create` would use
as its target). If the export differs, it overwrites the lock file and exits
non-zero so the file can be re-added. The portable environment.yml is edited by
hand.
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
LOCK_FILE = REPO_ROOT / "environment.win-64.lock.yml"
CONDA_BASES = ("miniforge3", "miniconda3", "anaconda3")


def env_name(env_file: Path) -> str:
    """Return the `name:` of a conda environment file."""
    match = re.search(
        r"^name:\s*(\S+)\s*$", env_file.read_text(encoding="utf-8"), re.MULTILINE
    )
    if match is None:
        raise ValueError(f"No 'name:' line in {env_file}")
    return match.group(1)


def find_conda() -> str:
    """Return the path to the conda executable; raise RuntimeError if not found."""
    conda_exe = os.environ.get("CONDA_EXE")
    if conda_exe and Path(conda_exe).exists():
        return conda_exe
    found = shutil.which("conda") or shutil.which("conda.exe")
    if found:
        return found
    # Hooks run from an IDE often have no activated conda shell (no CONDA_EXE,
    # no conda on PATH): fall back to the standard per-user install locations.
    home = Path.home()
    for base in CONDA_BASES:
        for candidate in (
            home / "AppData" / "Local" / base / "Scripts" / "conda.exe",
            home / base / "bin" / "conda",
        ):
            if candidate.exists():
                return str(candidate)
    raise RuntimeError(
        "Could not locate conda (CONDA_EXE is not set, 'conda' is not on PATH, "
        f"and no {'/'.join(CONDA_BASES)} install was found under your home "
        "directory). Set CONDA_EXE or install conda in a standard location "
        f"before committing changes to {LOCK_FILE.name}."
    )


def main() -> int:
    """Regenerate the lock file; return 1 if it changed, else 0."""
    argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    ).parse_args()
    name = env_name(LOCK_FILE)
    result = subprocess.run(
        [find_conda(), "env", "export", "-n", name],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.splitlines(keepends=True)
    if lines and lines[-1].startswith("prefix:"):
        lines = lines[:-1]
    new_content = "".join(lines)

    old_content = LOCK_FILE.read_text(encoding="utf-8")

    if new_content.strip() == old_content.strip():
        print(f"{LOCK_FILE.name} already matches the '{name}' environment.")
        return 0

    LOCK_FILE.write_text(new_content, encoding="utf-8")
    print(
        f"{LOCK_FILE.name} was stale relative to the '{name}' environment -- "
        f"regenerated it. Review the diff, `git add {LOCK_FILE.name}`, and commit "
        "again."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
