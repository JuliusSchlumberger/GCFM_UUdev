"""Pre-commit hook: keep environment.yml in sync with the hmt_sfincs_dev conda env.

Runs only when environment.yml is in the commit (`sync-environment-yml` hook in
.pre-commit-config.yaml), so local package drift does not block unrelated
commits. Strips conda's trailing `prefix:` line (a machine-local install path
that `conda env create` would use as its target). If the export differs, it
overwrites environment.yml and exits non-zero so the file can be re-added.
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path

ENV_NAME = "hmt_sfincs_dev"
REPO_ROOT = Path(__file__).resolve().parent.parent
ENV_FILE = REPO_ROOT / "environment.yml"


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
    for base in ("miniforge3", "miniconda3", "anaconda3"):
        candidate = Path.home() / "AppData" / "Local" / base / "Scripts" / "conda.exe"
        if candidate.exists():
            return str(candidate)
    raise RuntimeError(
        "Could not locate conda.exe (CONDA_EXE is not set, 'conda' is not on "
        "PATH, and no standard miniforge3/miniconda3/anaconda3 install was "
        "found under your user profile). Set CONDA_EXE or install conda in a "
        "standard location before committing changes to environment.yml."
    )


def main() -> int:
    """Regenerate environment.yml; return 1 if it changed, else 0."""
    conda_exe = find_conda()
    result = subprocess.run(
        [conda_exe, "env", "export", "-n", ENV_NAME],
        capture_output=True,
        text=True,
        check=True,
    )
    lines = result.stdout.splitlines(keepends=True)
    if lines and lines[-1].startswith("prefix:"):
        lines = lines[:-1]
    new_content = "".join(lines)

    old_content = ENV_FILE.read_text(encoding="utf-8") if ENV_FILE.exists() else ""

    if new_content.strip() == old_content.strip():
        print(f"{ENV_FILE.name} already matches the '{ENV_NAME}' environment.")
        return 0

    ENV_FILE.write_text(new_content, encoding="utf-8")
    print(
        f"{ENV_FILE.name} was stale relative to the '{ENV_NAME}' environment -- "
        "regenerated it. Review the diff, `git add environment.yml`, and commit again."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
