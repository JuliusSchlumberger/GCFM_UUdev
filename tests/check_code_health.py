"""Check workflow/scripts, workflow/src and tests/*.py for syntax and src imports.

Catches broken `from src.X import Y` (X not importable, Y missing), which
neither ast.parse nor `snakemake -n` reports. No dry-run: that needs every
catalogued raw file on disk. Exit code 0 if all checks pass, 1 otherwise.

Usage: python tests/check_code_health.py
"""

import ast
import glob
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "workflow"))


def check_syntax(files: list[str]) -> list[str]:
    """Return one message per file that does not parse."""
    errors = []
    for f in files:
        with open(f, encoding="utf-8") as fh:
            try:
                ast.parse(fh.read(), filename=f)
            except SyntaxError as e:
                errors.append(f"{f}: syntax error: {e}")
    return errors


def check_src_imports(files: list[str]) -> list[str]:
    """Return one message per unresolvable `from src... import name`."""
    errors = []
    module_cache: dict[str, object] = {}
    for f in files:
        with open(f, encoding="utf-8") as fh:
            tree = ast.parse(fh.read(), filename=f)
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.ImportFrom)
                and node.module
                and node.module.startswith("src")
            ):
                continue
            mod_name = node.module
            if mod_name not in module_cache:
                try:
                    module_cache[mod_name] = importlib.import_module(mod_name)
                except Exception as e:
                    module_cache[mod_name] = e
            mod = module_cache[mod_name]
            if isinstance(mod, Exception):
                errors.append(f"{f}: failed to import module '{mod_name}': {mod}")
                continue
            for alias in node.names:
                if alias.name != "*" and not hasattr(mod, alias.name):
                    errors.append(
                        f"{f}: '{alias.name}' not found in module '{mod_name}'"
                    )
    return errors


def main() -> int:
    """Run both checks and return the exit code."""
    files = (
        glob.glob(str(REPO_ROOT / "workflow/scripts/*.py"))
        + glob.glob(str(REPO_ROOT / "workflow/src/*.py"))
        + glob.glob(str(REPO_ROOT / "tests/*.py"))
    )

    errors = check_syntax(files) + check_src_imports(files)

    if errors:
        print(f"FAILED -- {len(errors)} issue(s) across {len(files)} files:\n")
        for e in errors:
            print(f"  {e}")
        return 1

    print(f"OK -- {len(files)} files, no syntax or src-import errors")
    return 0


if __name__ == "__main__":
    sys.exit(main())
