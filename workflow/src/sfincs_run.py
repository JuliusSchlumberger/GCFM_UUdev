"""Run the SFINCS binary as a subprocess and hand-write sfincs.inp files.

Used by every rule that executes SFINCS. sfincs.inp files that borrow geometry
from another model directory (13_build_sfincs.py, 14_run_spinup.py) are written
by hand because HydroMT's get_set_file_variable absolutizes any file reference
outside the model root instead of keeping a relative ../ path.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path


def parse_sfincs_inp(path: str | Path) -> dict[str, str]:
    """Parse a sfincs.inp file into a lowercased {key: value} dict.

    The format is plain ``key = value`` lines; comments start with ``!``.
    """
    cfg: dict[str, str] = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if "=" in line and not line.startswith("!"):
                key, _, val = line.partition("=")
                cfg[key.strip().lower()] = val.strip()
    return cfg


def set_sfincs_inp_keys(path: str | Path, updates: dict[str, object | None]) -> None:
    """Set keys of an existing sfincs.inp in place; a value of None removes the key.

    Missing keys are appended. Avoids hydromt: sf.write() rewrites every grid
    and subgrid file; sf.config.write() absolutizes relative paths.
    """
    path = Path(path)
    lines = path.read_text().splitlines()
    todo = {k.lower(): v for k, v in updates.items()}
    out = []
    for line in lines:
        key = line.partition("=")[0].strip().lower() if "=" in line else None
        if key in todo:
            value = todo.pop(key)
            if value is not None:
                out.append(f"{key:<20} = {value}")
            continue
        out.append(line)
    for key, value in todo.items():
        if value is not None:
            out.append(f"{key:<20} = {value}")
    path.write_text("\n".join(out) + "\n")


def forward_geometry_files(
    cfg: dict[str, str],
    source_root: str | Path,
    dest_root: str | Path,
    exclude: frozenset[str] = frozenset(),
) -> list[str]:
    """Build ``"key = <relative path>"`` lines forwarding ``cfg``'s ``*file`` entries.

    ``cfg`` is ``source_root``'s parsed sfincs.inp; paths are made relative to
    ``dest_root`` with os.path.relpath. Skips keys in ``exclude`` (typically
    {"rstfile"}) and entries whose target is missing or a 0-byte placeholder
    (e.g. sfincs_subgrid.nc/sfincs.weir when that feature is disabled).
    """
    source_root = Path(source_root)
    dest_root = Path(dest_root)
    lines = []
    for key, value in cfg.items():
        if not key.endswith("file") or key in exclude:
            continue
        fpath = source_root / value
        if fpath.exists() and fpath.stat().st_size > 0:
            rel = os.path.relpath(fpath, start=dest_root)
            lines.append(f"{key:<20} = {rel}")
    return lines


# Crest (and par1) format of sfincs.weir: 1 cm, the precision every crest is
# rounded to (src.surge.ceil_crest).
WEIR_Z_FMT = "%.2f"


def write_weir_file(sf) -> Path | None:
    """Rewrite ``sf``'s sfincs.weir with crests at 1 cm instead of hydromt's 0.1 m.

    ``sf.weirs.write()`` calls ``hydromt_sfincs.utils.write_geoms`` without
    ``fmt_z`` (default "%.1f", round to nearest) and offers no way to change it.
    SFINCS honours crests to the millimetre (one weir in a flat basin: crests of
    0.430 and 0.437 m start overtopping 7 mm of water level apart). Call AFTER
    ``sf.write()`` / ``sf.weirs.write()``; overwrites their file with the same
    writer and x/y format. Returns None when the model has no weirs.
    """
    from hydromt_sfincs import utils

    if sf.weirs.data is None or sf.weirs.data.empty:
        return None
    path = sf.config.get_set_file_variable(
        key="weirfile", value=None, default="sfincs.weir"
    )
    fmt_xy = "%11.6f" if sf.crs.is_geographic else "%11.1f"
    utils.write_geoms(
        path,
        utils.gdf2linestring(sf.weirs.data),
        stype="weir",
        fmt=fmt_xy,
        fmt_z=WEIR_Z_FMT,
    )
    return Path(path)


def run_sfincs_subprocess(
    sfincs_exe: str | Path,
    cwd: str | Path,
    timeout_s: float,
    log,
    label: str = "SFINCS run",
    n_threads: int | None = None,
) -> None:
    """Run the SFINCS binary in ``cwd`` with a wall-clock timeout, streaming its output.

    stdout/stderr go line by line to ``log`` (info/warning) and the terminal.
    ``cwd`` is where SFINCS resolves its relative file references. ``label`` is
    used only in error messages. ``n_threads`` sets OMP_NUM_THREADS for the
    subprocess only; SFINCS defaults to 1 thread when it is unset, so pass
    ``snakemake.threads`` (with ``threads: workflow.cores`` on the rule).
    None/0 leaves the environment unchanged.
    Raises RuntimeError on timeout or non-zero exit code.
    """
    sfincs_exe = Path(sfincs_exe).resolve()
    log.info(f"Running {label}: {sfincs_exe}")

    env = None
    if n_threads:
        env = os.environ.copy()
        env["OMP_NUM_THREADS"] = str(int(n_threads))
        log.info(f"{label}: OMP_NUM_THREADS={n_threads}")

    proc = subprocess.Popen(
        [str(sfincs_exe)],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # line-buffered
        env=env,
    )

    def _forward(pipe, log_fn):
        for line in pipe:
            line = line.rstrip()
            if line:
                log_fn(f"[sfincs] {line}")
                print(f"[sfincs] {line}", file=sys.stderr, flush=True)

    t_out = threading.Thread(target=_forward, args=(proc.stdout, log.info))
    t_err = threading.Thread(target=_forward, args=(proc.stderr, log.warning))
    t_out.start()
    t_err.start()

    # Wait before joining the reader threads: join() blocks until SFINCS closes
    # its pipes, which would make the timeout unreachable. Killing the process
    # closes its pipes, so the threads finish and join() returns.
    try:
        proc.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        proc.kill()
        t_out.join()
        t_err.join()
        raise RuntimeError(f"{label} exceeded {timeout_s}s timeout")

    t_out.join()
    t_err.join()

    if proc.returncode != 0:
        raise RuntimeError(f"{label} failed with exit code {proc.returncode}")
