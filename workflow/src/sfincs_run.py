"""
sfincs_run.py -- Shared SFINCS subprocess execution (Popen + threaded
stdout/stderr forwarding + timeout), used by every rule that executes the
SFINCS binary.
"""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path


def run_sfincs_subprocess(
    sfincs_exe: str | Path,
    cwd: str | Path,
    timeout_s: float,
    log,
    label: str = "SFINCS run",
    n_threads: int | None = None,
) -> None:
    """
    Execute the SFINCS binary in ``cwd``, streaming stdout/stderr line-by-line
    to both ``log`` and the terminal (via stderr) as it runs, with a timeout.

    Args:
        sfincs_exe: Path to the sfincs executable.
        cwd:        Working directory SFINCS runs in (its own relative file
                    references -- dep, msk, bnd, rstfile, ... -- resolve
                    against this).
        timeout_s:  Wall-clock timeout (seconds) before the process is killed.
        log:        Logger to write SFINCS's stdout (info) / stderr (warning) to.
        label:      Used only in the raised error messages (e.g.
                    "SFINCS spin-up", "SFINCS event run", "SFINCS calibration run").
        n_threads:  OpenMP thread count to give SFINCS (sets OMP_NUM_THREADS
                    for the subprocess only). SFINCS reads OMP_NUM_THREADS
                    directly (reporting it in its own startup banner) and
                    defaults to 1 thread when the variable is unset --
                    callers should pass their own ``snakemake.threads`` here
                    (paired with ``threads: workflow.cores`` on the rule, so
                    Snakemake reserves the whole machine for the run and
                    SFINCS actually uses it, instead of both blocking other
                    jobs AND running on a single core). None/0 leaves the
                    environment unchanged (whatever OMP_NUM_THREADS --
                    typically unset -- the parent process already has).

    Raises:
        RuntimeError: on timeout or non-zero exit code.
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

    # proc.wait() must run BEFORE joining the reader threads: t_out.join()/
    # t_err.join() block unconditionally until SFINCS's own stdout/stderr
    # pipes close, which only happens once it exits on its own -- so calling
    # them first makes the timeout unreachable until the process has already
    # finished, silently defeating it. Killing the process here closes its
    # pipes, which is what lets the reader threads finish and join() return.
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
