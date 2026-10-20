"""Optional line-level and wall-clock profiling for Snakemake scripts."""

import time
from pathlib import Path

try:
    from line_profiler import LineProfiler as _LP

    _AVAILABLE = True
except ImportError:
    _AVAILABLE = False


class ScriptProfiler:
    """Conditional line-level profiler for Snakemake scripts.

    Reads config["profiling"]["enabled"]. When True, wraps registered
    functions with LineProfiler and writes a timing report next to the
    script's log file (same name, suffix _profile.txt). When disabled, or
    line_profiler is missing, wrap() returns the function unchanged.
    """

    def __init__(self, snakemake):
        cfg = snakemake.config.get("profiling", {})
        self.enabled = bool(cfg.get("enabled", False)) and _AVAILABLE

        if bool(cfg.get("enabled", False)) and not _AVAILABLE:
            import warnings

            warnings.warn(
                "profiling.enabled=true but line_profiler is not installed. "
                "Run: pip install line_profiler",
                stacklevel=2,
            )

        if self.enabled:
            log_path = Path(snakemake.log[0])
            self._output = log_path.with_name("line_profiler_" + log_path.stem + ".txt")
            self._lp = _LP()
        else:
            self._lp = None
            self._output = None

        self._wall_times: dict[str, float] = {}
        self._wall_start: dict[str, float] = {}

    def wrap(self, func):
        """Return a line-profiled wrapper for func (func itself if disabled).

        Rebind the result to the same name so the profiled version is called.
        """
        if not self.enabled:
            return func
        wrapped = self._lp(func)
        return wrapped

    def time(self, label: str):
        """Return a context manager that adds the block's wall-clock time to label."""
        return _WallTimer(self, label)

    def stop(self):
        """Write the line-profiler report and wall-clock summary."""
        if not self.enabled:
            return

        with open(self._output, "w") as f:
            self._lp.print_stats(stream=f, output_unit=1e-3)

            if self._wall_times:
                f.write("\n\n" + "=" * 60 + "\n")
                f.write("WALL-CLOCK TIMINGS (profiler.time() blocks)\n")
                f.write("=" * 60 + "\n")
                total = sum(self._wall_times.values())
                for label, elapsed in sorted(
                    self._wall_times.items(), key=lambda kv: -kv[1]
                ):
                    pct = 100 * elapsed / total if total > 0 else 0
                    f.write(f"  {elapsed:9.3f} s  {pct:5.1f}%  {label}\n")
                f.write(f"  {'─' * 9}         {'─' * 5}\n")
                f.write(f"  {total:9.3f} s  100.0%  TOTAL\n")

        print(f"[profiling] Written: {self._output}")


class _WallTimer:
    def __init__(self, profiler: ScriptProfiler, label: str):
        self._p = profiler
        self._label = label

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._t0
        self._p._wall_times[self._label] = (
            self._p._wall_times.get(self._label, 0.0) + elapsed
        )
