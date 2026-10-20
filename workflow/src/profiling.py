"""Optional line-level profiling for Snakemake scripts."""

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
        enabled = bool(snakemake.config["profiling"]["enabled"])
        self.enabled = enabled and _AVAILABLE

        if enabled and not _AVAILABLE:
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

    def wrap(self, func):
        """Return a line-profiled wrapper for func (func itself if disabled).

        Rebind the result to the same name so the profiled version is called.
        """
        if not self.enabled:
            return func
        wrapped = self._lp(func)
        return wrapped

    def stop(self):
        """Write the line-profiler report."""
        if not self.enabled:
            return

        with open(self._output, "w") as f:
            self._lp.print_stats(stream=f, output_unit=1e-3)

        print(f"[profiling] Written: {self._output}")
