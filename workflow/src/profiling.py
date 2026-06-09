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
    script's log file (same name, suffix _profile.txt).

    Usage
    -----
    profiler = ScriptProfiler(snakemake)

    # Wrap each call you want to profile — the returned callable is the
    # wrapped version; line_profiler sees its internal lines.
    load_glofas_clip = profiler.wrap(load_glofas_clip)
    find_boundary_crossings = profiler.wrap(find_boundary_crossings)

    # ... rest of script unchanged; wrapped names shadow the originals ...

    profiler.stop()

    # When disabled, wrap() returns the original function unchanged — zero overhead.
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

    # ── primary API ──────────────────────────────────────────────────────────────

    def wrap(self, func):
        """Return a line-profiled wrapper for *func* (or *func* unchanged if disabled).

        Assign the return value back to the same name so the profiled
        version is what actually gets called:

            load_glofas_clip = profiler.wrap(load_glofas_clip)
        """
        if not self.enabled:
            return func
        wrapped = self._lp(func)
        return wrapped

    def time(self, label: str):
        """Context manager that records wall-clock time for an arbitrary block.

        Usage::

            with profiler.time("gpd.read_file"):
                land = gpd.read_file(path, bbox=bounds)
        """
        return _WallTimer(self, label)

    def stop(self):
        """Write the line-profiler report and wall-clock summary."""
        if not self.enabled:
            return

        with open(self._output, "w") as f:
            # line_profiler section
            self._lp.print_stats(stream=f, output_unit=1e-3)

            # wall-clock section
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
