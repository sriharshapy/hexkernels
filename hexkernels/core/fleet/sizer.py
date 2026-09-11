"""Sim-fleet sizer: probe how many parallel sim workers this host can sustain.

``probe_fleet_size`` runs a reference kernel (the default task's own
``baseline.c``) at increasing widths and finds the knee where adding more
workers stops helping. Results are cached to ``calibration.json`` keyed by
host fingerprint.
"""
import json
import os
import platform
import time
from typing import Callable, Optional

_FLEET_DIR = os.path.dirname(os.path.abspath(__file__))
_CACHE_FILE = os.path.join(_FLEET_DIR, "calibration.json")

# Reference kernel used to calibrate fleet width: every task ships a
# `baseline.c` that compiles cleanly against its own harness/kernel_api.h, so
# reusing the default task's baseline avoids maintaining a separate
# calibration-only kernel source. Resolved lazily (not at import time) since
# `core.evaluate.TASKS_DIR` is only populated once `data/v6/tasks`
# exists on disk.
_REF_TASK_ID: Optional[str] = None
_REF_SOURCE: Optional[str] = None


def _reference_defaults():
    """Return (ref_task_id, ref_source_path), resolved from hexkernels.core.evaluate."""
    global _REF_TASK_ID, _REF_SOURCE
    if _REF_TASK_ID is None or _REF_SOURCE is None:
        from hexkernels.core.evaluate import DEFAULT_TASK, TASKS_DIR
        _REF_TASK_ID = DEFAULT_TASK
        _REF_SOURCE = os.path.join(TASKS_DIR, DEFAULT_TASK, "baseline.c")
    return _REF_TASK_ID, _REF_SOURCE


class FleetCalibrationError(RuntimeError):
    """Raised when the reference kernel cannot be compiled+simulated."""


def _fingerprint() -> str:
    """Host identity string used as the calibration cache key."""
    return f"{platform.system()}-{platform.machine()}-{os.cpu_count()}cpu"


def _load_cache() -> dict:
    try:
        with open(_CACHE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _save_cache(data: dict) -> None:
    try:
        with open(_CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
    except OSError:
        pass  # Cache write failure is non-fatal.


def _widths_to_probe(max_width: int) -> list:
    """Generate ``[1, 2, 4, 8, 16, ...]`` capped at ``max_width``."""
    w = 1
    result = []
    while w <= max_width:
        result.append(w)
        w *= 2
    return result


def _default_run_reference(evaluate_batch_fn: Callable) -> Callable[[int, int], list]:
    """Return a callable(width, reps) -> list[feedback] using the real evaluate_batch."""
    ref_task_id, ref_source = _reference_defaults()

    def _run(width: int, reps: int) -> list:
        with open(ref_source, encoding="utf-8") as f:
            src = f.read()
        jobs = [{"task_id": ref_task_id, "source": src} for _ in range(reps)]
        return evaluate_batch_fn(jobs, width)

    return _run


def probe_fleet_size(max_width: Optional[int] = None, reps: Optional[int] = None,
                      force: bool = False, run_reference: Optional[Callable] = None) -> dict:
    """Probe and return the recommended parallel width for this host.

    Args:
        max_width: cap on widths tried (default ``min(2*cpu_count, 16)``).
        reps: jobs per width (default ``max(width, 4)``).
        force: ignore the cache and re-probe.
        run_reference: injectable callable(width, reps) -> list[feedback].
            Used by tests to avoid needing the real toolchain.

    Returns:
        dict with keys: ``recommended_width``, ``peak_throughput``, ``curve``,
        ``fingerprint``, ``cached``.

    Raises:
        FleetCalibrationError: if the reference kernel fails to compile/sim
            at width=1.
    """
    fp = _fingerprint()

    if not force:
        cache = _load_cache()
        if fp in cache:
            entry = cache[fp]
            return {
                "recommended_width": entry["recommended_width"],
                "peak_throughput": entry["peak_throughput"],
                "curve": entry["curve"],
                "fingerprint": fp,
                "cached": True,
            }

    if max_width is None:
        cpu = os.cpu_count() or 1
        max_width = min(2 * cpu, 16)

    # Set up the run_reference callable.
    if run_reference is None:
        from hexkernels.core.fleet.pool import evaluate_batch
        run_reference = _default_run_reference(evaluate_batch)

    widths = _widths_to_probe(max_width)

    # Validate reference at width=1 first.
    probe_reps = max(widths[0], 4)
    try:
        probe_results = run_reference(1, probe_reps)
    except Exception as exc:
        raise FleetCalibrationError(
            f"Reference kernel failed at width=1: {exc}"
        ) from exc

    # Check that the reference actually compiled and ran.
    any_ok = any(r.get("compiled") for r in probe_results)
    if not any_ok:
        raise FleetCalibrationError(
            "Reference kernel did not compile at width=1. "
            "Check that the Hexagon toolchain is installed and on PATH."
        )

    curve = []
    best_throughput = 0.0
    consecutive_declines = 0

    for w in widths:
        n = reps if reps is not None else max(w, 4)
        t0 = time.monotonic()
        run_reference(w, n)
        wall = time.monotonic() - t0
        if wall <= 0:
            wall = 1e-9
        throughput = n / wall
        mean_wall = wall / n

        curve.append({
            "width": w,
            "throughput": throughput,
            "mean_wall": mean_wall,
        })

        if throughput > best_throughput:
            best_throughput = throughput
            consecutive_declines = 0
        else:
            consecutive_declines += 1

        # Early stop: two consecutive widths below 0.9 * best.
        if throughput < 0.9 * best_throughput:
            if consecutive_declines >= 2:
                break

    # recommended_width = smallest width with throughput >= 0.95 * peak.
    peak_throughput = max(pt["throughput"] for pt in curve)
    recommended_width = curve[0]["width"]  # fallback to first
    for pt in curve:
        if pt["throughput"] >= 0.95 * peak_throughput:
            recommended_width = pt["width"]
            break

    result = {
        "recommended_width": recommended_width,
        "peak_throughput": peak_throughput,
        "curve": curve,
        "fingerprint": fp,
        "cached": False,
    }

    # Cache the result.
    cache = _load_cache()
    cache[fp] = {
        "recommended_width": recommended_width,
        "peak_throughput": peak_throughput,
        "curve": curve,
        "timestamp": time.time(),
    }
    _save_cache(cache)

    return result
