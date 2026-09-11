"""Parallel reward executor: ``evaluate_batch(jobs, width) -> list[feedback]``.

Each job is a dict with ``task_id`` and ``source`` plus arbitrary passthrough
keys. Results are returned IN INPUT ORDER, each augmented with the job's
passthrough keys. Any per-job exception is caught and returned as a
fail-closed feedback dict.
"""
import hashlib
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

EvaluateFn = Callable[[str, str], dict]
SourceEvaluateFn = Callable[..., dict]


def _default_evaluate_fn() -> EvaluateFn:
    """Return ``evaluate_fn(path, task_id)`` wrapping the real reward core.

    Matches the ``callable(path: str, task_id: str) -> dict`` signature
    expected everywhere else in the pool. Returns a feedback dict; never
    raises (compile/sim failures are captured inside the dict itself by
    ``evaluate``).
    """
    from hexkernels.core.evaluate import evaluate

    def _run(path: str, task_id: str) -> dict:
        return evaluate(path, task_id=task_id)

    return _run


def _fail_closed(job: dict, exc: Exception) -> dict:
    """Build a fail-closed feedback dict from a job dict and an exception."""
    result = {
        "compiled": False,
        "correct": False,
        "used_hvx": False,
        "cycles": None,
        "error_text": str(exc),
    }
    # Merge passthrough keys (everything except task_id and source).
    for k, v in job.items():
        if k not in ("task_id", "source"):
            result[k] = v
    return result


def as_path_fn(source_fn: SourceEvaluateFn) -> EvaluateFn:
    """Adapt a source-string evaluate fn into the path-based signature.

    ``evaluate_batch`` writes each job's source to a temp file and calls
    ``evaluate_fn(path, task_id)``. If your evaluator instead takes
    ``(src: str, task_id=...)``, wrap it with ``as_path_fn`` so the pool can
    call it correctly.

    Args:
        source_fn: callable(src: str, task_id: str) -> dict.

    Returns:
        callable(path: str, task_id: str) -> dict that reads the file at
        ``path`` (utf-8) and forwards to ``source_fn``.
    """
    def _run(path: str, task_id: str) -> dict:
        with open(path, encoding="utf-8") as f:
            src = f.read()
        return source_fn(src, task_id=task_id)
    return _run


def _source_key(job: dict) -> tuple:
    """Stable deduplication key: ``(task_id, sha256 of source)``."""
    src = job.get("source", "")
    digest = hashlib.sha256(src.encode("utf-8", errors="replace")).hexdigest()
    return (job.get("task_id", ""), digest)


def evaluate_batch(jobs: list, width: int, evaluate_fn: Optional[EvaluateFn] = None) -> list:
    """Evaluate a batch of jobs in parallel using a ``ThreadPoolExecutor``.

    Identical ``(task_id, source)`` pairs within the batch are evaluated only
    once; the result is copied to all duplicate indices (parity: same source
    -> same feedback by definition).

    Args:
        jobs: list of dicts with ``task_id``, ``source``, and any passthrough
            keys.
        width: max number of parallel workers (``ThreadPoolExecutor``
            ``max_workers``).
        evaluate_fn: callable(path: str, task_id: str) -> dict. If ``None``,
            uses the real reward core (file-path based).

    Returns:
        list of feedback dicts in the same order as ``jobs``, each augmented
        with the job's passthrough keys.
    """
    if not jobs:
        return []

    if evaluate_fn is None:
        evaluate_fn = _default_evaluate_fn()

    # --- source deduplication ---
    # Map each unique (task_id, source_hash) -> list of job indices.
    # We only submit one evaluate call per unique source; results are
    # broadcast back to all duplicate indices afterwards.
    key_to_indices: dict = {}
    for i, job in enumerate(jobs):
        k = _source_key(job)
        key_to_indices.setdefault(k, []).append(i)

    # unique_jobs: one representative job per key (the first occurrence).
    unique_jobs = [(indices[0], jobs[indices[0]]) for indices in key_to_indices.values()]

    # We need results in input order; submit with index as the key.
    results: list = [None] * len(jobs)

    def _run_job(idx: int, job: dict):
        task_id = job["task_id"]
        source = job["source"]
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(suffix=".c", prefix="fleet_")
            try:
                with os.fdopen(fd, "w", encoding="utf-8") as f:
                    f.write(source)
            except Exception:
                # fdopen failed to write -- fd may still be open.
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise
            fb = evaluate_fn(tmp_path, task_id)
        except Exception as exc:
            fb = _fail_closed(job, exc)
        else:
            # Merge passthrough keys into feedback.
            for k, v in job.items():
                if k not in ("task_id", "source"):
                    fb[k] = v
        finally:
            if tmp_path and os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
        return idx, fb

    with ThreadPoolExecutor(max_workers=width) as executor:
        futures = {executor.submit(_run_job, i, job): i for i, job in unique_jobs}
        for future in as_completed(futures):
            idx, fb = future.result()
            results[idx] = fb

    # Broadcast results from the first occurrence to all duplicate indices.
    for indices in key_to_indices.values():
        if len(indices) > 1:
            primary_fb = results[indices[0]]
            for dup_idx in indices[1:]:
                dup_job = jobs[dup_idx]
                dup_fb = dict(primary_fb)  # shallow copy of the feedback dict
                # Re-apply passthrough keys specific to the duplicate job
                # (they may differ from the primary job's passthrough keys).
                for k, v in dup_job.items():
                    if k not in ("task_id", "source"):
                        dup_fb[k] = v
                results[dup_idx] = dup_fb

    return results
