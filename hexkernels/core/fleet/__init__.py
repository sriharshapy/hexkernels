"""Sim-fleet sizer + parallel reward executor.

Re-exports the public surface consumed by later stages (eval collectors,
training loops): a parallel batch evaluator (:func:`evaluate_batch`), a
helper to adapt source-string evaluators to the path-based signature it
expects (:func:`as_path_fn`), and a fleet-width auto-tuner
(:func:`probe_fleet_size`) plus its error type.
"""
from hexkernels.core.fleet.pool import as_path_fn, evaluate_batch
from hexkernels.core.fleet.sizer import FleetCalibrationError, probe_fleet_size

__all__ = ["as_path_fn", "evaluate_batch", "probe_fleet_size", "FleetCalibrationError"]
