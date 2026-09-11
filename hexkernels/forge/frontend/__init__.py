"""Torch front end: trace an arbitrary PyTorch module to a clean primitive graph.

This package owns the stage between "a PyTorch module" and "a portable scalar C
reference": tracing + decomposition only. It does not read or write `data/v6/`
and does not emit C (that is a later stage).
"""
