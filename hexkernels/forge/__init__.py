"""Forge: PyTorch operator -> Linalg IR -> MLIR fusion/bufferisation ->
affine loops -> C reference, then an LLM pass reconstructs it as
accelerated Hexagon C. A kernel counts only when it compiled, ran, and
passed its own generated golden harness.
"""
