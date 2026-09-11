"""The primitive graph IR: two frozen dataclasses plus the shape helpers every
downstream emitter needs for flat, row-major indexing.

Deliberately minimal -- this module carries no torch import and no emission
logic. `trace.py` is the only producer of `Graph`/`Node` instances.
"""
from dataclasses import dataclass


@dataclass(frozen=True)
class Node:
    """One primitive op in the traced graph.

    `inputs` holds each argument's *name* reference: either the name of the
    producing `Node`/placeholder, or `f"{producer_name}#{k}"` when the
    argument is the k-th result of a multi-output producer (e.g. `var_mean`).
    Non-tensor arguments (python ints, dims, etc.) live in `args`/`kwargs`,
    not in `inputs`.
    """

    name: str
    target: str
    args: tuple
    kwargs: dict
    shape: tuple
    dtype: str
    inputs: tuple[str, ...]
    # PER-RESULT (shape, dtype) for an op that returns a TUPLE.
    #
    # `shape`/`dtype` above are None for such a node, because there is no single
    # answer -- `var_mean` returns a variance AND a mean, `max.dim` values AND
    # indices. That None reached `DTYPE_C[...]` and raised `KeyError: None`,
    # which is what 8 harvested ops were blocked behind. Consumers already
    # reference the k-th result as `producer#k` (see `inputs` above); this is
    # where the k-th result's own shape and dtype live.
    results: tuple = ()


@dataclass(frozen=True)
class Graph:
    """A traced module: placeholders (`inputs`), the primitive op sequence
    (`nodes`, in program order), and the names of the graph's outputs."""

    name: str
    nodes: tuple[Node, ...]
    inputs: tuple[Node, ...]
    outputs: tuple[str, ...]


# torch dtype string (str(tensor.dtype) with the "torch." prefix stripped) ->
# the C type every emitter downstream renders it as.
DTYPE_C: dict[str, str] = {
    "float32": "float",
    "int32": "int32_t",
    "int8": "int8_t",
    "int64": "int64_t",
    "uint8": "uint8_t",
    "float16": "_Float16",
    # `bool` as a byte, not C++ `bool`. Measured need: it was the single largest
    # coverage blocker in the harvest -- 63 eligible ops decompose through a
    # boolean intermediate (every comparison, `where`, the logical family, the
    # `isnan`/`isinf` predicates) and every one of them died on `KeyError:
    # 'bool'` in this table, which reads as "no emitter" while being a missing
    # C type. A byte matches torch's own 1-byte bool storage, so the golden
    # initialiser's 0/1 is bit-exact, and it keeps `&`/`|`/`^` well-defined
    # where C++ `bool` would integral-promote.
    "bool": "unsigned char",
}


def strides(shape):
    """Row-major element strides. Used by every emitter for flat indexing."""
    s = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        s[i] = s[i + 1] * shape[i + 1]
    return tuple(s)


def numel(shape):
    n = 1
    for e in shape:
        n *= e
    return n
