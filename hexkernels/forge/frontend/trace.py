"""Trace an arbitrary PyTorch module to a clean primitive `Graph` (graph.py).

Pipeline: `torch.export.export` -> `run_decompositions(decomp_table())` -> walk
`graph.nodes`, dropping the two node kinds that are not real operations
(`aten._assert_tensor_metadata` and `operator.getitem`) and resolving every
consumer of a `getitem` to reference the producer's k-th result directly
(`f"{producer_name}#{k}"`).

`decomp_table()` recipe is pinned by measurement (see
`run_artifacts/forge/frontend_pilot10.py`): `core_aten_decompositions()`
*updated with* `get_decompositions(DECOMP_FORCE)`. Passing `get_decompositions`
alone replaces the core table instead of extending it and yields a
less-decomposed graph (e.g. softmax stays a single opaque
`aten._softmax.default` instead of the measured
amax/sub/exp/sum/div chain) -- do not simplify this to one call.
"""
import operator

import torch
from torch._decomp import core_aten_decompositions, get_decompositions
from torch.export import export

from hexkernels.forge.frontend.graph import Graph, Node

# Ops the core table leaves fused; forced open because downstream emitters
# only know how to write scalar C loops for the decomposed primitives, not
# for these composite ops directly.
DECOMP_FORCE = (
    torch.ops.aten._softmax.default,
    torch.ops.aten.native_layer_norm.default,
    torch.ops.aten.gelu.default,
    torch.ops.aten._log_softmax.default,
)


def decomp_table() -> dict:
    table = dict(core_aten_decompositions())
    table.update(get_decompositions(DECOMP_FORCE))
    return table


def _dtype_str(dtype) -> str:
    return str(dtype).replace("torch.", "")


def _result_list(val):
    """[(shape, dtype), ...] for a tuple-producing node; () otherwise.

    Kept beside `_shape_dtype`, which returns (None, None) for the same input: that
    None is correct (there is no single shape) and this is where the information it
    cannot carry goes.
    """
    if isinstance(val, (tuple, list)):
        return tuple((tuple(v.shape), _dtype_str(v.dtype))
                     for v in val if v is not None and hasattr(v, "shape"))
    return ()


def _shape_dtype(val):
    """`node.meta["val"]` is a single FakeTensor for a single-result op, or a
    tuple/list of them for a multi-result op (e.g. var_mean). The latter has
    no single shape -- record `None` and let consumers index the producer by
    result number instead."""
    if val is None or isinstance(val, (tuple, list)):
        return None, None
    return tuple(val.shape), _dtype_str(val.dtype)


def _convert(value, alias, collected):
    """Recursively convert one arg/kwarg value: an `fx.Node` reference becomes
    its resolved name string (through `alias`, for getitem-produced names) and
    is appended to `collected` in encounter order; lists/tuples are walked
    element-wise; anything else (ints, dims, bools, ...) passes through
    unchanged."""
    if isinstance(value, torch.fx.Node):
        name = alias.get(value.name, value.name)
        collected.append(name)
        return name
    if isinstance(value, list):
        return [_convert(v, alias, collected) for v in value]
    if isinstance(value, tuple):
        return tuple(_convert(v, alias, collected) for v in value)
    return value


def trace(module, example_args, name) -> Graph:
    ep = export(module.eval(), example_args)
    ep = ep.run_decompositions(decomp_table())
    fx_nodes = ep.graph.nodes

    alias: dict[str, str] = {}  # getitem node name -> "producer#k"
    inputs: list[Node] = []
    nodes: list[Node] = []
    output_node = None

    for fx_node in fx_nodes:
        if fx_node.op == "placeholder":
            shape, dtype = _shape_dtype(fx_node.meta.get("val"))
            inputs.append(Node(
                name=fx_node.name, target="placeholder", args=(), kwargs={},
                shape=shape, dtype=dtype, inputs=(),
            ))
            continue

        if fx_node.op == "output":
            output_node = fx_node
            continue

        if fx_node.op != "call_function":
            # get_attr and any other bookkeeping ops carry no arithmetic value.
            continue

        target_str = str(fx_node.target)

        if "_assert_tensor_metadata" in target_str:
            continue

        if fx_node.target is operator.getitem:
            producer, index = fx_node.args[0], fx_node.args[1]
            producer_name = alias.get(producer.name, producer.name)
            alias[fx_node.name] = f"{producer_name}#{index}"
            continue

        collected: list[str] = []
        conv_args = tuple(_convert(a, alias, collected) for a in fx_node.args)
        conv_kwargs = {k: _convert(v, alias, collected) for k, v in fx_node.kwargs.items()}
        shape, dtype = _shape_dtype(fx_node.meta.get("val"))
        results = _result_list(fx_node.meta.get("val"))

        nodes.append(Node(
            name=fx_node.name, target=target_str, args=conv_args, kwargs=conv_kwargs,
            shape=shape, dtype=dtype, inputs=tuple(collected), results=results,
        ))

    outputs: tuple[str, ...] = ()
    if output_node is not None:
        collected_out: list[str] = []
        _convert(output_node.args[0], alias, collected_out)
        outputs = tuple(collected_out)

    return Graph(name=name, nodes=tuple(nodes), inputs=tuple(inputs), outputs=outputs)
