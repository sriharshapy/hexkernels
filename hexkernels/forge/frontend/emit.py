"""`Graph` -> one C translation unit (Task 2).

Declares a static buffer per intermediate node, emits each node's loop nest
in program order, and wraps the result in `extern "C" void
candidate_kernel(...)` -- `extern "C"` is required so the symbol name is not
mangled (`_Z16candidate_kernel...`), which would break harness linkage.

Parameters follow the graph's placeholder order exactly (torch.export lifts
parameters ahead of activations; that ordering is not "fixed" here), each a
`const <ctype> *`, followed by one output pointer per graph output.

Portable C++17 only -- no `Q6_`/`HVX_`/`hexagon_` tokens, no VTCM, no
intrinsic, no vector-width constant. No fusion, tiling or vectorisation:
this is the naive reference a later stage is asked to accelerate.

Emission is deterministic: the same `Graph` always produces byte-identical
C, because it only ever walks `graph.nodes`/`graph.inputs`/`graph.outputs`
in the order the (deterministic) tracer produced them.
"""
from hexkernels.forge.frontend.graph import DTYPE_C, Graph, numel
from hexkernels.forge.frontend.primitives import cname, emit_node



def output_shape_dtype(graph, out_name):
    """(shape, dtype) of one graph output, resolving a `producer#k` reference.

    A multi-result producer has `shape`/`dtype` of None -- there is no single
    answer -- and the k-th result's own pair lives in `Node.results`. Three places
    need this and each used to index `buffers[name]` directly, which raised
    `KeyError: 'var_mean#0'` because no buffer is registered under the composite
    name.
    """
    base = out_name.split("#")[0]
    by_name = {n.name: n for n in graph.inputs}
    by_name.update({n.name: n for n in graph.nodes})
    node = by_name[base]
    if "#" in out_name:
        k = int(out_name.split("#")[1])
        return node.results[k]
    return node.shape, node.dtype


def kernel_signature(graph: Graph) -> str:
    """The `candidate_kernel` declaration this graph implies, without `extern "C"`.

    Exposed rather than built inline because three consumers need to agree on it
    exactly: the emitted reference, the generated harness's prototype, and the
    acceleration prompt that tells a model what to write. A candidate whose
    signature differs by one parameter fails to link, and the error surfaces as
    a compile failure that reads like a model error.
    """
    buffers = {n.name: n for n in graph.inputs}
    buffers.update({n.name: n for n in graph.nodes})
    params = [f"const {DTYPE_C[i.dtype]} *{cname(i.name)}" for i in graph.inputs]
    for k, out_name in enumerate(graph.outputs):
        _shape, dt = output_shape_dtype(graph, out_name)
        params.append(f"{DTYPE_C[dt]} *out{k}")
    return f"void candidate_kernel({', '.join(params)})"


def emit_c(graph: Graph) -> str:
    buffers = {n.name: n for n in graph.inputs}
    buffers.update({n.name: n for n in graph.nodes})

    lines = ["#include <stdint.h>", "#include <math.h>", ""]

    for n in graph.nodes:
        if n.results:
            # One buffer per result. The node itself has no buffer: nothing can
            # reference `var_mean`, only `var_mean#0` and `var_mean#1`.
            for k, (shape, dtype) in enumerate(n.results):
                lines.append(f"static {DTYPE_C[dtype]} "
                             f"{cname(n.name + '#' + str(k))}[{numel(shape)}];")
        else:
            lines.append(
                f"static {DTYPE_C[n.dtype]} {cname(n.name)}[{numel(n.shape)}];")
    lines.append("")

    # `cname()` on every placeholder too: an input could just as easily be
    # named e.g. `div` by whatever traced it, and it becomes a C identifier
    # here the same way an intermediate node's name does (see primitives.py).
    lines.append(f'extern "C" {kernel_signature(graph)} {{')
    for n in graph.nodes:
        lines.append(emit_node(n, graph, buffers))

    for k, out_name in enumerate(graph.outputs):
        shape, _dt = output_shape_dtype(graph, out_name)
        count = numel(shape)
        lines.append(f"  for (int i = 0; i < {count}; i++) {{ out{k}[i] = {cname(out_name)}[i]; }}")

    lines.append("}")
    return "\n".join(lines) + "\n"
