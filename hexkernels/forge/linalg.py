"""Real Linalg IR for a traced module, via torch-mlir.

WHAT THIS ADDS OVER `forge.frontend.schedule`
---------------------------------------------
`schedule.py` derives `iterator_types` and indexing maps from a hand-written
table. This produces the compiler's own Linalg, which is the same information
plus the structure around it -- named ops (`linalg.matmul`,
`linalg.conv_2d_nchw_fchw`), the tensor types, the fill/init that establishes an
accumulator, and the exact affine-map aliases.

Both are kept, deliberately:

  * Linalg is ground truth, and `validate_schedule.py` diffs the table against
    it (7 exact / 3 named-op / 0 disagreements over both batches).
  * `schedule.py` still answers a question Linalg does not: `vectorizable_loop`
    -- the innermost parallel loop where *every* operand is unit- or zero-stride,
    which is a Hexagon HVX question, not a dialect one. It also has no
    dependency, so the pipeline degrades to it if torch-mlir is absent.

INSTALL -- READ THIS BEFORE CONCLUDING IT IS UNAVAILABLE
--------------------------------------------------------
`pip install torch-mlir` FAILS, and the failure is misleading. The PyPI package
was abandoned at 20221213.686 (December 2022) with cp310/cp37 wheels only, so on
any other Python pip reports "No matching distribution found (from versions:
none)". **That means "wrong Python for a dead package", not "unavailable"** --
misreading it as the latter cost this project a wrong design decision, and then
checking the also-stale `llvm/torch-mlir` snapshot releases (last: 2024-01-27)
cost a second one.

Current wheels are published here:

    https://github.com/llvm/torch-mlir-release/releases/tag/dev-wheels
    pip install --no-deps <torch_mlir-<date>-cp311-cp311-win_amd64.whl URL>

`--no-deps` is required and safe: the wheel declares only numpy and packaging --
**no torch pin** -- and imports against this repo's torch 2.7.1. The 2024 wheels
pinned `torch==2.3.0.dev20240122` and could not, which is why they were rejected.

These are nightlies. PIN AN EXACT DATED WHEEL: the API has already moved once
(pre-2024 `torch_mlir.compile` no longer exists; the entry point is now
`torch_mlir.fx.export_and_import`), and `aten::__rshift__` -- needed by the int8
requant chain -- lowers on the current build and did not on the 2024 one.
"""
import re

# Pinned for provenance. Update deliberately, and re-run validate_schedule.py
# afterwards: these are nightly builds and the lowering can change.
PINNED_WHEEL = "torch_mlir-20260802.543-cp311-cp311-win_amd64.whl"
WHEEL_SOURCE = "https://github.com/llvm/torch-mlir-release/releases/tag/dev-wheels"


class LinalgUnavailable(RuntimeError):
    """torch-mlir is not installed, or could not lower this graph."""


def available() -> bool:
    """Is torch-mlir importable? Callers degrade to `schedule.py` if not."""
    try:
        import torch_mlir.fx  # noqa: F401
    except Exception:
        return False
    return True


def generalize(mod) -> None:
    """Rewrite named Linalg ops into `linalg.generic`, in place.

    WHY THIS MATTERS FOR VALIDATION
    -------------------------------
    Linalg prefers NAMED ops -- `linalg.matmul`, `linalg.conv_2d_nchw_fchw` --
    whose iterator types and indexing maps live in the op's C++ definition and
    are therefore absent from the printed IR. Those are exactly the two ops with
    non-trivial schedules, so the validator could compare everything except the
    cases worth comparing: three of ten kernels came back "named-op", meaning
    "nothing to check", and a wrong entry for either would have passed.

    `linalg-generalize-named-ops` materialises the definition. The conv map it
    prints is the reason the `Term` representation exists:

        input  (d0..d6) -> (d0, d4, d2 * stride + d5 * dilation, ...)

    -- two loops indexing one operand axis with coefficients, which a plain
    "loop k walks axis a" table cannot express at all.

    One representational difference to expect, not a bug: Linalg lowers padding
    as an explicit `tensor.pad` producing a larger input, while
    `primitives._emit_convolution` keeps the original buffer and guards with a
    bounds `if`. Neither puts a constant in the map, so the maps still compare.
    """
    from torch_mlir.passmanager import PassManager
    with mod.context:
        PassManager.parse(
            "builtin.module(func.func(linalg-generalize-named-ops))"
        ).run(mod.operation)


def linalg_ir(module, example_args, generalized: bool = False) -> str:
    """Lower a module to Linalg-on-tensors and return the IR as text.

    Raises `LinalgUnavailable` rather than returning a sentinel, so a caller
    that requires real Linalg fails loudly instead of silently continuing with
    a partial prompt.

    `generalized=False` for prompts: the named form is what the compiler
    naturally emits and reads better. `generalized=True` for `validate_schedule`,
    which needs the maps the named form does not print (see `generalize`).
    """
    try:
        import torch_mlir.fx as fx
    except Exception as exc:
        raise LinalgUnavailable(
            f"torch-mlir not installed ({exc}). Install the pinned wheel "
            f"{PINNED_WHEEL} from {WHEEL_SOURCE} with --no-deps; note that "
            "`pip install torch-mlir` fails on the Python version because the "
            "PyPI package is abandoned, which is NOT the same as unavailable."
        ) from None
    try:
        mod = fx.export_and_import(
            module.eval(), *example_args,
            output_type=fx.OutputType.LINALG_ON_TENSORS)
    except Exception as exc:
        raise LinalgUnavailable(
            f"torch-mlir could not lower this graph: {type(exc).__name__}: "
            f"{str(exc)[:300]}") from None
    if generalized:
        try:
            generalize(mod)
        except Exception as exc:
            raise LinalgUnavailable(
                f"linalg-generalize-named-ops failed: {type(exc).__name__}: "
                f"{str(exc)[:300]}") from None
    return str(mod)


def structured_only(ir: str) -> str:
    """The Linalg worth showing a reader: affine-map aliases and structured ops.

    Full IR for one kernel runs to a few KB of SSA that is mostly allocation and
    constant plumbing (`tensor.empty`, `arith.constant`, bufferization). This keeps
    the two things the schedule lives in -- the `#map` aliases and the `linalg.*`
    operations with their `iterator_types` and `indexing_maps`.

    Structured ops are kept WHOLE, region and closing brace included. An earlier
    version filtered line by line and so emitted a dangling `^bb0` with no
    enclosing block and no closing `}` -- syntactically invalid IR, which is a bad
    thing to hand a model as ground truth even when the schedule attributes on the
    op line are still readable.
    """
    lines = ir.splitlines()
    out, i = [], 0
    while i < len(lines):
        s = lines[i].strip()
        if s.startswith("#map") or s.startswith("func.func"):
            out.append(s)
            i += 1
            continue
        if "linalg." in s:
            # Consume through the end of the op's region, by brace balance.
            depth = s.count("{") - s.count("}")
            out.append("  " + s)
            i += 1
            while depth > 0 and i < len(lines):
                b = lines[i].strip()
                out.append("    " + b)
                depth += b.count("{") - b.count("}")
                i += 1
            continue
        i += 1
    return "\n".join(out + ["}"])


def summary(ir: str) -> dict:
    """Iterator types and named ops, for reporting and for the validator."""
    # Anchored to end of line, NOT `[^>]*>`. A map's own body contains `->`, so
    # a non-greedy or negated-`>` match stops inside the arrow and yields
    # `affine_map<(d0, d1) ->` for every multi-dimensional map. That is what this
    # did until 2026-08-03, and the test guarding it asserted only
    # `startswith("affine_map<")`, which the truncated string satisfies. One
    # alias per line, so the line end is the reliable terminator.
    aliases = dict(re.findall(r"^(#map\d*) = (affine_map<.*>)$", ir, re.M))
    generics = []
    for m in re.finditer(r"linalg\.(\w+)\s*\{([^}]*)\}", ir):
        attrs = m.group(2)
        it = re.search(r'iterator_types = \[([^\]]*)\]', attrs)
        mp = re.search(r'indexing_maps = \[([^\]]*)\]', attrs)
        if not it:
            continue
        # A RANK-0 OP HAS AN EMPTY LIST, AND `"".split(",")` IS `[""]`, NOT `[]`.
        # That distinction is not pedantry: `iterator_types = []` is what the
        # compiler prints for a scalar op, which appears as soon as whole-tensor
        # reductions exist (`mean.default` lowers to a rank-2 reduction followed by
        # a rank-0 divide). Without the filter the empty string reached a consumer
        # that indexed its first character and the whole comparison crashed with
        # `IndexError: string index out of range`.
        generics.append({
            "op": m.group(1),
            "iterator_types": [x.strip().strip('"') for x in it.group(1).split(",")
                               if x.strip()],
            "indexing_maps": [aliases.get(x.strip(), x.strip())
                              for x in mp.group(1).split(",") if x.strip()]
            if mp else [],
        })
    return {
        "generics": generics,
        "named_ops": sorted(set(re.findall(
            r"linalg\.(matmul|conv_2d\w*|softmax|batch_matmul)\b", ir))),
        "maps": aliases,
    }
