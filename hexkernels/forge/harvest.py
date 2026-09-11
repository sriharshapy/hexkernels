"""Emit the op catalogue from every configured source.

Writes to ``run_artifacts/forge2/``:
  ops.jsonl        one row per schema, with its parse and its provenance
  environment.json the observed environment -- recorded as data, never asserted
  SUMMARY.md       counts by namespace and by structural class

Nothing is dropped. Rows that are not candidate kernels carry a ``klass`` and a
``reason`` so every exclusion is auditable rather than asserted.

    python -m hexkernels.forge.harvest
"""
import argparse
import collections
import json
import os

from hexkernels.forge.sources import torch_registry

SOURCES = {torch_registry.SOURCE_NAME: torch_registry}

# Structural classes. This is NOT the compute-pattern axis the design settles on
# for candidate kernels -- it is the coarse filter that decides which rows are
# even eligible for a pattern. Exactly one class per row; the classes partition.
KLASSES = ("kernel", "variant", "gradient", "plumbing", "no_tensor", "unsupported_domain")

# ALLOWLIST, not a blocklist. Only these namespaces can hold candidate kernels;
# everything else is a backend shim, a domain we do not target, or an internal.
#
# A blocklist was tried first and leaked 36 rows -- `_c10d_functional`,
# `_c10d_functional_autograd`, `_dtensor`, `symm_mem`, `mkldnn_prepacked`,
# `prepacked`, `inductor`, `onnx`, `export`, `_test`, `mkl`, `rngprims`,
# `sparse`, `quantization` -- because it matched namespace names exactly and
# torch spells the same domain several ways. An allowlist fails closed: a
# namespace added by a future torch is excluded with a reason until someone
# deliberately admits it.
#
# `prims` (PrimTorch references) is IN. `prim` (TorchScript interpreter
# bookkeeping) is OUT. They differ by one letter and collapsing them is an easy,
# silent bug -- 123 interpreter ops would be admitted as kernels.
_CANDIDATE_NS = {
    "aten": "the core operator set",
    "prims": "PrimTorch reference decompositions",
}

_NS_REASON = {
    "prim": "torchscript interpreter bookkeeping",
    "quantized": "qtensor -- does not export",
    "_quantized": "qtensor -- does not export",
    "quantization": "qtensor -- does not export",
    "sparse": "sparse/quantized linear backend",
    "c10d": "distributed collective",
    "_c10d_functional": "distributed collective",
    "_c10d_functional_autograd": "distributed collective",
    "_dtensor": "distributed tensor",
    "symm_mem": "symmetric-memory collective",
    "cuda": "cuda-only",
    "mkl": "mkl backend",
    "mkldnn": "mkldnn backend",
    "mkldnn_prepacked": "mkldnn backend",
    "onednn": "onednn backend",
    "prepacked": "xnnpack prepacked backend",
    "static_runtime": "runtime internal",
    "inductor": "compiler internal",
    "profiler": "profiler internal",
    "export": "export internal",
    "onnx": "onnx export shim",
    "rngprims": "stateful RNG",
    "_test": "test-only op",
}

_UNSUPPORTED_TOKENS = (
    ("sparse", "sparse tensor"),
    ("nested", "nested tensor"),
    ("mkldnn", "mkldnn backend"),
    ("_foreach", "multi-tensor apply, not a single kernel"),
    ("quantize", "qtensor -- does not export"),
    ("dequantize", "qtensor -- does not export"),
)

# Tensor in, tensor out, but no arithmetic: no FLOPs means nothing to accelerate.
_PLUMBING = {
    "view", "reshape", "expand", "permute", "transpose", "t", "squeeze",
    "unsqueeze", "flatten", "unflatten", "contiguous", "detach", "alias",
    "as_strided", "size", "stride", "numel", "dim", "is_contiguous", "clone",
    "to", "_to_copy", "copy", "copy_", "item", "narrow", "select", "slice",
    "chunk", "split", "unbind", "movedim", "swapaxes", "swapdims", "ravel",
    "broadcast_to", "reshape_as", "view_as", "expand_as", "type_as", "empty",
    "empty_like", "zeros_like", "ones_like", "full_like", "resize_", "set_",
}


def classify(row):
    """row -> (klass, reason). Exactly one class; reason names the trigger."""
    if row.get("parse_error"):
        return "no_tensor", "unparseable schema"
    ns, op = row["namespace"], row["op"]

    if ns not in _CANDIDATE_NS:
        why = _NS_REASON.get(ns, "namespace not in the candidate set")
        return "unsupported_domain", f"namespace:{ns} -- {why}"
    for tok, why in _UNSUPPORTED_TOKENS:
        if tok in op:
            return "unsupported_domain", f"name contains {tok!r} -- {why}"

    if op.startswith("_test"):
        return "unsupported_domain", "test-only op"

    if "backward" in op or op.endswith("_backward") or "grad" in op:
        return "gradient", "derivative op -- never a forward inference kernel"
    if row["in_place"]:
        return "variant", "in-place form of a base op"
    # `.out` is the common spelling, but torch also uses NAMED out-overloads --
    # `cummax.dimname_out`, `frexp.Tensor_out`, `histogram.bin_ct_out`. Matching
    # only the bare `.out` admitted 7 of those as fresh kernels.
    if row["out_variant"] or row["overload"].endswith("_out"):
        return "variant", "out= form of a base op"

    if not (row["has_tensor_in"] and row["has_tensor_out"]):
        return "no_tensor", "does not both take and return a tensor"

    base = op.lstrip("_")
    if base in _PLUMBING:
        return "plumbing", f"{base}: layout/metadata only, no arithmetic"
    # `_cast_Byte`, `_cast_Float`, ... are dtype conversions, not arithmetic.
    if op.startswith("_cast_"):
        return "plumbing", "dtype cast only, no arithmetic"

    return "kernel", "tensor in, tensor out, arithmetic"


def build():
    rows, env = [], {}
    for name, mod in SOURCES.items():
        env[name] = mod.environment()
        for row in mod.rows():
            row["source"] = name
            row["klass"], row["reason"] = classify(row)
            rows.append(row)
        env[name]["observed_schema_count"] = sum(1 for r in rows if r["source"] == name)
    return rows, env


def summarize(rows, env):
    by_klass = collections.Counter(r["klass"] for r in rows)
    by_ns = collections.Counter(r["namespace"] for r in rows if not r.get("parse_error"))
    kernel_ns = collections.Counter(r["namespace"] for r in rows if r["klass"] == "kernel")
    reasons = collections.Counter(r["reason"] for r in rows if r["klass"] != "kernel")

    L = ["# Forge v2 — operator catalogue", ""]
    for name, e in env.items():
        L += [f"**Source `{name}`** — torch {e['torch_version']}, "
              f"{e['observed_schema_count']} schemas observed.", "",
              f"> {e['citation']}", ""]
    L += ["Counts are observed values recorded as data. The same version string has yielded",
          "different totals on different builds, so none of these is asserted by a test.", ""]

    L += ["## By class", "", "| class | count |", "|---|---|"]
    for k in KLASSES:
        L.append(f"| {k} | {by_klass.get(k, 0)} |")
    L += [f"| **total** | **{sum(by_klass.values())}** |", ""]

    L += ["## Candidate kernels by namespace", "", "| namespace | kernels | of total |", "|---|---|---|"]
    for ns, n in kernel_ns.most_common():
        L.append(f"| {ns} | {n} | {by_ns.get(ns, 0)} |")
    L += ["", "## Why rows were excluded", "", "| reason | count |", "|---|---|"]
    for why, n in reasons.most_common(25):
        L.append(f"| {why} | {n} |")
    return "\n".join(L) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="benchmark")
    args = ap.parse_args(argv)
    os.makedirs(args.out, exist_ok=True)

    rows, env = build()
    with open(os.path.join(args.out, "ops.jsonl"), "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    with open(os.path.join(args.out, "environment.json"), "w", encoding="utf-8") as f:
        json.dump(env, f, indent=2, sort_keys=True)
    with open(os.path.join(args.out, "SUMMARY.md"), "w", encoding="utf-8") as f:
        f.write(summarize(rows, env))

    n_kernel = sum(1 for r in rows if r["klass"] == "kernel")
    print(f"{len(rows)} schemas -> {n_kernel} candidate kernels")
    print(f"wrote {args.out}/ops.jsonl, environment.json, SUMMARY.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
