"""Backfill n_primitives/n_plumbing/ops_per_element onto benchmark/pool.json,
using the exact same mine._depth_counts helper the miner itself uses at
selection time, so a backfilled value cannot diverge from a freshly-mined one.

Why backfilling is legitimate, not a workaround: the plan rejected a depth
*axis* for the test set on the explicit grounds that depth "does not
determine which mechanism a task needs" -- but that argument only holds if
depth stays MEASURABLE after the fact. Computing n_primitives/n_plumbing for
specs that were already selected on other criteria, using the miner's own
admissibility test, is exactly what that argument requires: it is the
design's own argument turned into code, not a shortcut around it.

n_primitives/n_plumbing/ops_per_element were backfilled onto the existing
pool this way rather than produced by re-mining, and this script is committed
so the procedure can be re-run and diffed from source control rather than
existing only as prose.

Run from repo root: python scripts/backfill_depth.py
"""
import json
import math
import pathlib
import subprocess
import sys

import torch

ROOT = pathlib.Path(subprocess.check_output(
    ["git", "rev-parse", "--show-toplevel"], text=True).strip())
sys.path.insert(0, str(ROOT))

from hexforge import mine as M
from hexforge.frontend.trace import trace

POOL = ROOT / "benchmark" / "pool.json"

hidx = M._harvest_index()

doc = json.loads(POOL.read_text(encoding="utf-8"))

n_single = n_fused = n_hmx = 0
errors = []

for bi, batch in enumerate(doc["batches"]):
    for si, spec in enumerate(batch):
        if "stages" in spec:
            n_fused += 1
            arity = 1 if spec["sig"] == "unary" else 2
            from hexforge.mined import FusedCall
            mod = FusedCall(spec["stages"])
            shape = tuple(spec["shape"])
            args = tuple(torch.rand(shape, dtype=torch.float32) + 0.5
                         for _ in range(arity))
            try:
                g = trace(mod, args, f"backfill_fuse_{spec['op']}")
            except Exception as e:                      # noqa: BLE001
                errors.append((bi, si, spec.get("op"), "fused-trace-failed", str(e)))
                continue
            targets = [n.target for n in g.nodes]
            n_prim, n_plumb = M._depth_counts(targets, hidx)
        else:
            dtype_str = spec["dtype"]
            dtype = getattr(torch, dtype_str)
            if spec["dtype"] == "float16":
                n_hmx += 1
            else:
                n_single += 1
            gsig = M.graph_signature(spec["op"], spec["overload"], spec["sig"],
                                      hidx, dtype, spec.get("synth_plan"))
            if gsig is None:
                errors.append((bi, si, spec.get("op"), "graph_signature-None", ""))
                continue
            n_prim, n_plumb = M._depth_counts(gsig, hidx)

        shape = spec["shape"]
        spec["n_primitives"] = n_prim
        spec["n_plumbing"] = n_plumb
        spec["ops_per_element"] = (n_prim / max(1, math.prod(shape))
                                    if shape else None)

print(f"single={n_single} hmx_twin={n_hmx} fused={n_fused}")
if errors:
    print(f"ERRORS: {len(errors)}")
    for e in errors[:20]:
        print(" ", e)
else:
    print("no errors")

POOL.write_text(json.dumps(doc, indent=1), encoding="utf-8")
print("wrote", POOL)
