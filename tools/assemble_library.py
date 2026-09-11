"""Assemble `kernels/` from the three upstream sources, into one schema.

A kernel is admitted ONLY if its accelerated source actually targets the
NSP -- HVX (`HVX_Vector`, `Q6_V*`, `Q6_W*`) or HMX. A scalar C file is never a
library entry; it ships as `reference.c`, which is a different role.

    python tools/assemble_library.py --hexbench ../hexbench \
        --v6 ../HVX-clean/data/v6 --out kernels

Three origins, one directory schema:

  expert  hand-written HVX/HMX kernels from the v6 corpus, each with a measured
          `expert_kernel_cycles` against its own scalar baseline.
  mined   kernels written against operators mined from the PyTorch registry.
          READ `tier_match`: most were authored for a different tier than the
          task they are filed under here, and will be rejected by the static
          entitlement gate. Retained deliberately -- see LIBRARY.md.
  model   kernels a language model wrote that the disassembly detector confirms
          reached HVX. One entry per task; losing seeds stay in `results/`.

Every entry gets the same files, so a consumer never branches on origin:

    kernels/<origin>/<name>/
        kernel.c       the accelerated kernel            (always)
        reference.c    scalar ground truth               (bundle=complete)
        harness.c      correctness + cycle harness       (bundle=complete)
        kernel_api.h   entry-point declaration           (v6 only)
        PROMPT.md      the ask that produced the kernel  (where one survives)
        spec.json      normalised metadata               (always)

`spec.json.bundle` is `complete` when reference+harness are present and
`kernel-only` when they are not, so "can I build this?" is one field and never
a directory listing.
"""

import argparse
import glob
import hashlib
import json
import os
import re
import shutil

# The admission test. `Q6_V`/`Q6_W` catch the HVX intrinsic families; the HMX
# names are checked separately so the metadata can record which mechanism a
# kernel reaches, not merely that it reaches one.
HVX_RE = re.compile(r"HVX_Vector|Q6_V[a-zA-Z0-9_]*\(|Q6_W[a-zA-Z0-9_]*\(")
HMX_RE = re.compile(r"\bHMX\b|mxmem|Q6_mx|hmx_", re.IGNORECASE)

# A harness embeds its golden vectors as base64, so its size tracks the task's
# WORKING SET, not its complexity: the largest is 93 MB for a single T3 kernel,
# and 71 of 538 account for 675 MB of an otherwise 16 MB corpus. Those are a
# build cache -- `run_batch` regenerates a whole batch in about 14 seconds -- so
# shipping them would trade the repo's usability for bytes that reproduce
# exactly. Over the cap, `HARNESS.md` records how to regenerate instead.
MAX_HARNESS_BYTES = 1_000_000

HARNESS_STUB = """# `harness.c` is not checked in for this kernel

Its golden vectors are base64-embedded, and at this task's working set that
makes the file {size:.1f} MB -- over the {cap:.0f} MB cap this corpus applies.

The harness is a **build cache, not a result**: it regenerates deterministically
from the task definition in `benchmark/selection.json`, a whole batch at a time,
in about 14 seconds.

```bash
python -m hexkernels.forge.run_batch --batch <N> --out witness_build/batch<N> --skip-verify
cp witness_build/batch<N>/{name}/harness.cpp kernels/{origin}/{name}/harness.c
```

Everything that is *not* regenerable -- the accelerated kernel, the scalar
reference, the prompt and the metadata -- is checked in beside this file.
"""

DTYPE_ALIAS = {
    "float32": "fp32", "float16": "fp16", "bfloat16": "bf16",
    "int32": "i32", "int16": "i16", "int8": "i8", "uint8": "u8",
}


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def read(path):
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        return fh.read()


def mechanisms_in_source(text):
    """What the SOURCE reaches for. Not a substitute for the ELF detector.

    This is a syntactic screen used for admission only. Whether a kernel really
    used the accelerator is decided by `hexkernels.anticheat` reading the
    disassembled ELF, and that verdict travels in `spec.json.verified`.
    """
    return {"hvx": bool(HVX_RE.search(text)), "hmx": bool(HMX_RE.search(text))}


def is_accelerated(text):
    m = mechanisms_in_source(text)
    return m["hvx"] or m["hmx"]


def witness_index(hexbench):
    """name -> witness build dir, for every task that has a scalar reference."""
    index = {}
    pattern = os.path.join(hexbench, "benchmark", "witness_build", "*", "*")
    for d in sorted(glob.glob(pattern)):
        if os.path.isdir(d) and os.path.exists(os.path.join(d, "kernel.cpp")):
            index.setdefault(os.path.basename(d), d)
    return index


def selection_index(hexbench):
    """name -> task spec, derived from the frozen corpus selection."""
    path = os.path.join(hexbench, "benchmark", "selection.json")
    if not os.path.exists(path):
        return {}
    sel = json.load(open(path, encoding="utf-8"))
    index = {}
    for batch in sel.get("batches", []):
        for task in batch:
            dt = DTYPE_ALIAS.get(task.get("dtype"), task.get("dtype"))
            name = dt + "_" + task.get("op", "")
            if task.get("overload"):
                name += "_" + task["overload"]
            index.setdefault(name, task)
    return index


def emit(out_root, origin, name, kernel_text, spec, extra_files):
    """Write one entry. `extra_files` maps destination name -> source path."""
    dest = os.path.join(out_root, origin, name)
    os.makedirs(dest, exist_ok=True)
    with open(os.path.join(dest, "kernel.c"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write(kernel_text)
    oversized = None
    for dest_name, src in extra_files.items():
        if not src or not os.path.exists(src):
            continue
        if dest_name == "harness.c" and os.path.getsize(src) > MAX_HARNESS_BYTES:
            oversized = os.path.getsize(src)
            continue
        shutil.copyfile(src, os.path.join(dest, dest_name))
    if oversized:
        with open(os.path.join(dest, "HARNESS.md"), "w",
                  encoding="utf-8", newline="\n") as fh:
            fh.write(HARNESS_STUB.format(size=oversized / 1e6,
                                         cap=MAX_HARNESS_BYTES / 1e6,
                                         name=name, origin=origin))
    present = sorted(os.listdir(dest))
    spec["files"] = present
    if {"reference.c", "harness.c"} <= set(present):
        spec["bundle"] = "complete"
    elif oversized and "reference.c" in present:
        # Everything non-regenerable is here; only the cache is missing.
        spec["bundle"] = "harness-regenerable"
        spec["harness_bytes"] = oversized
    else:
        spec["bundle"] = "kernel-only"
    spec["kernel_sha256"] = hashlib.sha256(kernel_text.encode()).hexdigest()
    with open(os.path.join(dest, "spec.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump(spec, fh, indent=2, sort_keys=True)
        fh.write("\n")
    return spec


def collect_expert(v6_root, out_root):
    """The v6 corpus: one task directory already holds the whole bundle."""
    entries = []
    for task_dir in sorted(glob.glob(os.path.join(v6_root, "tasks", "*"))):
        expert = os.path.join(task_dir, "expert.c")
        if not os.path.isdir(task_dir) or not os.path.exists(expert):
            continue
        text = read(expert)
        if not is_accelerated(text):
            continue                       # scalar expert: not a library entry
        name = os.path.basename(task_dir)
        v6_spec = {}
        spec_path = os.path.join(task_dir, "spec.json")
        if os.path.exists(spec_path):
            v6_spec = json.load(open(spec_path, encoding="utf-8"))
        spec = {
            "name": name,
            "origin": "expert",
            "entry": "candidate_kernel",
            "dtype": v6_spec.get("dtype"),
            "params": v6_spec.get("params", {}),
            "mechanisms_declared": v6_spec.get("mechanisms", []),
            "mechanisms_in_source": mechanisms_in_source(text),
            "category": v6_spec.get("category"),
            "domain": v6_spec.get("domain"),
            "level": v6_spec.get("level"),
            "signal": v6_spec.get("signal"),
            "tags": v6_spec.get("tags", []),
            "edge_cases": v6_spec.get("edge_cases", []),
            "verified": {
                # No ELF scan was run over the v6 corpus: these kernels are
                # attested by measured cycles, not by the disassembly detector.
                "elf_confirmed": None,
                # Measured on the simulator when the corpus was built.
                "baseline_kernel_cycles": v6_spec.get("baseline_kernel_cycles"),
                "expert_kernel_cycles": v6_spec.get("expert_kernel_cycles"),
                "expert_speedup": v6_spec.get("expert_speedup"),
                "source": "v6 corpus build (simulator)",
            },
            "provenance": {
                "corpus": "v6",
                "task_id": v6_spec.get("task_id", name),
                "upstream": v6_spec.get("provenance"),
            },
        }
        extra = {
            "reference.c": os.path.join(task_dir, "baseline.c"),
            "harness.c": os.path.join(task_dir, "harness.c"),
            "kernel_api.h": os.path.join(task_dir, "kernel_api.h"),
            "PROMPT.md": os.path.join(task_dir, "prompt.md"),
        }
        # Near-miss kernels: plausible WRONG implementations the harness must
        # reject. They are what makes a passing verdict mean something -- a
        # harness no wrong answer can fail is not testing anything.
        for nm in sorted(glob.glob(os.path.join(task_dir, "nearmiss_*.c"))):
            extra[os.path.basename(nm)] = nm
        entries.append(emit(out_root, "expert", name, text, spec, extra))
    return entries


def collect_mined(hexbench, out_root, witness, selection):
    """The mined candidates, paired with the witness build of the same name."""
    manifest_path = os.path.join(hexbench, "benchmark", "candidates", "MANIFEST.json")
    manifest = json.load(open(manifest_path, encoding="utf-8")) if os.path.exists(manifest_path) else {}
    entries = []
    for src in sorted(glob.glob(os.path.join(hexbench, "benchmark", "candidates", "*.cpp"))):
        name = os.path.basename(src)[:-4]
        text = read(src)
        if not is_accelerated(text):
            continue
        man = manifest.get(name, {})
        wdir = witness.get(name)
        task = selection.get(name, {})
        prov = {}
        if wdir and os.path.exists(os.path.join(wdir, "provenance.json")):
            prov = json.load(open(os.path.join(wdir, "provenance.json"), encoding="utf-8"))
        spec = {
            "name": name,
            "origin": "mined",
            "entry": "candidate_kernel",
            "dtype": DTYPE_ALIAS.get(task.get("dtype"), task.get("dtype")),
            "shape": task.get("shape"),
            "tier": man.get("task_tier") or prov.get("tier") or task.get("tier"),
            "working_set_bytes": task.get("working_set") or prov.get("working_set_bytes"),
            "mechanisms_declared": prov.get("mechanisms") or task.get("mechanisms", []),
            "mechanisms_in_source": mechanisms_in_source(text),
            "mechanism_reasons": prov.get("mechanism_reasons", []),
            "op": task.get("op"),
            "schema": task.get("schema"),
            "verified": {
                # NOT verified against this repo's task. See tier_match.
                "elf_confirmed": None,
                "source": "none",
                "note": ("authored for source_tier; the entitlement gate rejects "
                         "a kernel whose task tier differs here"),
            },
            "provenance": {
                "corpus": "mined",
                # The absolute path from the source pipeline is deliberately not
                # carried over; the content hash is the durable identifier.
                "sha256_at_import": man.get("sha256"),
                "task_tier": man.get("task_tier"),
                "source_tier": man.get("source_tier"),
                "tier_match": man.get("tier_match"),
            },
        }
        extra = {}
        if wdir:
            extra = {
                "reference.c": os.path.join(wdir, "kernel.cpp"),
                "harness.c": os.path.join(wdir, "harness.cpp"),
                "PROMPT.md": os.path.join(wdir, "prompt.md"),
            }
        entries.append(emit(out_root, "mined", name, text, spec, extra))
    return entries


def collect_model(hexbench, out_root, witness, selection):
    """Model-written kernels the ELF detector confirms reached HVX.

    One entry per task. Where several seeds produced an accelerated kernel, the
    winner is the one that is correct and whose mechanism scan fired, then the
    lowest seed -- so the choice is deterministic and re-runnable.
    """
    best = {}
    pattern = os.path.join(hexbench, "runs", "*", "*", "attempts", "*", "seed*", "candidate.cpp")
    for src in sorted(glob.glob(pattern)):
        text = read(src)
        if not is_accelerated(text):
            continue
        seed_dir = os.path.dirname(src)
        attempt_path = os.path.join(seed_dir, "attempt.json")
        if not os.path.exists(attempt_path):
            continue
        att = json.load(open(attempt_path, encoding="utf-8"))
        task = att.get("task", {})
        name = task.get("name")
        if not name:
            continue
        scan = att.get("mechanism_scan", {}) or {}
        grade = att.get("grade", {}) or {}
        correct = bool(grade.get("correct"))
        fired = bool(scan.get("mechanisms", {}).get("hvx"))
        seed = att.get("seed_index", 99)
        # rank: correct-and-fired first, then fired, then lowest seed
        rank = (0 if (correct and fired) else 1 if fired else 2, seed)
        if name not in best or rank < best[name][0]:
            best[name] = (rank, src, text, att)

    entries = []
    for name, (_, src, text, att) in sorted(best.items()):
        task = att.get("task", {})
        scan = att.get("mechanism_scan", {}) or {}
        grade = att.get("grade", {}) or {}
        model = att.get("model", {}) or {}
        wdir = witness.get(name)
        sel = selection.get(name, {})
        spec = {
            "name": name,
            "origin": "model",
            "entry": "candidate_kernel",
            "dtype": DTYPE_ALIAS.get(task.get("dtype"), task.get("dtype")),
            "shape": task.get("shape") or sel.get("shape"),
            "tier": task.get("tier"),
            "mechanisms_declared": task.get("mechanisms") or sel.get("mechanisms", []),
            "mechanisms_in_source": mechanisms_in_source(text),
            "op": task.get("op") or sel.get("op"),
            "verified": {
                # This is the anti-cheat verdict: read off the disassembled ELF.
                # `None` means no scan was run, which is NOT the same as a scan
                # that found nothing -- the three states are kept distinct.
                "elf_confirmed": (scan.get("mechanisms", {}).get("hvx")
                                  or scan.get("mechanisms", {}).get("hmx")
                                  if scan.get("scanned") else None),
                "source": "anticheat disassembly scan",
                "compiled": scan.get("compiled"),
                "correct": grade.get("correct"),
                "mechanisms_fired": scan.get("mechanisms", {}),
                "kernel_cycles": grade.get("kernel_cycles"),
                "expert_kernel_cycles": grade.get("expert_kernel_cycles"),
            },
            "provenance": {
                "corpus": "model",
                "rung": att.get("rung"),
                "eval_set": att.get("eval_set"),
                "model": model.get("resolved"),
                "seed_index": att.get("seed_index"),
                "turns_used": att.get("turns_used"),
                "task_key": task.get("key"),
                "generated_at": att.get("generated_at"),
            },
        }
        extra = {}
        if wdir:
            extra = {
                "reference.c": os.path.join(wdir, "kernel.cpp"),
                "harness.c": os.path.join(wdir, "harness.cpp"),
                "PROMPT.md": os.path.join(wdir, "prompt.md"),
            }
        entries.append(emit(out_root, "model", name, text, spec, extra))
    return entries


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hexbench", required=True)
    ap.add_argument("--v6", required=True)
    ap.add_argument("--out", default="kernels")
    args = ap.parse_args()

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    os.makedirs(args.out, exist_ok=True)

    witness = witness_index(args.hexbench)
    selection = selection_index(args.hexbench)
    print(f"witness bundles: {len(witness)}   selection tasks: {len(selection)}")

    expert = collect_expert(args.v6, args.out)
    mined = collect_mined(args.hexbench, args.out, witness, selection)
    model = collect_model(args.hexbench, args.out, witness, selection)

    index = {"expert": expert, "mined": mined, "model": model}
    counts = {k: len(v) for k, v in index.items()}
    complete = sum(1 for v in index.values() for e in v if e["bundle"] == "complete")
    regen = sum(1 for v in index.values() for e in v
                if e["bundle"] == "harness-regenerable")
    hmx = sum(1 for v in index.values() for e in v
              if e["mechanisms_in_source"]["hmx"])
    flat_all = [e for v in index.values() for e in v]
    elf = {"confirmed": 0, "refuted": 0, "unscanned": 0}
    for e in flat_all:
        state = e["verified"].get("elf_confirmed")
        elf["confirmed" if state else "unscanned" if state is None else "refuted"] += 1
    summary = {
        "counts": counts,
        "total": sum(counts.values()),
        "bundle_complete": complete,
        "bundle_harness_regenerable": regen,
        "bundle_kernel_only": sum(counts.values()) - complete - regen,
        "hmx_kernels": hmx,
        # Three states, never collapsed: the detector said yes, the detector
        # said no, or the detector was never run over this kernel.
        "elf_scan": elf,
    }
    flat = [e for v in index.values() for e in v]
    with open(os.path.join(args.out, "index.json"), "w", encoding="utf-8", newline="\n") as fh:
        json.dump({"summary": summary, "kernels": flat}, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
