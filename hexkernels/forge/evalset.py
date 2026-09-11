"""Resolve `benchmark/eval_core.json` into the tasks a rung actually runs.

`eval_core.json` names its 128 tasks by IDENTITY (`op.overload.dtype.tier`, plus
the fused epilogue), because that is what survives a re-mine. The pipeline, on the
other hand, reaches a task by POSITION: `hexkernels.forge.kernels.batch(N)[i]`. Nothing
connected the two, so every consumer re-derived the mapping by hand -- and a
hand-derived version of it matched only 91 of the 128 ids, because it dropped the
fused-epilogue suffix that `testset.task_id` appends. This module exists so that
mapping is written once and asserted.

WHY POSITION IS THE KEY AND THE NAME IS NOT
-------------------------------------------
Kernel names are NOT unique across the 320. `mined._load` names a kernel
`<dtype-prefix>_<op>` with no tier in it, so the same op at T0 and at T1 -- which
are deliberately different tasks, one of them often the negative control -- are
both called e.g. `fp16_linear`. Measured on the frozen set: 320 specs carry 298
distinct task ids and the 165 built references carry only 158 distinct names.

So a record keyed on the name silently overwrites its sibling. Every task here
therefore carries `(batch, index)`, which is unique by construction, and the name
and id are recorded beside it as labels rather than used as keys.

RESOLUTION IS ORDERED AND DUPLICATES ARE REPORTED, NOT PICKED SILENTLY. Where one
id matches several specs the first in selection order wins, and `duplicates` names
every id that happened -- an id resolving two ways is a fact about the frozen set
that a caller may want to refuse, not something this module should hide.
"""
import json
import os
import pathlib

from hexkernels.forge import testset

REPO = pathlib.Path(__file__).resolve().parents[1]
BENCH = REPO / "benchmark"
CORE = BENCH / "eval_core.json"
SELECTION = BENCH / "selection.json"
WITNESS = BENCH / "witness_build"


class TaskRef:
    """One evaluable task: where it lives, what it is, and what it is called.

    `batch`/`index` locate it in the pipeline (`kernels.batch(batch)[index]`).
    `task_id` is its identity in `eval_core.json`. `name` is the label the
    artifacts on disk use, and is not unique -- see the module docstring.
    """

    __slots__ = ("batch", "index", "name", "task_id", "tier", "dtype",
                 "mechanisms", "shape", "entry")

    def __init__(self, batch, index, name, task_id, entry):
        self.batch = batch
        self.index = index
        self.name = name
        self.task_id = task_id
        self.tier = entry["tier"]
        self.dtype = entry["dtype"]
        self.mechanisms = tuple(entry["mechanisms"])
        self.shape = tuple(entry["shape"])
        self.entry = entry

    @property
    def key(self) -> str:
        """Filesystem- and JSON-safe unique key: `b<batch>i<index>_<name>`."""
        return f"b{self.batch}i{self.index}_{self.name}"

    def __repr__(self):
        return f"<TaskRef {self.key} {self.task_id}>"


def _kernel_name(entry) -> str:
    """The name `mined._load` will give this entry.

    Duplicated from `mined._load` deliberately: that function builds torch
    modules and draws golden inputs, which costs seconds per spec and needs torch
    installed. Resolving 128 ids should need neither, so the naming rule -- three
    lines and frozen with the selection -- is repeated here and
    `test_evalset_names_match_mined` asserts the two never diverge.
    """
    pre = entry["dtype"].replace("float", "fp").replace("int", "i")
    if entry.get("stages"):
        return pre + "_" + "__".join(op for op, _ov in entry["stages"])
    ov = entry["overload"]
    return f"{pre}_{entry['op']}" + (f"_{ov}" if ov else "")


def all_tasks(selection_path=SELECTION) -> list:
    """Every one of the 320, in selection order, as TaskRefs."""
    with open(selection_path, encoding="utf-8") as f:
        doc = json.load(f)
    first = doc["first_batch"]
    out = []
    for bi, chunk in enumerate(doc["batches"]):
        for ki, entry in enumerate(chunk):
            out.append(TaskRef(first + bi, ki, _kernel_name(entry),
                               testset.task_id(entry), entry))
    return out


def core_tasks(core_path=CORE, selection_path=SELECTION):
    """The eval-core tasks as TaskRefs, plus what did not resolve.

    Returns `(tasks, report)`. `report` carries `missing` (core ids absent from
    the selection -- must be empty on the frozen set) and `duplicates` (core ids
    matching more than one spec, where the first was taken).
    """
    with open(core_path, encoding="utf-8") as f:
        core_ids = list(json.load(f)["task_ids"])
    by_id = {}
    duplicates = []
    for ref in all_tasks(selection_path):
        if ref.task_id in by_id:
            duplicates.append(ref.task_id)
        else:
            by_id[ref.task_id] = ref
    tasks, missing = [], []
    for tid in core_ids:
        ref = by_id.get(tid)
        if ref is None:
            missing.append(tid)
        else:
            tasks.append(ref)
    dupes = sorted({d for d in duplicates if d in set(core_ids)})
    return tasks, {"requested": len(core_ids), "resolved": len(tasks),
                   "missing": missing, "duplicates": dupes}


def reference_status(ref, witness_root=WITNESS) -> dict:
    """Has this task's REFERENCE passed stage (g), and where is its harness?

    A candidate is graded against `harness.cpp`, which embeds the golden vectors.
    That harness is only trustworthy once the reference has passed it -- if the
    emitter and the golden disagree, a candidate's verdict is about the harness
    and not about the candidate (`run_batch`'s stage (g) exists to say so). So a
    task whose reference is unbuilt or failing is reported `verified: False` and
    a caller must record its attempts as ungraded rather than judge them.
    """
    bdir = os.path.join(str(witness_root), f"batch{ref.batch}")
    res = os.path.join(bdir, "results.json")
    kdir = os.path.join(bdir, ref.name)
    harness = os.path.join(kdir, "harness.cpp")
    out = {"verified": False, "reason": "", "dir": kdir,
           "harness": harness if os.path.exists(harness) else None,
           "results_json": res if os.path.exists(res) else None}
    if not os.path.exists(res):
        out["reason"] = "reference not built (no results.json for this batch)"
        return out
    with open(res, encoding="utf-8") as f:
        refs = json.load(f).get("reference") or {}
    verdict = refs.get(ref.name)
    if verdict is None:
        out["reason"] = "batch built but this kernel is absent from it"
        return out
    if not verdict.get("correct"):
        out["reason"] = "reference did not pass its own harness"
        return out
    if out["harness"] is None:
        out["reason"] = "reference passed but harness.cpp is missing on disk"
        return out
    out["verified"] = True
    out["timing"] = bool(verdict.get("timing"))
    out["ref_insns"] = verdict.get("insns")
    out["ref_pcycles"] = verdict.get("pcycles")
    return out


def summarise(tasks=None, witness_root=WITNESS) -> dict:
    """Counts of gradable vs pending, per tier -- what a run can report today."""
    if tasks is None:
        tasks, _ = core_tasks()
    per_tier = {}
    for ref in tasks:
        st = reference_status(ref, witness_root)
        row = per_tier.setdefault(ref.tier, {"total": 0, "gradable": 0})
        row["total"] += 1
        row["gradable"] += 1 if st["verified"] else 0
    return {"n": len(tasks), "per_tier": per_tier,
            "gradable": sum(r["gradable"] for r in per_tier.values())}


def main(argv=None) -> int:
    """Report the eval core's resolution and grading readiness."""
    import argparse
    ap = argparse.ArgumentParser(description=main.__doc__)
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)

    tasks, report = core_tasks()
    summary = summarise(tasks)
    if args.json:
        print(json.dumps({"report": report, "summary": summary,
                          "tasks": [{"key": t.key, "task_id": t.task_id,
                                     "tier": t.tier,
                                     "gradable": reference_status(t)["verified"]}
                                    for t in tasks]}, indent=1))
        return 0
    print(f"eval core: {report['resolved']}/{report['requested']} ids resolved")
    if report["missing"]:
        print(f"  MISSING {len(report['missing'])}: {report['missing'][:5]}")
    if report["duplicates"]:
        print(f"  ids matching >1 spec (first taken): {report['duplicates']}")
    print(f"gradable now (reference passed stage g): {summary['gradable']}"
          f"/{summary['n']}")
    for tier in sorted(summary["per_tier"]):
        row = summary["per_tier"][tier]
        print(f"  {tier}: {row['gradable']:3d}/{row['total']:3d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
