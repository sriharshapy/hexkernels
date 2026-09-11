"""Run one batch of kernels through every pipeline stage, five at a time.

    python -m hexkernels.forge.run_batch --batch 1 --out run_artifacts/forge2/batch1

BOTH IRs ARE REQUIRED INPUTS, NOT ADD-ONS
----------------------------------------
This pipeline constructs kernels from **the Torch FX graph and Linalg IR**. That
is what it is for. A kernel built without either is not a product of this
pipeline, so:

  * every kernel MUST carry an FX-derived primitive graph (stage a), and
  * every kernel MUST carry real Linalg IR (stage b2).

The code path for a missing Linalg is fail-SOFT so the failure is *visible and
attributed* -- `linalg_error` names it, and the reference/harness still build so
the batch is diagnosable. It is NOT there to make Linalg skippable. A batch whose
`REPORT.md` shows Linalg coverage below 100% is not shippable; find out why.

STAGES, and what each one proves
--------------------------------
    a. trace      module -> primitive graph        REQUIRED (FX)
    b. annotate   iterator types + indexing maps, derived
    b2. linalg    the compiler's own Linalg IR     REQUIRED (torch-mlir)
    c. plan       which mechanisms the SIZE entitles this kernel to
    d. emit       portable scalar C++ -- the question, not the answer
    e. golden     run the module on seeded inputs
    f. harness    a main() that checks a candidate against those values
    g. VERIFY     the reference compiles, runs, and passes its own harness
    h. prompt     the acceleration request, written to disk

Stage (g) is the gate. If the emitted reference cannot pass the harness
generated beside it, then the emitter and the golden disagree and nothing
downstream means anything -- a candidate judged against a broken harness tells
you about the harness. So a kernel whose reference fails is reported as failed
and no prompt is written for it.

Accelerated candidates are verified by `verify_candidates`, kept separate
because producing them needs a model and this driver must stay runnable, and
testable, without one.
"""
import argparse
import hashlib
import json
import os
import shutil
import traceback

from hexkernels.core import target as _target
from hexkernels.forge.frontend.emit import emit_c, kernel_signature
from hexkernels.forge.frontend.oracle import golden, harness_c
from hexkernels.forge.frontend.schedule import annotate, graph_text
from hexkernels.forge.frontend.trace import trace
from hexkernels.forge import linalg as _linalg
from hexkernels.forge import lint as _lint
from hexkernels.forge import prompt as _prompt
from hexkernels.forge import provenance as _prov
from hexkernels.forge.kernels import batch as _batch
from hexkernels.forge.mechanism import plan_for
from hexkernels.forge import verify as _verify
from hexkernels.forge.verify import verify, work_ratio

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

STAGES = ("trace", "annotate", "plan", "emit", "golden", "harness",
          "linalg", "provenance", "verify_reference", "prompt")


def build(spec) -> dict:
    """Stages (a)-(f) plus the prompt. No toolchain needed; pure Python.

    Returns a dict with the artifacts and a `failed_stage` of None on success.
    Exceptions are captured per kernel rather than raised, so one broken kernel
    does not abort a batch -- the report is more useful than the traceback.
    """
    art = {"name": spec.name, "failed_stage": None, "error": "",
           "expect_tier": spec.expect_tier, "note": spec.note}
    try:
        art["stage"] = "trace"
        g = trace(spec.module, spec.args, spec.name)
        art["graph"] = g
        art["primitives"] = [n.target for n in g.nodes]

        art["stage"] = "annotate"
        scheds = annotate(g)
        art["schedule_text"] = graph_text(g)
        art["iterator_types"] = [list(s.iterator_types) for s in scheds]
        art["vectorizable"] = [s.vectorizable_loop() for s in scheds]

        art["stage"] = "plan"
        plan = plan_for(g, dtype_bytes=spec.dtype_bytes)
        art["plan"] = plan
        art["tier"] = plan.tier
        art["working_set_bytes"] = plan.working_set_bytes
        art["mechanisms"] = sorted(plan.mechanisms)

        art["stage"] = "emit"
        art["signature"] = kernel_signature(g)
        art["kernel_c"] = emit_c(g)

        # Carried for `forge2.lint`, which cannot derive either from the source:
        # the element total makes "declares N elements, the task has M" provable,
        # and the dtype is what says how many lanes a 128-byte vector holds (32
        # fp32, 64 fp16, 128 int8) -- dividing by the wrong one leaves the tail of
        # the buffer untouched. Taken from the FIRST INPUT rather than the output,
        # because a reduction's output is a scalar and says nothing about the
        # traversal. `None` when the graph has no ranked input, and `lint` skips
        # the rule rather than guessing.
        first_in = next((n for n in g.inputs if n.shape), None)
        art["nelem"] = None
        art["dtype"] = None
        if first_in is not None:
            n = 1
            for d in first_in.shape:
                n *= d
            art["nelem"] = n
            art["dtype"] = first_in.dtype

        art["stage"] = "golden"
        gold = golden(spec.module, spec.args, seed=0)

        art["stage"] = "harness"
        art["harness_c"] = harness_c(g, gold)

        # Real Linalg, when torch-mlir is installed. Optional by design: the
        # derived schedule carries the same iterator types and maps and also
        # answers vectorizable_loop, so a prompt is complete without it. A
        # lowering failure is recorded, never fatal -- an absent Linalg section
        # must not cost a kernel its reference.
        art["stage"] = "linalg"
        art["linalg_ir"] = None
        art["linalg_error"] = None
        if _linalg.available():
            try:
                ir = _linalg.linalg_ir(spec.module, spec.args)
                art["linalg_ir"] = _linalg.structured_only(ir)
                art["linalg_full"] = ir
            except _linalg.LinalgUnavailable as exc:
                art["linalg_error"] = str(exc)
        else:
            art["linalg_error"] = "torch-mlir not installed"

        # PROVENANCE. Every primitive must resolve to a harvested PyTorch schema
        # that passes selection, or the kernel has no origin and cannot be kept.
        # Raised, not recorded: it is caught below like any other stage failure, and
        # `main` then DESTROYS the artifacts rather than shipping them with a note.
        art["stage"] = "provenance"
        art["provenance"] = _prov.assert_proven(
            art, _prov.harvest_index(_prov.load_harvest()))

        art["stage"] = "prompt"
        art["prompt"] = _prompt.build(spec.name, art["signature"],
                                      art["kernel_c"], g, plan,
                                      linalg_ir=art["linalg_ir"])
    except Exception as exc:                      # noqa: BLE001 - reported, not raised
        art["failed_stage"] = art.get("stage")
        art["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"
    return art


# A reference gets a far longer simulator budget than a candidate, and the
# asymmetry is the point rather than a workaround.
#
# `verify`'s 900 s default is calibrated for a CANDIDATE, where exceeding it is a
# real signal: an accelerated kernel that needs a quarter-hour of simulation is a
# bad kernel. A reference is the opposite -- it is deliberately naive, so its
# runtime is a property of the TASK's size, not of anyone's work. The dense
# convolution reference here is 151M scalar MACs through a 7-deep nest with a
# bounds check per tap, and it timed out at 900 s.
#
# Conflating the two budgets does not make the reference slow, it makes the task
# UNVERIFIABLE: with no reference verdict the harness is unproven, and a candidate
# judged against an unproven harness tells you about the harness. So the reference
# is allowed to take as long as its arithmetic takes.
REFERENCE_TIMEOUT_S = 7200


def reference_key(art, timing=False, timeout=REFERENCE_TIMEOUT_S) -> str:
    """Content hash of everything a reference verdict depends on.

    Keyed on the kernel source, the harness source (which embeds the golden
    vectors) and the timing flag -- so a hit means the *identical* translation
    units were run, and editing either invalidates it automatically. There is no
    staleness window to reason about.
    """
    h = hashlib.sha256()
    for part in (art.get("kernel_c") or "", art.get("harness_c") or "",
                 f"timing={timing}", f"target={_target.current().arch}",
                 # In the key because a TIMEOUT is a cacheable verdict: raising the
                 # limit must not return the old "timed out" result forever.
                 f"timeout={timeout}"):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    return h.hexdigest()


def _cache_path(out_dir) -> str:
    return os.path.join(out_dir, ".reference_cache.json")


def load_reference_cache(out_dir) -> dict:
    try:
        with open(_cache_path(out_dir), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def save_reference_cache(out_dir, cache) -> None:
    with open(_cache_path(out_dir), "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2, sort_keys=True, default=str)


def _history_path(out_dir):
    return os.path.join(out_dir, ".candidate_history.json")


def candidate_history(out_dir) -> dict:
    try:
        with open(_history_path(out_dir), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def record_candidate_round(out_dir, cands) -> dict:
    """Append this round's candidate verdicts and return the DELTA against the last
    round whose source differed.

    WHY: a fix round has to be able to fail visibly. Measured on `fp16_mm_softmax`
    -- 10,153 of 524,288 elements wrong, "fixed" by widening before the subtract,
    which took it to 10,150. Three wrong elements out of ten thousand is not a fix,
    and it was reported as one because nothing compared the two numbers. The real
    bug was elsewhere (128-byte loads on 64-byte rows).

    Keyed on a hash of the candidate SOURCE, so re-running an unchanged candidate
    is not a "round" and cannot manufacture a delta. Only rounds where the source
    actually changed are compared.
    """
    hist = candidate_history(out_dir)
    deltas = {}
    for name, v in (cands or {}).items():
        if not v.get("present"):
            continue
        sha = v.get("source_sha")
        rounds = hist.setdefault(name, [])
        prev = next((r for r in reversed(rounds) if r.get("source_sha") != sha), None)
        now = {"source_sha": sha,
               "correct": bool(v.get("correct")),
               "compiled": bool(v.get("compiled")),
               "errors": (v.get("failure") or {}).get("errors"),
               "n": (v.get("failure") or {}).get("n"),
               "lint_errors": len(v.get("lint_errors") or [])}
        if rounds and rounds[-1].get("source_sha") == sha:
            rounds[-1] = now                       # same source: not a new round
        else:
            rounds.append(now)
        if prev is not None:
            deltas[name] = {"before": prev, "after": now}
    with open(_history_path(out_dir), "w", encoding="utf-8") as f:
        json.dump(hist, f, indent=2, sort_keys=True)
    return deltas


def verdict_delta_line(name, d) -> str:
    """One line saying whether a changed candidate actually improved.

    Says NO CHANGE IN OUTCOME when the wrong-element count moved by less than 1%,
    because that is the case the bookkeeping exists for -- a 0.03% move reported as
    a fix.
    """
    b, a = d["before"], d["after"]
    if a["correct"] and not b["correct"]:
        return f"**{name}: FIXED** (was {b['errors']} wrong, now correct)"
    if b["correct"] and not a["correct"]:
        return f"**{name}: REGRESSED** (was correct, now {a['errors']} wrong)"
    if a["correct"] and b["correct"]:
        return f"{name}: still correct after a source change"
    be, ae = b.get("errors"), a.get("errors")
    if be is None or ae is None:
        return (f"{name}: changed, compiled {b['compiled']} -> {a['compiled']}, "
                "still not correct")
    if be == 0:
        return f"{name}: changed, {be} -> {ae} wrong"
    moved = (be - ae) / be
    if abs(moved) < 0.01:
        return (f"**{name}: NO MATERIAL CHANGE** -- {be} -> {ae} wrong "
                f"({moved:+.2%}). A change this small is not a fix; the cause is "
                "very likely elsewhere")
    return f"{name}: {be} -> {ae} wrong ({moved:+.1%}), still not correct"


def verify_reference(art, timing=False, cache=None,
                     timeout=REFERENCE_TIMEOUT_S) -> dict:
    """Stage (g): does the emitted reference pass the harness generated for it?

    `cache` (content-keyed, see `reference_key`) exists because at T2/T3 sizes a
    reference is expensive to simulate -- a 1024x128 @ 128x512 scalar fp16 matmul
    is 67M MACs and took over 20 minutes -- and `run_batch` re-verifies references
    on every candidate round. A hit is marked `from_cache` and the report prints
    it, so a cached row can never be mistaken for a fresh execution.
    """
    if art["failed_stage"]:
        return {"compiled": False, "correct": False,
                "error_text": f"build failed at {art['failed_stage']}",
                "mechanisms": {}}
    key = (reference_key(art, timing=timing, timeout=timeout)
           if cache is not None else None)
    if key is not None and key in cache:
        return dict(cache[key], from_cache=True)
    r = verify(art["kernel_c"], art["harness_c"],
               name=f"{art['name']}_ref", timing=timing, timeout=timeout)
    if key is not None:
        cache[key] = dict(r, from_cache=False)
    return r


def measure_candidate_work(art, cand_source, timing=False,
                           timeout=REFERENCE_TIMEOUT_S) -> dict:
    """Run the candidate's program with the kernel called TWICE.

    The difference against the single-call run is exactly one execution of the
    kernel, which is what lets the harness be subtracted from both arms without
    modelling anything. See `verify.double_call_probe` for why the obvious
    empty-kernel probe is biased and was replaced.

    Not cached: it depends on the candidate source, which is the thing most
    likely to be edited between runs, and the run is no more expensive than
    verifying the candidate itself.
    """
    if art["failed_stage"]:
        return {}
    probe = _verify.double_call_probe(cand_source, art["signature"])
    r = _verify.verify(probe, art["harness_c"], name=f"{art['name']}_x2",
                       timing=timing, timeout=timeout)
    return {k: r.get(k) for k in ("insns", "pcycles", "compiled", "ran",
                                  "correct")}


def verify_candidates(arts, candidate_dir, timing=False) -> dict:
    """Stage (h): judge accelerated candidates found in `candidate_dir`.

    A candidate is `<candidate_dir>/<name>.cpp`. Missing files are reported as
    absent rather than as failures -- "not attempted" and "attempted and wrong"
    are different results and collapsing them would flatter the pipeline.
    """
    out = {}
    for art in arts:
        path = os.path.join(candidate_dir, f"{art['name']}.cpp")
        if not os.path.exists(path):
            out[art["name"]] = {"present": False}
            continue
        if art["failed_stage"]:
            out[art["name"]] = {"present": True, "compiled": False,
                                "correct": False,
                                "error_text": "no harness (reference build failed)"}
            continue
        with open(path, encoding="utf-8") as f:
            src = f.read()
        # Provenance before correctness. A candidate that reached into the R&D side
        # is rejected WITHOUT being compiled -- there is no verdict worth having,
        # because a correct result would only make it harder to discard.
        leaks = _prov.rd_leaks(src)
        if leaks:
            out[art["name"]] = {
                "present": True, "compiled": False, "ran": False,
                "correct": False, "rd_leak": leaks, "mechanisms": {},
                "error_text": f"REJECTED: references R&D-only material {leaks}; "
                              "not compiled. See forge2.provenance.rd_leaks."}
            continue
        # THE STATIC GATE, before the compile and long before the simulator.
        # `forge2.lint` rules each encode a mistake that actually cost a round in
        # batches 1-25 -- a stack-local DMA descriptor, a descriptor reused with
        # dstate still COMPLETE, a VTCM base outside the aperture, an intrinsic
        # that does not exist, a T0 kernel prefetching past its entitlement. All
        # of those either crash, or return plausible wrong numbers whose cause has
        # nothing to do with the arithmetic, so finding them here saves the whole
        # compile-simulate-diagnose cycle.
        #
        # ERRORS BLOCK, WARNINGS ARE CARRIED. `error` means provable from the
        # source text; `warn` means a human should look. A gate that blocks on
        # heuristics gets switched off, and a switched-off gate catches nothing --
        # so warnings ride along in the verdict and are printed in the report
        # instead of stopping the run.
        findings = _lint.lint(
            src, signature=art.get("signature"), tier=art.get("tier"),
            mechanisms=art.get("mechanisms") or (),
            nelem=art.get("nelem"), dtype=art.get("dtype"))
        blocking = _lint.errors(findings)
        warnings = [str(f) for f in findings if f.severity == "warn"]
        if blocking:
            out[art["name"]] = {
                "present": True, "compiled": False, "ran": False,
                "correct": False, "rd_leak": [], "mechanisms": {},
                "lint_errors": [str(f) for f in blocking],
                "lint_warnings": warnings,
                "source_sha": hashlib.sha256(src.encode("utf-8")).hexdigest()[:16],
                "error_text": "REJECTED by the static gate, not compiled:\n"
                              + _lint.format_findings(blocking)}
            continue
        v = verify(src, art["harness_c"], name=f"{art['name']}_cand", timing=timing)
        v["present"] = True
        v["rd_leak"] = []
        v["lint_errors"] = []
        v["lint_warnings"] = warnings
        v["source_sha"] = hashlib.sha256(src.encode("utf-8")).hexdigest()[:16]
        out[art["name"]] = v
    return out


def destroy_artifacts(art, out_dir) -> str:
    """Remove a kernel's directory, candidate and cached verdict.

    Called when provenance fails. Deleting rather than flagging is the point: a
    kernel with no chain back to PyTorch is unusable however correct it is, and a
    flagged-but-present artifact is one careless `--candidates` run away from being
    counted. There is nothing to salvage, so nothing is kept.
    """
    d = os.path.join(out_dir, art["name"])
    if os.path.isdir(d):
        shutil.rmtree(d, ignore_errors=True)
    cand = os.path.join(out_dir, "candidates", f"{art['name']}.cpp")
    if os.path.exists(cand):
        os.remove(cand)
    return d


def _write_artifacts(art, out_dir):
    d = os.path.join(out_dir, art["name"])
    os.makedirs(d, exist_ok=True)
    files = {"kernel.cpp": art.get("kernel_c"),
             "harness.cpp": art.get("harness_c"),
             "schedule.txt": art.get("schedule_text"),
             "linalg.mlir": art.get("linalg_full"),
             "prompt.md": art.get("prompt")}
    if art.get("provenance"):
        files["PROVENANCE.md"] = _prov.render(art["provenance"])
        with open(os.path.join(d, "provenance.json"), "w", encoding="utf-8") as f:
            json.dump(art["provenance"], f, indent=2, sort_keys=True)
    for fn, text in files.items():
        if text:
            with open(os.path.join(d, fn), "w", encoding="utf-8") as f:
                f.write(text)
    if art.get("plan") is not None:
        with open(os.path.join(d, "plan.txt"), "w", encoding="utf-8") as f:
            f.write(art["plan"].explain() + "\n")
    return d


def _num(v) -> str:
    """Thousands-separated, or "-" when the value was not measured.

    Exists because three of the work table's cells are absent whenever the
    harness baseline could not be subtracted, and formatting a missing one with
    `:,` raises `ValueError: Cannot specify ',' with 's'` -- which it did, in the
    middle of a seven-batch re-run, after the first two batches had already been
    written.
    """
    return f"{v:,}" if isinstance(v, int) else "-"


def render_report(arts, refs, cands=None, destroyed=(), work=None,
                  deltas=None) -> str:
    lines = ["# Batch report", "",
             "A kernel counts only when it COMPILED, RAN on the simulator, and "
             "PASSED its own generated golden harness.", ""]
    # Origin first, because it is the precondition for everything else meaning
    # anything: a kernel with no chain back to PyTorch cannot enter the corpus, so
    # its correctness is not a result.
    lines += ["## Origin", "",
              "Every kernel below resolves each traced primitive to a harvested "
              "PyTorch schema (`torch._C._jit_get_all_schemas()`); the per-kernel "
              "chain is in `<kernel>/PROVENANCE.md`. Anything that did not resolve "
              "was DESTROYED and is not listed as a result.", ""]
    if destroyed:
        lines += [f"**DESTROYED for want of a PyTorch origin: {len(destroyed)}**", ""]
        lines += [f"- `{n}` — {w}" for n, w in destroyed] + [""]
    else:
        lines += ["**Destroyed for want of a PyTorch origin: 0**", ""]
    prov_n = sum(1 for a in arts if (a.get("provenance") or {}).get("proven"))
    lines += [f"**Provenance — {prov_n}/{len(arts)} kernels traced to a PyTorch "
              "schema**", ""]

    # SAFEGUARD 2 OF THE PRIMITIVE/SELECTION SPLIT (2026-08-05).
    #
    # A primitive inside a kernel's graph now needs only to BE a harvested PyTorch op,
    # not to be independently kernel-worthy -- which is what makes attention buildable
    # (17 primitives, 2 of them `permute` and `full_like`). The risk of that relaxation
    # is a corpus quietly filling with plumbing, so the ratio is printed for every
    # kernel rather than left to be discovered.
    #
    # It is a REPORT, not a gate: there is no defensible threshold, and a kernel whose
    # decomposition happens to be shape-heavy is not thereby wrong. What would be
    # wrong is not knowing.
    prov = [(a["name"], (a.get("provenance") or {})) for a in arts]
    prov = [(n, r) for n, r in prov if r.get("n_primitives")]
    if prov:
        tot_p = sum(r["n_primitives"] for _n, r in prov)
        tot_b = sum(r.get("n_plumbing", 0) for _n, r in prov)
        lines += ["### Primitive composition", "",
                  "Every primitive resolves to a harvested PyTorch schema. "
                  "`plumbing` counts those that are not independently "
                  "kernel-worthy — shape, metadata and constant-materialising ops "
                  "(`permute`, `view`, `clone`, `scalar_tensor`) that the SELECTED "
                  "op decomposes through. They are admissible inside a graph and "
                  "can never themselves be selected as a task; this column exists "
                  "so a corpus drifting toward plumbing is visible.", "",
                  "| kernel | primitives | plumbing |", "|---|---|---|"]
        for n, r in prov:
            lines.append(f"| {n} | {r['n_primitives']} | "
                         f"{r.get('n_plumbing', 0)} |")
        lines += ["", f"**{tot_b}/{tot_p} primitives are plumbing "
                      f"({tot_b / tot_p:.0%})**", ""]
    lines += ["## Reference (the question)", "",
             "| kernel | tier | working set | prims | compiled | ran | correct | "
             "mechanisms |", "|---|---|---|---|---|---|---|---|"]
    for a in arts:
        r = refs[a["name"]]
        mech = ",".join(k for k, v in (r.get("mechanisms") or {}).items() if v) or "-"
        lines.append(
            f"| {a['name']} | {a.get('tier','-')} | "
            f"{a.get('working_set_bytes','-')} | "
            f"{len(a.get('primitives') or [])} | {r.get('compiled')} | "
            f"{r.get('ran')} | {r.get('correct')} | {mech} |")

    ok = sum(1 for a in arts if refs[a["name"]].get("correct"))
    lines += ["", f"**References passing their own harness: {ok}/{len(arts)}**", ""]
    cached = [a["name"] for a in arts if refs[a["name"]].get("from_cache")]
    if cached:
        lines += [f"> {len(cached)} of these verdicts were REUSED from a previous "
                  f"run of the byte-identical kernel and harness "
                  f"({', '.join(cached)}); they were not re-simulated for this "
                  "report. Re-run with `--no-cache` to force execution.", ""]

    # Both IRs are required inputs to this pipeline, so their coverage is a
    # headline number rather than a footnote. FX comes from the trace stage, so a
    # kernel with primitives has it by construction; Linalg depends on
    # torch-mlir, so it is the one that can silently go missing.
    fx_n = sum(1 for a in arts if a.get("primitives"))
    lin_n = sum(1 for a in arts if a.get("linalg_ir"))
    lines += [f"**IR coverage — FX {fx_n}/{len(arts)}, "
              f"Linalg {lin_n}/{len(arts)}**", ""]
    if lin_n < len(arts):
        lines += ["> **Linalg coverage is short, so this batch is not shippable.** "
                  "Both the FX graph and Linalg IR are required inputs to this "
                  "pipeline; a kernel missing either was not built the way the "
                  "pipeline claims. Reasons below.", ""]
        for a in arts:
            if not a.get("linalg_ir"):
                lines.append(f"- `{a['name']}`: {a.get('linalg_error') or 'unknown'}")
        lines.append("")

    fails = [(a["name"], a["failed_stage"] or "verify",
              (a["error"] or refs[a["name"]].get("error_text", ""))[:400])
             for a in arts if a["failed_stage"] or not refs[a["name"]].get("correct")]
    if fails:
        lines += ["### Failures", ""]
        for n, stage, err in fails:
            lines += [f"**{n}** — failed at `{stage}`", "", "```", err.strip(), "```", ""]

    lines += ["## Entitled mechanisms (derived from size)", "",
              "| kernel | tier | mechanisms |", "|---|---|---|"]
    for a in arts:
        lines.append(f"| {a['name']} | {a.get('tier','-')} | "
                     f"{','.join(a.get('mechanisms') or []) or '-'} |")

    if cands:
        lines += ["", "## Accelerated candidates", "",
                  "| kernel | present | compiled | correct | mechanisms |",
                  "|---|---|---|---|---|"]
        for a in arts:
            c = cands.get(a["name"], {})
            mech = ",".join(k for k, v in (c.get("mechanisms") or {}).items() if v) or "-"
            lines.append(
                f"| {a['name']} | {c.get('present', False)} | "
                f"{c.get('compiled','-')} | {c.get('correct','-')} | {mech} |")
        # THE STATIC GATE'S OUTPUT BELONGS IN THE REPORT, both halves. Errors mean
        # the candidate was never compiled, which the table above shows as
        # compiled=False and would otherwise be indistinguishable from a compiler
        # error. Warnings did not block, and a warning nobody reads is a warning
        # that may as well not fire -- this is where they get read.
        gated = [(a["name"], cands.get(a["name"], {}).get("lint_errors") or [])
                 for a in arts]
        gated = [(n, e) for n, e in gated if e]
        if gated:
            lines += ["", "### Rejected by the static gate, before compiling", "",
                      "`forge2.lint` rules are each a mistake that cost a round in "
                      "batches 1-25. A candidate below was NOT compiled and NOT "
                      "simulated, so its `compiled=False` above is this, not a "
                      "compiler error.", ""]
            for n, errs in gated:
                lines += [f"* **{n}**"] + [f"  * {e}" for e in errs]
        warned = [(a["name"], cands.get(a["name"], {}).get("lint_warnings") or [])
                  for a in arts]
        warned = [(n, w) for n, w in warned if w]
        if warned:
            lines += ["", "### Static-gate warnings (did not block)", "",
                      "Patterns worth a look that are not provable from the source "
                      "alone. A gate that blocked on these would get switched off.",
                      ""]
            for n, ws in warned:
                lines += [f"* **{n}**"] + [f"  * {w}" for w in ws]
        if deltas:
            # ROUND OVER ROUND. `fp16_mm_softmax` was once reported fixed after a
            # change that moved 10,153 wrong elements to 10,150 -- three in ten
            # thousand. Nothing compared the two numbers, so nothing contradicted
            # the claim. This does.
            lines += ["", "### Change since the last round", "",
                      "Only candidates whose SOURCE changed appear here; "
                      "re-running an unchanged candidate is not a round.", ""]
            for n in sorted(deltas):
                lines.append("* " + verdict_delta_line(n, deltas[n]))
        if work:
            lines += ["", "### Work done, with the harness subtracted", "",
                      "The raw reference/candidate ratio is NOT the speedup: both "
                      "runs execute the same harness, which decodes a base64 image "
                      "of every golden buffer and compares it element by element, "
                      "and at these sizes that dwarfs the kernel. `harness` is "
                      "subtracted using a third run of the identical program with "
                      "the candidate called TWICE, whose difference against the "
                      "single-call run is exactly one execution of the kernel -- so "
                      "both columns are measured, not modelled. Instructions "
                      "retired: no memory model and no calibration, so this is not "
                      "a cycle claim.", "",
                      "| kernel | harness | ref work | cand work | kernel ratio | "
                      "raw ratio |",
                      "|---|---|---|---|---|---|"]
            for a in arts:
                w = work.get(a["name"])
                if not w:
                    continue
                # Every cell is formatted through `_num`, which returns "-" for a
                # missing value. Three of these keys are absent whenever the
                # baseline probe could not be subtracted (it did not run, or it
                # measured at least as much as one of the real runs), and an
                # inline conditional over the whole row got that wrong once --
                # `ValueError: Cannot specify ',' with 's'` mid-batch, after the
                # earlier kernels had already been reported.
                kr = w.get("kernel_ratio")
                kr_cell = f"**{kr}x**" if kr is not None else "-"
                lines.append(
                    f"| {a['name']} | {_num(w.get('harness_insns'))} | "
                    f"{_num(w.get('ref_work'))} | {_num(w.get('cand_work'))} | "
                    f"{kr_cell} | {w['ratio']}x |")
        cok = sum(1 for a in arts if cands.get(a["name"], {}).get("correct"))
        cgen = sum(1 for a in arts
                   if any((cands.get(a["name"], {}).get("mechanisms") or {}).values()))
        lines += ["", f"**Candidates correct: {cok}/{len(arts)}; "
                      f"using at least one mechanism: {cgen}/{len(arts)}**", ""]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch", type=int, default=1)
    # DERIVED FROM --batch, NOT A FIXED DEFAULT.
    #
    # This used to default to `run_artifacts/forge2/batch1` regardless of --batch,
    # so `--batch 16` with --out forgotten overwrote BATCH 1's REPORT.md,
    # results.json and reference cache, and dropped batch 16's ten kernel
    # directories into batch1/. It happened, twice in one session, and was
    # recovered only because the artifacts are in git. A default that silently
    # destroys another batch's results is a footgun with no upside: the only
    # sensible destination for `--batch N` is batch N's own directory.
    ap.add_argument("--out", default=None,
                    help="artifact directory; defaults to "
                         "run_artifacts/forge2/batch<BATCH>")
    ap.add_argument("--candidates", default=None,
                    help="directory of <name>.cpp accelerated candidates")
    ap.add_argument("--timing", action="store_true")
    ap.add_argument("--skip-verify", action="store_true",
                    help="build artifacts only; no toolchain needed")
    ap.add_argument("--reference-timeout", type=int,
                    default=REFERENCE_TIMEOUT_S,
                    help="simulator budget for a REFERENCE, which is naive by "
                         "design and so slow in proportion to the task's size "
                         "(candidates keep verify()'s tighter default)")
    ap.add_argument("--no-baseline", action="store_true",
                    help="skip the double-call work probe; instruction ratios "
                         "then include the harness and read near 1")
    ap.add_argument("--no-cache", action="store_true",
                    help="re-simulate references even when the identical kernel "
                         "and harness were already verified (see reference_key)")
    args = ap.parse_args(argv)
    if args.out is None:
        args.out = os.path.join(REPO, "benchmark", f"batch{args.batch}")

    specs = _batch(args.batch)
    os.makedirs(args.out, exist_ok=True)

    cache = None if args.no_cache else load_reference_cache(args.out)
    arts, refs, destroyed, bases, deltas = [], {}, [], {}, {}
    for spec in specs:
        art = build(spec)
        # ORIGIN OR NOTHING. A kernel whose primitives do not resolve to a
        # harvested PyTorch schema is DESTROYED, not reported with a caveat: it
        # cannot enter the corpus however correct it is, and an artifact left on
        # disk is one careless `--candidates` run away from being counted.
        if art["failed_stage"] == "provenance":
            gone = destroy_artifacts(art, args.out)
            destroyed.append((spec.name, art["error"].splitlines()[0]))
            print(f"  {spec.name:20s} DESTROYED (no PyTorch origin) -> {gone}",
                  flush=True)
            continue
        arts.append(art)
        _write_artifacts(art, args.out)
        if args.skip_verify:
            refs[spec.name] = {"compiled": None, "ran": None, "correct": None,
                               "mechanisms": {}}
        else:
            refs[spec.name] = verify_reference(art, timing=args.timing,
                                               cache=cache,
                                               timeout=args.reference_timeout)
            if cache is not None:
                save_reference_cache(args.out, cache)
        r = refs[spec.name]
        print(f"  {spec.name:20s} tier={art.get('tier','-'):3s} "
              f"prims={len(art.get('primitives') or []):2d} "
              f"ref_correct={r.get('correct')}"
              f"{' (cached)' if r.get('from_cache') else ''}", flush=True)

    cands = None
    if args.candidates:
        cands = verify_candidates(arts, args.candidates, timing=args.timing)
        # Round-over-round bookkeeping, so a fix round can fail VISIBLY. Keyed on
        # the candidate source hash: re-running an unchanged candidate is not a new
        # round and produces no delta.
        deltas = record_candidate_round(args.out, cands)
        if not args.no_baseline:
            for a in arts:
                path = os.path.join(args.candidates, f"{a['name']}.cpp")
                if not (cands.get(a["name"], {}).get("correct")
                        and os.path.exists(path)):
                    continue
                with open(path, encoding="utf-8") as f:
                    bases[a["name"]] = measure_candidate_work(
                        a, f.read(), timing=args.timing,
                        timeout=args.reference_timeout)

    work = {}
    for a in arts:
        w = work_ratio(refs.get(a["name"], {}),
                       (cands or {}).get(a["name"], {}),
                       probe=bases.get(a["name"]))
        if w:
            work[a["name"]] = w
    report = render_report(arts, refs, cands, destroyed=destroyed, work=work,
                           deltas=deltas)
    with open(os.path.join(args.out, "REPORT.md"), "w", encoding="utf-8") as f:
        f.write(report)
    with open(os.path.join(args.out, "results.json"), "w", encoding="utf-8") as f:
        json.dump({"reference": refs, "candidates": cands,
                   "work_probe": bases, "work": work}, f, indent=2,
                  sort_keys=True, default=str)
    print(f"\nwrote {args.out}/REPORT.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
