"""Turn attempt records into the numbers a rung reports.

Pure: takes the records, returns a dict, renders text. No I/O, no simulator, no
network -- so the summary can be recomputed from `attempts.jsonl` at any time and a
disagreement between two runs of this module is a bug in this module.

WHAT IT REFUSES TO DO, and why each refusal is here rather than left to a reader:

  * IT WILL NOT POOL TWO EVAL SETS. PLAN.md section 3 budgets silicon on 64 tasks
    while `eval_core.json` holds 128, so two values of n are in play. Averaging a
    rate over a mixture of them produces a number belonging to neither, so
    `summarise` raises instead. Same for two rungs or two schema versions.
  * IT WILL NOT COUNT AN UNGRADED ATTEMPT AS A FAILURE. An attempt whose reference
    is unbuilt has no verdict, and folding it in as `correct: false` would report
    the reference build's progress as the model's competence. Rates are over
    GRADED attempts and the ungraded count is reported beside them.
  * IT WILL NOT REPORT A SINGLE-SEED RATE AS THE RESULT. PLAN.md section 2: "a
    single-seed number is not a result." The per-seed spread is reported next to
    the pooled rate so a reader sees the variance rather than having to ask.

THE HEADLINE IS `genuine`, NOT `correct`. Correct-but-scalar is the failure this
benchmark exists to make visible, so a summary that led with correctness would be
reporting the very number the thesis says is misleading.
"""
import statistics

TIERS = ("T0", "T1", "T2", "T3")
MECHANISMS = ("hvx", "hmx", "dma", "vtcm", "l2fetch")


def _rate(num, den):
    return None if not den else round(num / den, 4)


def _homogeneous(recs):
    """One eval set, one rung, one schema version -- or refuse."""
    for field in ("eval_set", "rung", "schema_version"):
        seen = sorted({str(r.get(field)) for r in recs})
        if len(seen) > 1:
            raise ValueError(
                f"records mix {field}={seen}; refusing to aggregate across them")


def summarise(recs) -> dict:
    """Every reported number for one model at one rung."""
    if not recs:
        return {"n_attempts": 0}
    _homogeneous(recs)

    graded = [r for r in recs if (r.get("grade") or {}).get("graded")]
    ungraded = [r for r in recs if not (r.get("grade") or {}).get("graded")]
    gen_failed = [r for r in recs if r.get("generation_error")]

    def count(rows, field):
        return sum(1 for r in rows if (r.get("grade") or {}).get(field))

    tasks = {r["task"]["key"] for r in recs}
    seeds = sorted({r["seed_index"] for r in recs})

    # PER-SEED, so the spread is visible. Each seed is one complete pass over the
    # graded tasks, which is the unit PLAN.md's "-7 to +3 tasks" was measured in.
    per_seed = {}
    for s in seeds:
        rows = [r for r in graded if r["seed_index"] == s]
        per_seed[s] = {"graded": len(rows), "compiled": count(rows, "compiled"),
                       "correct": count(rows, "correct"),
                       "genuine": count(rows, "genuine")}

    per_tier = {}
    for tier in TIERS:
        rows = [r for r in graded if r["task"]["tier"] == tier]
        allrows = [r for r in recs if r["task"]["tier"] == tier]
        if not allrows:
            continue
        per_tier[tier] = {
            "attempts": len(allrows), "graded": len(rows),
            "compiled_rate": _rate(count(rows, "compiled"), len(rows)),
            "correct_rate": _rate(count(rows, "correct"), len(rows)),
            "genuine_rate": _rate(count(rows, "genuine"), len(rows)),
        }

    # MECHANISM ENGAGEMENT, over attempts whose TASK was entitled to that
    # mechanism. Denominator is entitlement, not the whole set: a kernel too small
    # to justify DMA cannot fail to use DMA.
    per_mech = {}
    for mech in MECHANISMS:
        rows = [r for r in graded
                if mech in (r["task"].get("entitled_mechanisms") or [])]
        if not rows:
            continue
        fired = sum(1 for r in rows
                    if mech in (r["grade"].get("mechanisms_fired") or []))
        with_correct = sum(1 for r in rows if r["grade"].get("correct")
                           and mech in (r["grade"].get("mechanisms_fired") or []))
        per_mech[mech] = {"entitled_attempts": len(rows), "fired": fired,
                          "fired_rate": _rate(fired, len(rows)),
                          "correct_and_fired": with_correct}

    # PER TASK, pooled over seeds: how many tasks were EVER solved, and how many
    # were solved every time. The gap between them is the sampling variance the
    # plan requires 5 seeds to expose.
    by_task = {}
    for r in graded:
        by_task.setdefault(r["task"]["key"], []).append(r["grade"])
    ever_correct = sum(1 for v in by_task.values() if any(g.get("correct") for g in v))
    always_correct = sum(1 for v in by_task.values()
                         if v and all(g.get("correct") for g in v))
    ever_genuine = sum(1 for v in by_task.values() if any(g.get("genuine") for g in v))

    sims = [r["grade"]["scalar_similarity"] for r in graded
            if r["grade"].get("scalar_similarity") is not None]
    halluc = [r for r in graded if r["grade"].get("hallucinated_intrinsics")]
    costs = [r["usage"]["cost_usd"] for r in recs
             if (r.get("usage") or {}).get("cost_usd") is not None]
    toks_in = sum((r.get("usage") or {}).get("prompt_tokens") or 0 for r in recs)
    toks_out = sum((r.get("usage") or {}).get("completion_tokens") or 0 for r in recs)

    return {
        "eval_set": recs[0]["eval_set"],
        "rung": recs[0]["rung"],
        "schema_version": recs[0]["schema_version"],
        "model": (recs[0].get("model") or {}).get("requested"),
        "model_resolved": sorted({(r.get("model") or {}).get("resolved")
                                  for r in recs} - {None}),
        "n_tasks": len(tasks),
        "seeds": seeds,
        "n_attempts": len(recs),
        "generation_failed": len(gen_failed),
        "graded": len(graded),
        "ungraded": len(ungraded),
        "ungraded_reasons": _reasons(ungraded),
        "pooled": {
            "compiled_rate": _rate(count(graded, "compiled"), len(graded)),
            "correct_rate": _rate(count(graded, "correct"), len(graded)),
            "genuine_rate": _rate(count(graded, "genuine"), len(graded)),
            "compiled": count(graded, "compiled"),
            "correct": count(graded, "correct"),
            "genuine": count(graded, "genuine"),
        },
        "per_seed": per_seed,
        "seed_spread": _spread(per_seed),
        "per_tier": per_tier,
        "per_mechanism": per_mech,
        "per_task": {"n": len(by_task), "ever_correct": ever_correct,
                     "always_correct": always_correct,
                     "ever_genuine": ever_genuine},
        "scalar_similarity": {
            "n": len(sims),
            "median": round(statistics.median(sims), 4) if sims else None,
            "max": max(sims) if sims else None},
        "hallucinated_intrinsics": {
            "attempts": len(halluc),
            "rate": _rate(len(halluc), len(graded)),
            "distinct": len({n for r in halluc
                             for n in r["grade"]["hallucinated_intrinsics"]})},
        "usage": {"prompt_tokens": toks_in, "completion_tokens": toks_out,
                  "cost_usd": round(sum(costs), 4) if costs else None,
                  "attempts_priced": len(costs)},
        "mechanism_scan": mechanism_scan(recs),
        "scan_verdict_disagreements": scan_verdict_disagreements(recs),
    }


def mechanism_scan(recs) -> dict:
    """The mechanism half of the thesis, over EVERY attempt that was scanned.

    Reported separately from the graded rates because it has a different, larger
    denominator and a weaker claim. A scan costs ~0.4 s (compile + objdump) against
    ~12 minutes for a T2 correctness run, so this reaches the whole eval set long
    before the simulator does -- and `genuine = correct AND fired`, so every attempt
    with nothing firing has `genuine` settled as False without being executed.

    `genuine_ruled_out` is therefore a SOUND lower bound on the number of non-genuine
    attempts. The complement is not a bound on genuine ones: firing says nothing
    about correctness.
    """
    scanned = [r for r in recs if (r.get("mechanism_scan") or {}).get("scanned")]
    if not scanned:
        return {}
    unscannable = [r for r in recs if r.get("mechanism_scan")
                   and not r["mechanism_scan"].get("scanned")]
    fired = [r for r in scanned if r["mechanism_scan"].get("mechanisms_fired")]
    per_tier, per_mech = {}, {}
    for tier in TIERS:
        rows = [r for r in scanned if r["task"]["tier"] == tier]
        if rows:
            per_tier[tier] = {
                "scanned": len(rows),
                "fired": sum(1 for r in rows
                             if r["mechanism_scan"]["mechanisms_fired"]),
                "any_q6": sum(1 for r in rows
                              if r["mechanism_scan"].get("uses_any_q6_intrinsic"))}
    for mech in MECHANISMS:
        rows = [r for r in scanned
                if mech in (r["task"].get("entitled_mechanisms") or [])]
        if rows:
            per_mech[mech] = {
                "entitled": len(rows),
                "fired": sum(1 for r in rows if mech in
                             r["mechanism_scan"]["mechanisms_fired"])}
    # WHO PUT THE VECTOR CODE THERE -- the model, or the compiler?
    #
    # The anti-cheat reads the binary, which is what makes it trustworthy: it cannot
    # be fooled by a mention of an intrinsic in a comment. But it also cannot tell
    # WHO caused the instruction to exist. Measured at rung 0: every attempt that
    # fired a mechanism contains no `Q6_` intrinsic anywhere in its source, so
    # hexagon-clang auto-vectorised a scalar loop and the flag fired without the
    # model ever asking for the accelerator.
    #
    # That distinction is the difference between "the model used the hardware" and
    # "the model's scalar code happened to be auto-vectorisable", and the paper's
    # claim is about the first. Split rather than merged, because merging them would
    # let the compiler's work be reported as the model's.
    by_intrinsic = [r for r in fired
                    if r["mechanism_scan"].get("uses_any_q6_intrinsic")]
    return {
        "scanned": len(scanned),
        "unscannable": len(unscannable),
        "fired": len(fired),
        "fired_rate": _rate(len(fired), len(scanned)),
        "fired_with_model_written_intrinsics": len(by_intrinsic),
        "fired_by_compiler_autovectorisation": len(fired) - len(by_intrinsic),
        "genuine_ruled_out": sum(
            1 for r in scanned if r["mechanism_scan"].get("genuine_ruled_out")),
        "uses_any_q6_intrinsic": sum(
            1 for r in scanned
            if r["mechanism_scan"].get("uses_any_q6_intrinsic")),
        "tasks_covered": len({r["task"]["key"] for r in scanned}),
        "per_tier": per_tier,
        "per_mechanism": per_mech,
    }


def scan_verdict_disagreements(recs) -> list:
    """Attempts where the cheap scan and the executed verdict disagree.

    Both read the same static detectors on the same source, so they must agree; a
    disagreement means one of the two paths is judging something other than what it
    claims. Cheap to check and worth checking, because the scan's whole value rests
    on it being the same measurement the graded path makes.
    """
    out = []
    for r in recs:
        g, s = r.get("grade") or {}, r.get("mechanism_scan") or {}
        if not (g.get("graded") and s.get("scanned")):
            continue
        if g.get("compiled") and sorted(g.get("mechanisms_fired") or []) != \
                sorted(s.get("mechanisms_fired") or []):
            out.append({"task": r["task"]["key"], "seed": r["seed_index"],
                        "graded": sorted(g.get("mechanisms_fired") or []),
                        "scanned": sorted(s.get("mechanisms_fired") or [])})
    return out


def _reasons(ungraded):
    out = {}
    for r in ungraded:
        why = ((r.get("grade") or {}).get("skip_reason")
               or r.get("generation_error") or "not graded yet")
        out[why[:80]] = out.get(why[:80], 0) + 1
    return dict(sorted(out.items(), key=lambda kv: -kv[1]))


def _spread(per_seed):
    """Min/max/range of each rate across seeds -- the variance, stated."""
    out = {}
    for field in ("compiled", "correct", "genuine"):
        vals = [row[field] for row in per_seed.values()]
        if not vals:
            continue
        out[field] = {"min": min(vals), "max": max(vals),
                      "range": max(vals) - min(vals),
                      "mean": round(statistics.mean(vals), 2)}
    return out


def _pct(x):
    return "n/a" if x is None else f"{100 * x:.1f}%"


def render(summary, model=None) -> str:
    """A summary a reader can check, with the caveats attached rather than implied."""
    if not summary.get("n_attempts"):
        return "# No attempts\n"
    s = summary
    lines = [
        f"# Rung {s['rung']} -- {model or s['model']}",
        "",
        f"Eval set `{s['eval_set']}` | {s['n_tasks']} tasks x "
        f"{len(s['seeds'])} seeds = {s['n_attempts']} attempts | "
        f"schema v{s['schema_version']}",
    ]
    if s["model_resolved"]:
        lines.append(f"Model resolved to: {', '.join(s['model_resolved'])}")
    lines += [
        "",
        f"**Graded {s['graded']} of {s['n_attempts']}** "
        f"({s['ungraded']} not graded, {s['generation_failed']} generation "
        f"failures). Rates below are over GRADED attempts only.",
        "",
        "| metric | rate | count |",
        "|---|---|---|",
        f"| compiled | {_pct(s['pooled']['compiled_rate'])} | "
        f"{s['pooled']['compiled']}/{s['graded']} |",
        f"| correct | {_pct(s['pooled']['correct_rate'])} | "
        f"{s['pooled']['correct']}/{s['graded']} |",
        f"| **genuine** (correct and entitled mechanism fires) | "
        f"**{_pct(s['pooled']['genuine_rate'])}** | "
        f"{s['pooled']['genuine']}/{s['graded']} |",
        "",
    ]
    sp = s.get("seed_spread") or {}
    if sp:
        lines += ["## Seed spread (per-seed counts, not a single-seed number)", ""]
        lines += ["| metric | min | mean | max | range |", "|---|---|---|---|---|"]
        for field in ("compiled", "correct", "genuine"):
            if field in sp:
                r = sp[field]
                lines.append(f"| {field} | {r['min']} | {r['mean']} | "
                             f"{r['max']} | {r['range']} |")
        lines.append("")

    if s.get("per_tier"):
        lines += ["## By tier", "",
                  "| tier | graded | compiled | correct | genuine |",
                  "|---|---|---|---|---|"]
        for tier in TIERS:
            row = s["per_tier"].get(tier)
            if row:
                lines.append(
                    f"| {tier} | {row['graded']}/{row['attempts']} | "
                    f"{_pct(row['compiled_rate'])} | {_pct(row['correct_rate'])} "
                    f"| {_pct(row['genuine_rate'])} |")
        lines.append("")

    if s.get("per_mechanism"):
        lines += ["## Mechanism engagement (denominator is ENTITLEMENT)", "",
                  "| mechanism | entitled attempts | fired | rate | correct and fired |",
                  "|---|---|---|---|---|"]
        for mech in MECHANISMS:
            row = s["per_mechanism"].get(mech)
            if row:
                lines.append(f"| {mech} | {row['entitled_attempts']} | "
                             f"{row['fired']} | {_pct(row['fired_rate'])} | "
                             f"{row['correct_and_fired']} |")
        lines.append("")

    ms = s.get("mechanism_scan") or {}
    if ms:
        lines += [
            "## Mechanism scan (no simulator, whole eval set)",
            "",
            f"Static disassembly of a `-c` compile, ~0.4 s per attempt against "
            f"~12 min for a T2 correctness run. Covers **{ms['tasks_covered']} "
            f"tasks / {ms['scanned']} attempts** ({ms['unscannable']} could not be "
            f"compiled, so they make no mechanism claim).",
            "",
            f"- attempts using an entitled mechanism: **{ms['fired']}** "
            f"({_pct(ms['fired_rate'])}) -- of which "
            f"{ms['fired_with_model_written_intrinsics']} wrote the intrinsics "
            f"themselves and **{ms['fired_by_compiler_autovectorisation']} were "
            f"the compiler auto-vectorising scalar code**",
            f"- attempts naming any `Q6_` intrinsic at all: "
            f"**{ms['uses_any_q6_intrinsic']}**",
            f"- **`genuine` ruled out without executing anything: "
            f"{ms['genuine_ruled_out']}** (genuine = correct and fired, so nothing "
            f"firing settles it)",
            "",
        ]
        if ms.get("per_tier"):
            lines += ["| tier | scanned | used a mechanism | named any Q6_ |",
                      "|---|---|---|---|"]
            for tier in TIERS:
                row = ms["per_tier"].get(tier)
                if row:
                    lines.append(f"| {tier} | {row['scanned']} | {row['fired']} "
                                 f"| {row['any_q6']} |")
            lines.append("")
        dis = s.get("scan_verdict_disagreements") or []
        lines.append(
            f"Scan vs executed verdict: **{len(dis)} disagreements** over the "
            f"attempts where both exist." if dis is not None else "")
        if dis:
            for d in dis[:5]:
                lines.append(f"  - {d['task']} seed{d['seed']}: graded "
                             f"{d['graded']} vs scanned {d['scanned']}")
        lines.append("")

    pt = s["per_task"]
    lines += [
        "## Per task, pooled over seeds",
        "",
        f"- solved at least once: {pt['ever_correct']}/{pt['n']}",
        f"- solved on every seed: {pt['always_correct']}/{pt['n']}",
        f"- genuine at least once: {pt['ever_genuine']}/{pt['n']}",
        "",
        "## Diagnostics",
        "",
        f"- hallucinated intrinsics: {s['hallucinated_intrinsics']['attempts']} "
        f"attempts ({_pct(s['hallucinated_intrinsics']['rate'])}), "
        f"{s['hallucinated_intrinsics']['distinct']} distinct invented names",
    ]
    sim = s["scalar_similarity"]
    if sim["n"]:
        lines.append(
            f"- similarity to the naive scalar reference (rung 0 never shows it): "
            f"median {sim['median']}, max {sim['max']} over {sim['n']} attempts")
    u = s["usage"]
    lines.append(f"- tokens: {u['prompt_tokens']:,} in / "
                 f"{u['completion_tokens']:,} out"
                 + (f" | ${u['cost_usd']:.2f} over {u['attempts_priced']} priced "
                    f"attempts" if u["cost_usd"] else " | cost not priced"))
    if s["ungraded_reasons"]:
        lines += ["", "## Why attempts are ungraded", ""]
        for why, n in s["ungraded_reasons"].items():
            lines.append(f"- {n} x {why}")
    return "\n".join(lines) + "\n"
