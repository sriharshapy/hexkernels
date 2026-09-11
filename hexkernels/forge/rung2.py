"""Rung 2 of the ladder: the whole Forge apparatus, and up to five turns.

    # 1. first ask (network only). Shares rung 0's generator; --rung picks the prompt.
    python -m hexkernels.forge.rung0 generate --rung 2 --model gpt-5.6-luna --seeds 5 --jobs 8
    # 2. the MECHANISM headline, before any simulator time is committed (~150 s)
    python -m hexkernels.forge.rung0 sweep --rung 2 --model gpt-5.6-luna
    # 3. grade, waves, final sweep, report
    bash scripts/finish_rung2.sh gpt-5.6-luna 5 5
    # 4. the per-turn mechanism curve, from the saved superseded candidates
    python -m hexkernels.forge.rung0 sweep --rung 2 --model gpt-5.6-luna --all-turns
    python -m hexkernels.forge.rung2 histogram --model gpt-5.6-luna

WHAT RUNG 2 MEASURES. PLAN.md section 2: "Forge: generated reference + goldens,
entitled mechanisms, hardware facts, lint gate, `diagnose` retry", asking "does
verification-grounded generation reach the hardware?" Rung 1 reached 100% correct
with 0 mechanisms, so PLAN.md section 6's second kill criterion now rests here.

WHAT A TURN MAY SAY, and why this module is thin. Everything rung 1 was denied:
`prompt.retry_facts` (~15 named intrinsics) and `verify.diagnose`'s inference about
WHICH elements are wrong. Both already have a home -- `model_client.feedback` is the
Forge pipeline's own retry text -- so rung 2 does not write its own prose. It adapts
the stored grade record into the verdict dict that function consumes, and the claim
"rung 2 is Forge" is then true of the words and not merely of the pipeline.

THE SEAM. `verify` appends `diagnose`'s inference onto `error_text` after
`rung1.DIAGNOSE_SEAM`. Rung 1 cuts there and discards the second half; rung 2 keeps
both. Same seam, opposite side, and that is the entire difference between the two
rungs' retry channels.

WHAT IS NOT CARRIED OVER FROM RUNG 1. The `HVXENV` -> `CHECK` neutralisation. Rung 1
redacted the harness's own verdict token because it contains "HVX" and a rung-1
prompt may not. Rung 2's mechanism budget names hvx, hmx, dma, vtcm and l2fetch in
the FIRST prompt on purpose, so redacting the same word in the feedback would be
theatre while corrupting the toolchain's text. `test_rung2_turns.py` asserts the
token passes through, so this is a decision on the record rather than a guard
someone forgot to port.
"""
import argparse
import concurrent.futures
import json
import traceback

from hexkernels.forge import evalset, model_client, rung0, rung1

#: PLAN.md section 2's budget, and deliberately rung 1's number: holding the cap
#: fixed is what makes the rung-1 -> rung-2 delta attributable to prompt and
#: feedback CONTENT rather than to how many chances each rung got.
MAX_TURNS = 5

_ANSWER_SHAPE = ("Fix it and return the complete corrected C++ translation unit, "
                 "with the same entry point and the same signature. Return only the "
                 "code, no prose and no markdown fence.")


def verdict_from_grade(grade: dict) -> dict:
    """The stored grade record, reshaped into what `model_client.feedback` consumes.

    An ADAPTER and not a translation: every field it needs is already recorded by
    `rung0.grade_one`, so this adds no state and cannot drift from the verdict. The
    only real work is the seam -- `diagnose`'s inference arrives glued onto
    `error_text`, and `feedback` wants the two apart so it can label them
    differently ("the harness reports" versus "these are hypotheses").
    """
    head, _, tail = (grade.get("error_text") or "").partition(rung1.DIAGNOSE_SEAM)
    diagnosis = [ln.strip() for ln in tail.splitlines() if ln.strip()] if tail else []
    return {
        "rd_leak": list(grade.get("rd_leak") or []),
        "lint_errors": list((grade.get("lint") or {}).get("errors") or []),
        "compiled": bool(grade.get("compiled")),
        "ran": bool(grade.get("ran")),
        "correct": bool(grade.get("correct")),
        "error_text": head.rstrip(),
        # FALLING BACK TO `error_text` IS DELIBERATE. `model_client.feedback`'s
        # wrong-values branch renders "The harness reports:" from `stdout`, while
        # its compile branch renders from `error_text` -- and `verify` does not
        # guarantee both are populated for the same failure. Without the fallback a
        # wrong-answer turn whose harness line landed only in `error_text` would
        # tell the model its numbers were wrong and then show it nothing, which is
        # the one thing a retry may not do.
        "stdout": grade.get("stdout") or head.rstrip(),
        "diagnosis": diagnosis,
    }


def feedback_kind(grade: dict) -> str:
    """Which stage this attempt died at. "" when it passed.

    PLAN.md section 5.B asks for a per-stage failure taxonomy, and rung 2 is the
    first rung with more than one stage in play. It also separates a POLICY
    rejection from a CAPABILITY failure: `lint._r_entitlement` makes using more
    mechanism than the size grants an error, so some rung-2 turns are spent on
    entitlement rather than on getting the numbers right. Reported together they
    would inflate "rung 2 needed more turns" into a claim about the model.

    Same order as `model_client.feedback`, because that is the order the pipeline
    applies the gates in.
    """
    if grade.get("correct"):
        return ""
    if grade.get("rd_leak"):
        return "rd_leak"
    if (grade.get("lint") or {}).get("errors"):
        return "lint"
    if not grade.get("compiled"):
        return "compile"
    if not grade.get("ran"):
        return "ran"
    return "wrong"


def feedback(grade: dict) -> str:
    """What one failed verdict is allowed to tell the model. "" when it passed."""
    if grade.get("correct"):
        return ""
    body = model_client.feedback(verdict_from_grade(grade))
    if not body:
        return ""
    return body.rstrip() + "\n\n" + _ANSWER_SHAPE


def followup_prompt(first: str, prior_candidate: str, grade: dict) -> str:
    """The next turn's ask: the original question, the model's answer, the complaint.

    The first ask is RESTATED rather than assumed to be in context, for the same
    reason rung 1 restates it: `rung0.complete` sends one prompt and the API call is
    stateless, so a turn referring to "the kernel described above" without the
    description asks the model to work from nothing. At rung 2 the restated ask is
    large (facts, Linalg, schedule, budget), which is why the cached-token share
    rises and why `rung0.cost_usd` had to learn a third rate.
    """
    return "\n\n".join([
        first.rstrip(),
        "You have already answered this once. Your previous answer was:",
        f"```cpp\n{prior_candidate.rstrip()}\n```",
        feedback(grade).rstrip(),
    ]) + "\n"


def needs_another_turn(rec: dict, max_turns: int = MAX_TURNS) -> bool:
    """Is this attempt a FAILED, GRADED attempt with budget left?

    Ungraded is not failed -- the 70 eval-core attempts whose reference never passed
    stage (g) carry `graded: false` with a reason, and retrying them would spend
    money on a task nothing can grade while inflating the turn histogram with
    attempts that never got a verdict.
    """
    g = rec.get("grade") or {}
    if not g.get("graded") or g.get("correct"):
        return False
    return int(rec.get("turns_used") or 1) < max_turns


def _add_numeric(existing: dict, incoming: dict) -> dict:
    """Recursively fold INCOMING's numeric leaves into EXISTING, dict by dict.

    Generalises what `_add_usage` used to special-case for exactly one nested key,
    `prompt_tokens_details`. Real records also carry
    `completion_tokens_details.reasoning_tokens` (e.g. 1024 against a
    `completion_tokens` of 1224), and naming one nested dict while skipping every
    other left a folded multi-turn record with turn 1's reasoning count sitting
    beside an N-turn completion total -- a wrong ratio wherever the reasoning share
    is reported. Headline totals were never at risk: `completion_tokens` already
    includes reasoning, so this only matters to a reader who wants the split.
    """
    out = dict(existing or {})
    for k, v in (incoming or {}).items():
        if isinstance(v, dict):
            out[k] = _add_numeric(out.get(k) or {}, v)
        elif isinstance(v, (int, float)):
            out[k] = (out.get(k) or 0) + v
        # A non-numeric, non-dict leaf (a string, a null) has nothing to sum, so it
        # is left as whatever `existing` already had -- same as the fields this
        # function does not mention at all.
    return out


def _add_usage(total: dict, more: dict) -> dict:
    """Accumulate one turn's usage into the record's top-level total.

    RUNG 1'S SECOND ACCOUNTING DEFECT, fixed here rather than after the run.
    `rung1.cmd_turn` recorded each turn's usage inside `turns[]` and never folded it
    into `usage`, so anything summing the top-level field missed it -- 24,533 in /
    10,305 out over rung 1's 12 turns. At rung 2 turns are the point, so the
    under-report would be a fraction of the rung's whole cost, and that cost is the
    thing the rung has to justify.

    `turns_counted` is written so a reader can tell an accumulated total from a
    turn-1-only one, which is what rung 1's records silently were.
    """
    out = _add_numeric(total, more)
    out["turns_counted"] = int(out.get("turns_counted") or 1) + 1
    return out


def _turn_record(rec, turn, prompt_text, response, cand, grade_before):
    """Append one turn to the record's audit trail, then reset it for re-grading."""
    history = list(rec.get("turns") or [])
    history.append({
        "turn": turn,
        "prompt": {"sha256": rung0._sha(prompt_text), "bytes": len(prompt_text)},
        "candidate": {"sha256": rung0._sha(cand), "bytes": len(cand)},
        # VERBATIM, because what the model was told is the rung's independent
        # variable and a summary of it is not evidence.
        "feedback_given": feedback(grade_before),
        "feedback_kind": feedback_kind(grade_before),
        "verdict_before": {k: grade_before.get(k) for k in
                           ("compiled", "ran", "correct", "error_text")},
        "usage": response.get("usage") or {},
        "finish_reason": response.get("finish_reason"),
        "at": rung0._now(),
    })
    rec["turns"] = history
    rec["turns_used"] = turn
    rec["usage"] = _add_usage(rec.get("usage") or {}, response.get("usage") or {})
    # RE-PRICE, not just re-count. `_add_usage` folds this turn's tokens into the
    # running total, but `generate_one` only ever priced turn 1 -- left alone, the
    # record would carry N turns of tokens against one turn of dollars, and
    # `aggregate` sums exactly this field as the rung's headline cost.
    rung0.reprice(rec["usage"])
    # THE VERDICT IS DROPPED, not updated: it belongs to the PREVIOUS candidate, and
    # leaving it in place would make `grade` skip the attempt it just rewrote. It is
    # not lost -- `verdict_before` above keeps it with the turn that earned it.
    rec["grade"] = {}
    return rec


def cmd_turn(args) -> int:
    """One retry WAVE: re-ask every failed attempt that has turn budget left.

    WAVES, NOT PER-ATTEMPT LOOPS (rung-1 spec, decision 2). A per-attempt loop would
    interleave an API call with a simulator run hundreds of times, fusing the two
    resources rung 0 deliberately split: a simulator timeout on one task would block
    the network call for the next. A wave also makes "how many attempts still fail
    after wave N" a measured quantity instead of loop state.
    """
    tasks = rung0.select_tasks(evalset.core_tasks()[0], args)
    key = rung0.load_key(args.key_file)
    root, model = args.root, args.model

    todo = []
    for ref in tasks:
        for s in range(1, args.seeds + 1):
            adir = rung0.attempt_dir(model, ref, s, root, 2)
            rec = rung0._existing(adir)
            if rec is None:
                continue
            if needs_another_turn(rec, args.max_turns):
                todo.append((ref, s, adir, rec))

    print(f"{len(todo)} attempts still failing with turn budget left "
          f"(max {args.max_turns})")
    if not todo:
        print("nothing to retry: every graded attempt is either correct or out of "
              "budget")
        return 0

    kinds = {}
    for _, _, _, rec in todo:
        k = feedback_kind(rec.get("grade") or {})
        kinds[k] = kinds.get(k, 0) + 1
    print("  this wave by failure stage: "
          + ", ".join(f"{k}={v}" for k, v in sorted(kinds.items())))

    counts = {"ok": 0, "err": 0}

    def work(item):
        ref, s, adir, rec = item
        turn = int(rec.get("turns_used") or 1) + 1
        # ONE ATTEMPT MAY NOT TAKE THE WHOLE WAVE DOWN WITH IT, same discipline as
        # `rung0.cmd_generate.work`. `pool.map` propagates the first exception and
        # abandons everything still queued -- the exact `WinError 32` hazard that
        # comment describes, here on `read_text`/`_write_atomic` calls that run
        # AFTER the API money for this attempt is already spent. An unguarded raise
        # would lose that paid work and stop every other attempt in the wave.
        try:
            grade_before = dict(rec.get("grade") or {})
            first = (rung0.run_dir(model, root, 2) / "prompts" /
                     f"{ref.key}.md").read_text(encoding="utf-8")
            prior = (adir / "candidate.cpp").read_text(encoding="utf-8")
            text = followup_prompt(first, prior, grade_before)

            res = rung0.complete(model, text, key, max_tokens=args.max_tokens,
                                 seed=None if args.seed_base is None
                                 else args.seed_base + s,
                                 temperature=args.temperature)
            if res.get("error"):
                rec["generation_error"] = f"turn {turn}: {res['error']}"
                rung0._write_atomic(adir / "attempt.json",
                                    json.dumps(rec, indent=1) + "\n")
                return ref, s, turn, res["error"]

            cand = rung0.strip_fence(res["text"] or "")
            # The superseded candidate is kept beside the new one. What changed
            # between turns is this rung's subject, and `sweep --all-turns` reads
            # exactly these files to build the per-turn mechanism curve -- rung 1's
            # single most informative attempt would not exist under an
            # overwrite-in-place design.
            rung0._write_atomic(adir / f"candidate_turn{turn - 1}.cpp", prior)
            rung0._write_atomic(adir / f"prompt_turn{turn}.md", text)
            rung0._write_atomic(adir / "candidate.cpp", cand)
            rec["candidate"] = {"sha256": rung0._sha(cand), "bytes": len(cand)}
            rec["response"] = {"text_bytes": len(res.get("text") or "")}
            rec = _turn_record(rec, turn, text, res, cand, grade_before)
            rung0._write_atomic(adir / "attempt.json",
                                json.dumps(rec, indent=1) + "\n")
            return ref, s, turn, None
        except Exception as exc:               # noqa: BLE001 - recorded, not raised
            rec["generation_error"] = (f"turn {turn} driver error: "
                                       f"{type(exc).__name__}: {exc}")
            rung0._write_atomic(adir / "attempt.json",
                                json.dumps(rec, indent=1) + "\n")
            rung0._write_atomic(adir / "driver_error.txt", traceback.format_exc())
            return ref, s, turn, str(exc)

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for ref, s, turn, err in pool.map(work, todo):
            counts["err" if err else "ok"] += 1
            n = counts["ok"] + counts["err"]
            print(f"  [{n}/{len(todo)}] {ref.key:38s} seed{s} turn{turn} "
                  f"{'ERR ' + str(err)[:50] if err else 'rewritten'}", flush=True)

    print(f"\nwave done: {counts['ok']} re-asked, {counts['err']} failed. "
          f"Now re-grade: python -m hexkernels.forge.rung0 grade --rung 2 --model {model}")
    return 0


def turn_histogram(model, seeds, root=None) -> dict:
    """How many attempts ended at each turn, and whether they ended correct."""
    out = {}
    for rec in rung0.collect(model, seeds, root or rung0.RUNS, 2):
        g = rec.get("grade") or {}
        if not g.get("graded"):
            continue
        row = out.setdefault(int(rec.get("turns_used") or 1),
                             {"attempts": 0, "correct": 0})
        row["attempts"] += 1
        row["correct"] += 1 if g.get("correct") else 0
    return dict(sorted(out.items()))


def turn_taxonomy(model, seeds, root=None) -> dict:
    """Every turn ever spent, counted by the failure stage that provoked it.

    PLAN.md section 5.B's per-stage failure taxonomy. It is reported SEPARATELY from
    the turn histogram because the two answer different questions: the histogram
    says how many chances an attempt needed, this says what it was failing at. In
    particular `lint` includes entitlement rejections, which are a policy verdict of
    ours rather than a capability failure of the model's, so pooling them into "rung
    2 needed more turns" would overstate the difficulty of the rung.
    """
    out = {}
    for rec in rung0.collect(model, seeds, root or rung0.RUNS, 2):
        for turn in rec.get("turns") or []:
            k = turn.get("feedback_kind") or "unclassified"
            out[k] = out.get(k, 0) + 1
    return dict(sorted(out.items()))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("turn", help="one retry wave over every failed attempt")
    t.add_argument("--model", default="gpt-5.6-luna")
    t.add_argument("--seeds", type=int, default=5)
    t.add_argument("--root", default=str(rung0.RUNS))
    t.add_argument("--only", default=None)
    t.add_argument("--tier", default=None)
    t.add_argument("--max-turns", type=int, default=MAX_TURNS)
    t.add_argument("--jobs", type=int, default=8,
                   help="network only; the simulator is never touched here")
    t.add_argument("--max-tokens", type=int, default=16384)
    t.add_argument("--temperature", type=float, default=1.0)
    t.add_argument("--seed-base", type=int, default=None)
    t.add_argument("--key-file", default=None)
    t.set_defaults(fn=cmd_turn)

    h = sub.add_parser("histogram", help="attempts and correctness per turn used")
    h.add_argument("--model", default="gpt-5.6-luna")
    h.add_argument("--seeds", type=int, default=5)
    h.add_argument("--root", default=str(rung0.RUNS))
    h.set_defaults(fn=lambda a: (print(json.dumps(
        {"turns": turn_histogram(a.model, a.seeds, a.root),
         "by_failure_stage": turn_taxonomy(a.model, a.seeds, a.root)},
        indent=1)), 0)[1])

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
