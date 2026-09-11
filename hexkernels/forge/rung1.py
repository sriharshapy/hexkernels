"""Rung 1 of the ladder: the reference is handed over, and up to five turns allowed.

    # 1. first ask (network only). Shares rung 0's generator; --rung picks the prompt.
    python -m hexkernels.forge.rung0 generate --rung 1 --model gpt-5.6-luna --seeds 5 --jobs 8
    # 2. grade (ONE simulator, serial)
    python -m hexkernels.forge.rung0 grade --rung 1 --model gpt-5.6-luna --tier T0,T1
    python -m hexkernels.forge.rung0 grade --rung 1 --model gpt-5.6-luna
    # 3. one retry WAVE for whatever failed, then grade again; repeat to --max-turns
    python -m hexkernels.forge.rung1 turn --model gpt-5.6-luna --seeds 5
    python -m hexkernels.forge.rung0 grade --rung 1 --model gpt-5.6-luna
    # 4. mechanisms (no simulator) and the report
    python -m hexkernels.forge.rung0 sweep  --rung 1 --model gpt-5.6-luna
    python -m hexkernels.forge.rung0 report --rung 1 --model gpt-5.6-luna

WHAT RUNG 1 MEASURES. PLAN.md section 2: "+ PyTorch reference + scalar code, up to 5
turns", asking "does help with CORRECTNESS produce MECHANISM?" So the help is confined
to correctness. `prompt.build_rung1` adds the reference and nothing else, and this
module's `feedback` adds the toolchain's complaint and nothing else.
`hexkernels/forge/tests/test_rung1.py` and `test_rung1_turns.py` are the guards, because the
gap between this rung and rung 2 is what PLAN.md section 6 calls the spine.

WHY TURNS RUN IN WAVES rather than one loop per attempt. A per-attempt loop would
interleave an API call with a simulator run 640 times over, fusing the two resources
rung 0 deliberately split (see `rung0`'s own header): the money-spending half and the
wall-clock half would then fail together, and a simulator timeout in task 3 would block
the API call for task 4. A wave is generate-for-everything-unfinished, then
grade-everything, then repeat -- semantically identical, because a turn's feedback
depends only on that attempt's own last verdict, but the expensive half stays batched,
serial and restartable.

It also makes the turn budget observable: after each wave, the number of attempts still
failing is a measured quantity rather than something buried in per-attempt loop state.

WHAT A TURN MAY SAY, decided 2026-08-20. The compiler's or the harness's report on the
model's OWN code -- nothing of ours. Specifically NOT `prompt.retry_facts` (which names
~15 intrinsics and is what rung 2's `model_client.feedback` uses) and NOT
`verify.diagnose`'s inference about which elements are wrong. Echoing a compiler
diagnostic is not a leak even when the diagnostic names an intrinsic, because the model
wrote the code that provoked it; `diagnose` is this repository reasoning about
vector-shaped failure modes, which is rung 2's affordance.
"""
import argparse
import concurrent.futures
import json

from hexkernels.forge import evalset, rung0

#: PLAN.md section 2's budget for this rung.
MAX_TURNS = 5

#: Where `verify` glues `diagnose`'s inference onto the harness line. `feedback` cuts
#: the string here rather than trusting `error_text` to be clean -- see the module
#: docstring, and `test_rung1_turns.py::test_it_strips_diagnose_reasoning...`.
DIAGNOSE_SEAM = "\n  likely cause:"

_ANSWER_SHAPE = ("Fix it and return the complete corrected C++ translation unit, "
                 "with the same entry point and the same signature. Return only the "
                 "code, no prose and no markdown fence.")


#: The harness names its own verdict token `HVXENV_CORRECT` / `HVXENV_INCORRECT`, and
#: that string CONTAINS "HVX".
#:
#: MEASURED by `test_rung1_turns.py`, which caught it: a wrong-answer turn that echoed
#: `error_text` verbatim would tell the model, in our words, that the check it just
#: failed is called HVX-something -- on every failing attempt, at the exact moment the
#: model is deciding what to try next. `test_rung0.py` forbids "hvx" in a prompt for
#: precisely this reason, so allowing it back in through the feedback channel would
#: leak past the guard rather than through it.
#:
#: Neutralised, not dropped: the numbers after the token (error count, element index,
#: got/want) are the correctness information the turn exists to deliver. This is OUR
#: label, which is what makes redacting it right -- a compiler diagnostic quoting an
#: intrinsic the MODEL wrote is preserved, because that text is about the model's own
#: code (see `test_it_preserves_a_compiler_diagnostic_that_names_an_intrinsic`).
HARNESS_TOKEN_PREFIX = "HVXENV"
HARNESS_TOKEN_NEUTRAL = "CHECK"


def toolchain_report(grade: dict) -> str:
    """`error_text` with our own reasoning cut off, leaving only the toolchain's."""
    text = (grade.get("error_text") or "").split(DIAGNOSE_SEAM)[0].rstrip()
    return text.replace(HARNESS_TOKEN_PREFIX, HARNESS_TOKEN_NEUTRAL)


def feedback(grade: dict) -> str:
    """What one failed verdict is allowed to tell the model. "" when it passed."""
    if grade.get("correct"):
        return ""
    report = toolchain_report(grade) or "(no output was captured)"
    if not grade.get("compiled"):
        head = "Your kernel did not compile. The toolchain reported:"
    else:
        head = ("Your kernel compiled and ran, but produced the wrong values. The "
                "harness reported:")
    return "\n".join([head, "", "```", report, "```", "", _ANSWER_SHAPE])


def followup_prompt(first: str, prior_candidate: str, grade: dict) -> str:
    """The next turn's ask: the original question, the model's answer, the complaint.

    The first ask is RESTATED rather than assumed to be in context. The API call is
    stateless here (`rung0.complete` sends one prompt), so a turn that referred to "the
    kernel described above" without the description would be asking the model to work
    from nothing.
    """
    return "\n\n".join([
        first.rstrip(),
        "You have already answered this once. Your previous answer was:",
        f"```cpp\n{prior_candidate.rstrip()}\n```",
        feedback(grade).rstrip(),
    ]) + "\n"


def needs_another_turn(rec: dict, max_turns: int = MAX_TURNS) -> bool:
    """Is this attempt a FAILED, GRADED attempt with budget left?

    Ungraded is not failed. The 70 eval-core attempts whose reference never passed
    stage (g) carry `graded: false` with a reason, and retrying them would spend money
    on a task nothing can grade while inflating the turn histogram with attempts that
    never got a verdict.
    """
    g = rec.get("grade") or {}
    if not g.get("graded") or g.get("correct"):
        return False
    return int(rec.get("turns_used") or 1) < max_turns


def _turn_record(rec, turn, prompt_text, response, cand, grade_before):
    """Append one turn to the record's audit trail, then reset it for re-grading."""
    history = list(rec.get("turns") or [])
    history.append({
        "turn": turn,
        "prompt": {"sha256": rung0._sha(prompt_text), "bytes": len(prompt_text)},
        "candidate": {"sha256": rung0._sha(cand), "bytes": len(cand)},
        "feedback_given": feedback(grade_before),
        "verdict_before": {k: grade_before.get(k) for k in
                           ("compiled", "ran", "correct", "error_text")},
        "usage": response.get("usage") or {},
        "finish_reason": response.get("finish_reason"),
        "at": rung0._now(),
    })
    rec["turns"] = history
    rec["turns_used"] = turn
    # THE VERDICT IS DROPPED, not updated: it belongs to the PREVIOUS candidate, and
    # leaving it in place would make `grade` skip the attempt it just rewrote. It is not
    # lost -- `verdict_before` above keeps it with the turn that earned it, which is
    # what makes a per-turn correctness curve computable afterwards.
    rec["grade"] = {}
    return rec


def cmd_turn(args) -> int:
    """One retry WAVE: re-ask every failed attempt that has turn budget left."""
    tasks = rung0.select_tasks(evalset.core_tasks()[0], args)
    key = rung0.load_key(args.key_file)
    root, model = args.root, args.model

    todo = []
    for ref in tasks:
        for s in range(1, args.seeds + 1):
            adir = rung0.attempt_dir(model, ref, s, root, 1)
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

    counts = {"ok": 0, "err": 0}

    def work(item):
        ref, s, adir, rec = item
        turn = int(rec.get("turns_used") or 1) + 1
        grade_before = dict(rec.get("grade") or {})
        first = (rung0.run_dir(model, root, 1) / "prompts" /
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
        # The superseded candidate is kept beside the new one. Rung 1's whole subject is
        # what CHANGED between turns, and that is unrecoverable once overwritten.
        rung0._write_atomic(adir / f"candidate_turn{turn - 1}.cpp", prior)
        rung0._write_atomic(adir / f"prompt_turn{turn}.md", text)
        rung0._write_atomic(adir / "candidate.cpp", cand)
        rec["candidate"] = {"sha256": rung0._sha(cand), "bytes": len(cand)}
        rec["response"] = {"text_bytes": len(res.get("text") or "")}
        rec = _turn_record(rec, turn, text, res, cand, grade_before)
        rung0._write_atomic(adir / "attempt.json", json.dumps(rec, indent=1) + "\n")
        return ref, s, turn, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for ref, s, turn, err in pool.map(work, todo):
            counts["err" if err else "ok"] += 1
            n = counts["ok"] + counts["err"]
            print(f"  [{n}/{len(todo)}] {ref.key:38s} seed{s} turn{turn} "
                  f"{'ERR ' + str(err)[:50] if err else 'rewritten'}", flush=True)

    print(f"\nwave done: {counts['ok']} re-asked, {counts['err']} failed. "
          f"Now re-grade: python -m hexkernels.forge.rung0 grade --rung 1 --model {model}")
    return 0


def turn_histogram(model, seeds, root=None) -> dict:
    """How many attempts ended at each turn, and whether they ended correct.

    The rung-1 result is not one number: "correct after 1 turn" and "correct after 4"
    are different claims about the scaffolding, and PLAN.md section 2 records turns used
    per attempt precisely so they can be told apart.
    """
    out = {}
    for rec in rung0.collect(model, seeds, root or rung0.RUNS, 1):
        g = rec.get("grade") or {}
        if not g.get("graded"):
            continue
        row = out.setdefault(int(rec.get("turns_used") or 1),
                             {"attempts": 0, "correct": 0})
        row["attempts"] += 1
        row["correct"] += 1 if g.get("correct") else 0
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
        turn_histogram(a.model, a.seeds, a.root), indent=1)), 0)[1])

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
