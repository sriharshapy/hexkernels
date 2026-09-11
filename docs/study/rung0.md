# Rung 0 — the bare baseline, as implemented

Written 2026-08-17, alongside `hexkernels/forge/rung0.py`. **Measured results and their
caveats live in [`rung0-results.md`](rung0-results.md);** this file is the design and
the decisions behind it.

Rung 0 of the ladder in `PLAN.md` §2: *bare prompt, 1 shot — where is the frontier,
unaided?* This file records what was built, the judgement calls made while building
it, and the three that are still the user's to ratify.

## How to run it

```bash
# 1. generate -- network only, no toolchain. Costs money; resumable per attempt.
python -m hexkernels.forge.rung0 generate --model gpt-5.6-luna --seeds 5 --jobs 8

# 2. pre-trace the facts grading needs, so grading never spends its start-up there
python -m hexkernels.forge.rung0 facts

# 3. grade -- ONE simulator, serial. Small tiers first; run OUTSIDE Claude Code.
python -m hexkernels.forge.rung0 grade --model gpt-5.6-luna --tier T0,T1
python -m hexkernels.forge.rung0 grade --model gpt-5.6-luna

# 4. report
python -m hexkernels.forge.rung0 report --model gpt-5.6-luna --seeds 5
```

## Simulation is serial, and that shapes everything downstream

Standing instruction (2026-08-17): **never parallelise `hexagon-sim` on Windows; one
sim thread.** `grade` defaults to `--jobs 1` and *refuses* more, rather than
clamping, so nobody believes they got parallelism they did not get.

Plan around the cost rather than around the constraint. A T0 attempt is seconds; a
**T2 attempt is ~4 minutes**, so grading in task order puts hours of large kernels in
front of a result that is otherwise minutes away. `--tier T0,T1` is the scheduling
tool for that, and every record carries its tier so a partial run can never be read
as a whole one.

Before any grading run, confirm no simulator is already up
(`tasklist | grep -ci hexagon-sim`). Killing stray Python workers is not enough: a
backgrounded shell loop can survive and relaunch them.

## Two corrections applied mid-run, and what was discarded

Both are recorded because a corrected experiment has to be able to say what it threw
away and why. `hexkernels.forge.rung0 invalidate` clears **verdicts only** -- never the
expensive generation half -- so the attempts simply re-enter the grading queue.

1. **352 verdicts taken while four simulators ran concurrently were voided.** Under
   contention a run can exceed its timeout and be recorded as a wrong answer, which
   understates correctness. They are being re-taken serially.
2. **The entry-point lint rule was rejecting valid kernels on formatting.** It
   collapsed runs of whitespace but not whitespace *added* next to punctuation, so
   `candidate_kernel( const _Float16 *x)` was failed against
   `candidate_kernel(const _Float16 *x)` -- a declaration that links perfectly. It
   blocked 19 of 342 graded attempts (5.6%), each recorded as "did not compile".
   Fixed in `hexkernels/forge/lint.py::_canon_sig`, with
   `hexkernels/forge/tests/test_lint_signature.py` pinning both halves: formatting forgiven,
   renamed parameters / changed types / reordered arguments / missing `extern "C"`
   still rejected.

`generate` and `grade` are separate commands because they are bound by different
resources: 640 HTTPS calls that cost money and cannot be repeated for free, versus
640 simulator runs that cost only wall-clock and can be repeated forever. Fused,
one simulator timeout would block the next API call, and a crash halfway would
leave the expensive half half-done.

## What the bare prompt contains, and what it withholds

The whole argument of rung 0 is that it is a **control**: it is what defeats the
objection *"you showed the model scalar code, so of course it wrote scalar code."*
That only holds if the prompt really is bare, so the withheld set is enforced by
tests in `hexkernels/forge/tests/test_rung0.py` rather than left to review.

| given | withheld |
|---|---|
| target name (`Hexagon NSP`, architecture v75) | HVX vector width, L1D/L2/VTCM sizes |
| the operator: `aten::` name + schema | the scalar reference implementation |
| fixed values of non-tensor arguments | the Linalg IR |
| every buffer's shape, dtype, direction | the loop schedule and `vectorizable_loop` |
| the exact `candidate_kernel` signature | the mechanism budget and the tier |
| link/layout/header rules, no-fence rule | every measured ISA fact and intrinsic name |
| the anti-copying provenance rule | the names `hvx`, `hmx`, `dma`, `vtcm`, `l2fetch` |

Two consequences worth stating because they change how a failure should be read:

- **The vendor headers are not enumerated.** `prompt.CONTRACT` (rung 2) lists
  `<hvx_hexagon_protos.h>` and `<hmx_hexagon_protos.h>`, which would name the two
  mechanisms the benchmark is about and hand over the answer as a build
  instruction. Rung 0 says only that the SDK's headers are on the include path. A
  model that wants HVX has to know the header, and a wrong include is a real rung-0
  failure recorded as one.
- **Shapes are given.** This is the one place the implementation departs from the
  note "rung 0 must stay genuinely bare (no scalar, no shape hint)". The signature
  carries only pointers — no extents — so without shapes a kernel cannot know a
  single loop bound, and every attempt would fail for a reason unrelated to whether
  the model reaches for the accelerator. Withholding them would measure nothing.
  **This is the first item for the user to ratify or overturn.**

## The attempt record is the durable interface

One JSON file per attempt, `schema_version: 1`, at
`runs/rung0/<model>/attempts/<task-key>/seed<K>/attempt.json`. Rungs 1–3 and the
aggregator all read it, which is why it is versioned. Two choices in it are easy to
get wrong and expensive to fix after results exist:

- **A task is keyed by `(batch, index)`, never by kernel name.** Names collide by
  construction: `mined._load` puts no tier in the name, so the same op at T0 and T1
  — deliberately different tasks, often one being the other's negative control —
  are both `fp16_linear`. Measured on the frozen set: 320 specs carry 298 distinct
  task ids, and the 165 built references carry 158 distinct names. A record keyed on
  the name silently overwrites its sibling.
- **`eval_set` is stamped on every record** and `aggregate.summarise` raises rather
  than pool two of them, because §3 budgets silicon on 64 tasks while
  `eval_core.json` holds 128.

`hexkernels/forge/evalset.py` is what resolves `eval_core.json`'s 128 identities to pipeline
positions. It exists because doing it by hand matched only 91 of 128 — the
hand-rolled version dropped the fused-epilogue suffix `testset.task_id` appends.

## What `genuine` means, and why it is the headline

`genuine` = `correct` ∧ at least one **entitled** mechanism actually fires in the
disassembly. Entitlement is the task's size-derived mechanism list, so a kernel too
small to justify DMA cannot fail to use DMA. `hvx` entitlement is checked against
the `hvx_compute` flag, not the bare `hvx` flag: a kernel that vector-loads and
vector-stores while doing all arithmetic in scalar registers trips `hvx` and is
exactly the false positive the anti-cheat exists to remove.

Correct-but-scalar is a task **failure** at rung 0, and the record says so
explicitly rather than leaving a reader to derive it.

## Grading is honest about what it cannot judge yet

63 of the 128 eval-core tasks are T2/T3 whose references have not been built
(`scripts/build_refs_parallel.sh` is the job that changes that). Their attempts are
recorded as **ungraded with a reason**, never as incorrect: a candidate judged
against a harness its own reference never passed tells you about the harness, which
is what `run_batch`'s stage (g) exists to say. `aggregate` reports rates over graded
attempts only, with the ungraded count beside them.

Gradable today: T0 32/32, T1 32/32, T2 1/32, T3 0/32 — 65 of 128.

## Open items for the user

1. **Shapes in the rung-0 prompt** — included, against the earlier "no shape hint"
   note. See above for why. Overturning this makes rung 0 unanswerable, so if the
   note is meant literally the ladder needs a different rung-0 definition.
2. **Which GPT-5.6 variant.** The plan says "GPT-5.6"; the API offers
   `gpt-5.6-luna`, `-sol` and `-terra` and no plain `gpt-5.6`. A 3-task pilot picked
   **luna**: 3/3 correct vs sol 2/3 and terra 1/3, at the fewest output tokens (987
   mean vs 1381 / 743) and lowest latency (12.6 s). n=3, so this is weak evidence —
   one flag changes it, and every record carries the resolved model id.
3. **Cost is recorded as `null`** unless `--price-in` / `--price-out` are passed.
   There is deliberately no built-in price table: a stale constant would look like a
   measured number. Token counts are always recorded, so any total can be computed
   after the fact.
