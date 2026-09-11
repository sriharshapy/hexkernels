# Rung 1 — design, and the calls made without ratification

Built 2026-08-20. PLAN.md §2 defines rung 1 as "+ PyTorch reference + scalar code, **up
to 5 turns**", asking: **does help with correctness produce mechanism?**

Rung 0 answered the unaided question: 562/570 correct (98.6%) and **0 of 636 attempts
naming a single `Q6_` intrinsic**. Rung 1 hands over the thing rung 0 withheld and lets
the model iterate, holding everything else fixed.

## Two decisions the user made

1. **A retry turn may carry the compiler's or the harness's report on the model's own
   code, and nothing of ours.** Explicitly NOT `prompt.retry_facts` (≈15 named
   intrinsics, which is what rung 2's `model_client.feedback` sends) and NOT
   `verify.diagnose`'s inference about which elements are wrong.
2. **Scope: same as rung 0** — 128 tasks × 5 seeds, all four tiers, same model
   (`gpt-5.6-luna`), so the delta is attributable to the scaffolding and nothing else.

## Six decisions made without ratification — please check these

**1. "PyTorch reference" is realised as the operator identity plus the verified scalar
C++, with no synthesized `torch` snippet.** Rung 0's `task_statement` already gives the
aten call, its schema and its fixed non-tensor arguments. Emitting Python as well would
mean hand-mapping aten names to `torch.*`, and a snippet that misstated the semantics
would corrupt the measurement *in the direction of looking like a model failure*. The
emitted C++ is a verified statement of the same semantics — stage (g) proves it
reproduces PyTorch's goldens bit-for-bit. **If you wanted a literal Python reference,
this is the line to change.**

**2. Turns run in waves, not per-attempt loops.** Generate for everything unfinished →
grade everything → repeat. Semantically identical, because a turn's feedback depends only
on that attempt's own last verdict, but it preserves the split rung 0's header argues
for: the money-spending half and the wall-clock half stay separate, so a simulator
timeout on task 3 cannot block the API call for task 4. It also makes "how many attempts
still fail after wave N" a measured quantity.

**3. Grading, sweeping and reporting are shared across rungs, not copied.** They read
`candidate.cpp` and write a verdict; nothing about that depends on which prompt produced
the candidate. `rung0.py` gained a `--rung` flag defaulting to 0, so every rung-0 path is
byte-identical to before. This also means a rung-0/rung-1 delta compares *prompts* rather
than two grading implementations.

**4. The record gains `turns` additively, without bumping `schema_version`.** Each turn
records its prompt sha, candidate sha, the feedback given verbatim, the verdict that
provoked it, and usage. Superseded candidates are kept as `candidate_turn<N>.cpp` —
what changed between turns is rung 1's whole subject, and it is unrecoverable once
overwritten. `rung` already distinguishes the trees and the aggregator refuses to pool
rungs, so a new optional field breaks no reader. **Bump it if you disagree.**

**5. Rung 1 needs its own provenance rule.** Rung 0's says "Write this kernel yourself.
Do not copy it from, or pattern it on, any existing solution" — carried over verbatim it
would forbid using the very reference this rung hands over. `RUNG1_PROVENANCE` grants the
permission explicitly, as rung 2's `PROVENANCE_RULE` does, minus that rule's examples
(which name `hmx_helpers.h`, the VTCM base, a crouton offset and the schedule).

**6. The harness's verdict token is neutralised in feedback.** `HVXENV_INCORRECT`
contains "HVX". Echoed verbatim it would name the mechanism to the model, in our words,
on every failing attempt, at the moment it is choosing what to try next — and
`test_rung0.py` bans "hvx" from a prompt for exactly that reason, so the feedback channel
must not readmit it. `HVXENV` → `CHECK`; the numbers after it (error count, element
index, got/want) are preserved, since they are the correctness information the turn
exists to deliver. A compiler diagnostic quoting an intrinsic the *model* wrote is
preserved, because that text is about the model's own code.

## What is withheld, and what enforces it

`hexkernels/forge/tests/test_rung1.py` mirrors `test_rung0.py`: the scalar reference that rung 0
must not contain, rung 1 must; everything else rung 0 withholds, rung 1 still withholds —
hardware facts, working set, tier, mechanism budget, schedule, Linalg, primitive
decomposition, any `hvx`/`hmx`/`dma`/`vtcm`/`l2fetch` string, any `Q6_`/`V6_` name.

One test scans **every** emitted reference in `witness_build` rather than one task's,
because rung 1 pastes `kernel.cpp` into 128 prompts and a single mechanism word in any
one of them leaks that task's answer. It passes: no emitted reference names a mechanism.

`test_rung1_turns.py` pins the feedback boundary, including the two traps: `verify`
appends `diagnose`'s inference to `error_text` (so it is cut at the seam), and the harness
token contains "HVX" (so it is neutralised).

## Pipeline

```bash
python -m hexkernels.forge.rung0 generate --rung 1 --model gpt-5.6-luna --seeds 5 --jobs 8
python -m hexkernels.forge.rung0 grade    --rung 1 --model gpt-5.6-luna --tier T0,T1   # serial
python -m hexkernels.forge.rung0 grade    --rung 1 --model gpt-5.6-luna
python -m hexkernels.forge.rung1 turn     --model gpt-5.6-luna --seeds 5    # one wave
python -m hexkernels.forge.rung0 grade    --rung 1 --model gpt-5.6-luna     # then re-grade
python -m hexkernels.forge.rung0 sweep    --rung 1 --model gpt-5.6-luna     # no simulator
python -m hexkernels.forge.rung0 report   --rung 1 --model gpt-5.6-luna
python -m hexkernels.forge.rung1 histogram --model gpt-5.6-luna
```

Grading stays **one `hexagon-sim`, serial** (standing instruction, 2026-08-17). `turn`
touches no simulator, so its `--jobs 8` is network parallelism only.

## Coverage

114 of 128 tasks are gradable — T0 32/32, T1 32/32, T2 29/32, T3 21/32 — identical to
rung 0, because gradability is a property of the reference and not of the rung. The 14
missing tasks are the ones whose references never passed stage (g) (open item 4 in
`rung0-results.md`).

## What to expect, and the trap in reading it

Rung 0 already reached 98.6% correct, so **the turn loop has very little to fix** — at
most ~8 graded failures out of 570. Expect a turn histogram concentrated almost entirely
at turn 1. That is a finding, not a bug: it says correctness help has nearly no headroom
left at this benchmark's sizes, which sharpens rather than weakens the rung-1 question.

The number that matters is therefore **not** correctness but mechanism engagement. If
rung 1 still shows 0 attempts naming an intrinsic, the claim strengthens considerably:
the model had a correct reference in hand, five chances to improve it, and still never
reached for the accelerator.

Expect scalar-similarity to jump — rung 1 shows the reference, so agreement with it is no
longer evidence of independent authorship. Rung 0's 0.136 median was the answer to the
"you showed it scalar code" objection; that answer belongs to rung 0 only, and any table
printing both must say so.
