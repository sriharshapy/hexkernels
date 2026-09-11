# Rung 2 — design, and the calls made without ratification

Written 2026-08-25. PLAN.md §2 defines rung 2 as "**Forge**: generated reference + goldens,
entitled mechanisms, hardware facts, lint gate, `diagnose` retry", asking: **does
verification-grounded generation reach the hardware?**

This is the rung PLAN.md §6 calls the spine. Rung 0 measured the frontier unaided (562/570
correct, **0 of 636 attempts naming a single `Q6_` intrinsic**). Rung 1 handed over the
correct reference and five turns and got **100% correct, 0 mechanisms** — and removed the
only attempt at one. So contribution #3 now rests entirely on rung 2: if rung 2 ≈ rung 1,
PLAN.md §6's second kill criterion fires and the paper becomes benchmark + metric.

Rung 1 also left rung 2 a specific, falsifiable prediction. `b37i3_i32_bitwise_not` seed 5
held the *compute* vocabulary (`Q6_V_vnot_V` is real) and fabricated the *memory*
vocabulary (`Q6_V_ld`, `Q6_V_st` — zero occurrences in the vendor headers). That is a
load/store knowledge gap, and `prompt.retry_facts` is the thing that answers it. Rung 2 is
where that gets tested rather than asserted.

## Two decisions the user made

1. **Five turns, and a turn carries `model_client.feedback` in full.** The cap matches
   rung 1's, so the rung-1 → rung-2 delta is prompt and feedback CONTENT and not turn
   budget. A turn carries the whole Forge channel, in the order the pipeline applies it:
   provenance leak → lint errors + `ISA_FACTS` → compiler output → simulator fault →
   wrong values + `verify.diagnose` hypotheses + `ISA_FACTS`. This is exactly what rung 1
   was denied (rung-1 spec, decision 1).

   PLAN.md §5.B's separate "retry budget fixed at 3" for the autonomous Forge yield gate
   needs no separate run: k=1 and k=3 are read off this rung's turn histogram.

2. **Staging follows rung 1's script; the mechanism reading is taken early.** Simulator
   cost is bounded the way `finish_rung1.sh` already bounds it (below), and the mechanism
   sweep — which needs no simulator at all — runs right after generation so the headline
   number exists before any simulator hours are committed.

## Carried from rung 0/1 unchanged, and not reopened

| decision | source |
|---|---|
| One `hexagon-sim`, serial. Every `grade` is `--jobs 1`, sequenced, never backgrounded | standing instruction, 2026-08-17 |
| Turns run in **waves**, not per-attempt loops, so API spend and wall-clock stay separate and "how many still fail after wave N" is measured | rung-1 spec, decision 2 |
| **T0,T1 graded first**, then T2,T3 (T0 ~1 s, T3 median ~2.5 min) | `finish_rung1.sh` constraint 2 |
| Waves **scoped by tier**, so cheap tiers converge before expensive ones start | `finish_rung1.sh` |
| The **final sweep runs last** — a wave rewrites `candidate.cpp`, so an earlier sweep describes candidates that no longer exist | `finish_rung1.sh` constraint 3 |
| `SIM_TIMEOUT=9000` threaded through the main pass **and** the in-wave re-grade | rung-1 fix, 2026-08-21 |
| An exhausted budget is **ungraded**, never a wrong answer | rung-0 fix, 2026-08-19 |
| No `--timing`. The simulator is a correctness + mechanism gate; cycles come from silicon | ratified 2026-08-17 |
| Grade / sweep / report are **shared across rungs** behind `--rung`, never copied | rung-1 spec, decision 3 |
| Scope: 128 tasks × 5 seeds, all four tiers, same model (`gpt-5.6-luna`), so the delta is the scaffolding | rung-1 spec, decision 2 |
| Ungraded ≠ failed: the 70 attempts whose reference never passed stage (g) are never retried | `rung1.needs_another_turn` |

## Seven decisions made without ratification — please check these

**1. "Goldens" means the verifier, not prompt text.** PLAN.md §2 lists "generated
reference + goldens" among what rung 2 is given, and the code settles how to read it:
goldens are numeric arrays embedded in `harness_c`, and at T3 that is millions of
elements. They cannot go in a prompt. What rung 2 gets is the *generated reference* in the
prompt and the *goldens as the thing that judges it* — which is the Forge apparatus
(`run_batch` stages (e)-(g)) rather than a prompt section. **If you meant literal golden
values in the prompt, this is the line to change**, and it would need a sampling rule
(first N elements? a checksum?) that nothing in the pipeline currently has.

**2. Rung 2's prompt is `prompt.build()` PLUS rung 0/1's task statement and buffer
table.** `build()` is the batch Forge prompt and it omits both — it opens with the
reference and the schedule, because a corpus author already knows the operator. Carried
over verbatim, rung 2 would *withhold* the aten identity, its schema, its fixed non-tensor
arguments and the shape table that rungs 0 and 1 both hand over, and the ladder would stop
being monotone at exactly the rung the paper's argument depends on. A reviewer would be
right to call that out.

So `build_rung2` = rung-1's opening (name, arch, `task_statement`, `_buffer_table`,
reference) + `hardware_facts` + Linalg + schedule + mechanism budget + `PROVENANCE_RULE` +
the full `CONTRACT`. Two consequences worth stating rather than discovering:

  - Rung 2's prompt is a strict SUPERSET of the batch Forge prompt, so "rung 2 is Forge"
    is true of the pipeline and the feedback but not byte-exact on the first ask. Disclose
    it; the added text is redundant given the reference (the reference computes the
    operator, the signature carries the shapes), so it cannot manufacture a mechanism
    delta — it only removes a confound.
  - `test_rung1.py::test_it_sits_between_rung0_and_rung2_in_size` currently defines
    "rung 2" as `build()`. It has to point at `build_rung2` instead.

**3. `ISA_FACTS` stays retry-only.** `prompt.retry_facts` documents why the line sits
between the first ask and a retry. Keeping it there buys a second measurement for free:
**turn 1 contains no intrinsic names by construction**, so turn-1-only numbers isolate
"first-prompt scaffolding" from "intrinsic names injected on retry" out of a single run,
with no extra arm.

**The guard `prompt.py` cites for this does not exist in this repository.**
`retry_facts`' docstring says adding `ISA_FACTS` to `build()` "fails
`test_prompt_does_not_hand_over_the_answer`", and that test was never ported — PLAN.md
§5.A left the suites behind deliberately, and the only occurrences of the name are two
comments in `prompt.py`. So rung 2 must WRITE that guard rather than inherit it, and it is
the single most important test in this rung: without it, nothing stops an intrinsic name
reaching a first prompt, and the rung-2 headline would be unfalsifiable.

**4. Rung 2's feedback IS `model_client.feedback`, reached through a thin adapter.**
Rung 2 does not get its own feedback prose. `rung2._verdict_from_grade(grade)` reshapes the
stored grade record into the verdict dict `model_client.feedback` already consumes, so the
ladder's rung 2 and the batch Forge pipeline say the same words to the model. Every field
it needs is already recorded — no schema change:

| verdict key | where it comes from |
|---|---|
| `rd_leak` | `grade["rd_leak"]` |
| `lint_errors` | `grade["lint"]["errors"]` |
| `compiled`, `ran`, `correct`, `stdout` | `grade[...]` |
| `error_text` | `grade["error_text"]` split at `rung1.DIAGNOSE_SEAM`, first half |
| `diagnosis` | the same split, second half |

`verify` appends `diagnose`'s inference onto `error_text` at that seam, and rung 1 cuts
there and **throws the second half away**. Rung 2 keeps both halves. The same seam, the
opposite side — that symmetry is the whole difference between the two rungs' retry
channels, and it is worth one sentence in the paper.

**5. The `HVXENV` → `CHECK` neutralisation is DROPPED at rung 2.** Rung 1 redacted the
harness's own verdict token because it contains "HVX" and `test_rung0.py` bans that string
from a prompt. At rung 2 the mechanism budget names `hvx`, `hmx`, `dma`, `vtcm` and
`l2fetch` in the first prompt on purpose. Redacting the same word in the feedback while
printing it in the prompt would be theatre, and it would corrupt the toolchain's own text
for no gain. **`test_rung2_turns.py` asserts the token passes through**, so this is a
recorded choice rather than a rung-1 guard someone forgot to port.

**6. The per-turn mechanism curve is computed post-hoc, not plumbed through the wave.**
Rung 1's open item — count mechanism engagement on the FINAL candidate or on ANY turn —
stops being a footnote at rung 2, where waves will rewrite candidates in volume. Rather
than threading a scan into `turn` (which would add an ordering constraint to the one part
of the pipeline that currently has none), `sweep --all-turns` scans every saved
`candidate_turn<N>.cpp` plus the final `candidate.cpp` and writes
`mechanism_scan_turns: [{turn, ...}]`. It touches no simulator, so it can run at any time,
and it cannot be invalidated by wave ordering. This works only because rung 1 chose to
keep superseded candidates instead of overwriting in place.

**7. Entitlement lint errors will now actually fire, and will consume turns.**
`lint._r_entitlement` makes using MORE than the size grants an error — `l2fetch` at a tier
that does not grant it, `dma` likewise. At rungs 0 and 1 the rule was dead code: nothing
called an intrinsic. At rung 2 the budget invites mechanism use, so some attempts will
spend a turn on entitlement rather than on correctness. That is the gate working as
designed, and it means **the turn taxonomy must separate entitlement rejections from
correctness failures** — otherwise "rung 2 needed more turns" reads as a capability
statement when part of it is a policy statement. Each turn records its
`feedback_kind` ∈ {`rd_leak`, `lint`, `compile`, `ran`, `wrong`} for exactly this reason.

## Two accounting defects fixed as part of this rung, not after it

Both were found on 2026-08-21 and neither mattered at rung 1's scale. Both matter here.

1. **`rung0.cost_usd` bills cached input at the uncached rate.** Rung 1 already had 49.6%
   of input tokens cached; rung 2's prompt is far larger and every retry restates it, so
   the cache-hit fraction goes up and the error goes up with it. Needs the third rate:
   `(prompt - cached)*in + cached*cached_rate + completion*out`.
2. **Per-turn usage is not folded into the record's top-level `usage`.** At rung 1 that hid
   24,533 in / 10,305 out across 12 turns. At rung 2 turns are the point, so anything
   summing `usage` would under-report the rung's cost by whatever fraction the retry loop
   accounts for — which is the number the rung exists to justify.

A third item from the same day is **in scope if it is cheap**: a verdict cache keyed on
candidate sha256. Rung 1 wasted 5.68 B instructions (~36 min) re-simulating byte-identical
candidates across seeds, 21 of them at T3.

## What is given, what is withheld, and what enforces it

`hexkernels/forge/tests/test_rung2.py` mirrors `test_rung1.py` **inverted**. Rung 1's tests assert
absence; rung 2's assert presence, because at this rung the scaffolding IS the treatment:

- the reference, the task statement and the buffer table (everything rung 1 carried)
- `hardware_facts`: HVX width, L1D, L2, VTCM size and base
- the Linalg IR when torch-mlir produced it, and the derived schedule with
  `vectorizable_loop`
- the mechanism budget, with the arithmetic that justifies each grant
- the full `CONTRACT`, which enumerates `<hvx_hexagon_protos.h>` and
  `<hmx_hexagon_protos.h>` (rung 0 deliberately did not) and carries the measured DMA
  descriptor-alignment and priming rules
- `PROVENANCE_RULE`, the version with examples

Still withheld: **no intrinsic names in the first prompt**, no worked example, no target
code. The guard is `test_rung2.py::test_the_first_prompt_names_no_intrinsic`, written here
for the first time (see unratified call 3) and covering `build_rung2` and `build()` both —
`build()` needs it too, since `run_batch` sends that prompt to build the corpus.

Determinism is tested as at every other rung: same spec in, byte-identical text out.

## Pipeline

```bash
python -m hexkernels.forge.rung0 generate --rung 2 --model gpt-5.6-luna --seeds 5 --jobs 8
python -m hexkernels.forge.rung0 sweep    --rung 2 --model gpt-5.6-luna   # THE HEADLINE, ~150 s, no simulator
bash scripts/finish_rung2.sh gpt-5.6-luna 5 5                     # grade, waves, final sweep, report
python -m hexkernels.forge.rung0 sweep    --rung 2 --model gpt-5.6-luna --all-turns
python -m hexkernels.forge.rung2 histogram --model gpt-5.6-luna
```

`finish_rung2.sh` is `finish_rung1.sh` with the rung flag changed and the wave calling
`rung2 turn`. The early sweep is safe to run beside a grading pass — it touches no
simulator, so it does not break the one-`hexagon-sim` rule — and it is not the *final*
sweep, which still runs last.

## Coverage

114 of 128 tasks are gradable — T0 32/32, T1 32/32, T2 29/32, T3 21/32 — identical to
rungs 0 and 1, because gradability is a property of the reference and not of the rung. The
14 missing tasks are the ones whose references never passed stage (g)
(`rung0-results.md`, open item 4). Mechanism scanning covers all 128.

## What to expect, and the traps in reading it

**The turn histogram will actually spread, and that is not a regression.** Rung 1 used 12
retry turns in 640 attempts because rung 0 had left almost nothing to fix. Rung 2 invites
the model to write intrinsics, so it will produce failure classes the earlier rungs never
saw: fabricated names caught by lint, float intrinsics that pass `-fsyntax-only` and cannot
be selected at -O2, wrong lane order after a widening deal, DMA descriptors on the stack.
Every one of those is what `ISA_FACTS` and `diagnose` exist for. **Correctness falling
below rung 1's 100% at turn 1 is expected** — the model is attempting something harder —
and the number to read is the final one alongside the taxonomy.

**Simulator cost is the schedule risk, and it is unpriceable in advance.** Each wave
re-grades its failing set on one serial simulator, and the two heaviest eval-core attempts
measured 9.98 B and 7.82 B instructions — ~64 and ~50 min each at 2.6 M instructions/s.
The mitigation is the ordering above (mechanism headline first, cheap tiers converge
before expensive ones start), not a smaller scope: PLAN.md §2 forbids a sub-5-seed number,
and T2/T3 are where `dma`, `vtcm` and `hmx` entitlements live.

**Two results, and both are publishable.** If mechanism engagement rises, contribution #3
holds and the "verification-grounded pipeline fixes it" claim is measured rather than
asserted. If it stays at zero *with* hardware facts, an entitlement budget, a lint gate
and fifteen named intrinsics on retry, PLAN.md §6's rung-2 kill criterion fires — and the
finding is considerably stronger than rung 1's, because everything a reviewer would
propose as the fix will have been tried. PLAN.md's instruction applies either way: **do
not write an intro that depends on failure.**

**HMX has no achievability witness.** T3 grants `hmx` and the corpus holds 0 of 15
witnesses for it, so a rung-2 HMX attempt has no demonstrated ceiling to be measured
against. Report `hmx` engagement, but do not report a speedup-vs-expert for it until
§5.D's missing-reference item is closed.
