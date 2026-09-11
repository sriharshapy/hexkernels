# Rung 1 — measured results

**Status: COMPLETE** (2026-08-21). 570 of 640 attempts graded — the same 570 as rung 0,
so every tier is paired. The 70 ungraded are the tasks whose reference never passed
stage (g), identical to rung 0.

- Model `gpt-5.6-luna` · eval set `core128` · schema v1 · 5 seeds · rung 1
- Generated 2026-08-20 (128/128 tasks, 640 attempts, 0 failures)
- Prompt: rung 0's ask **plus the correct-but-slow scalar reference, and nothing else**
  (`prompt.build_rung1`). Design and six unratified calls:
  [`superpowers/specs/2026-08-20-rung1-design.md`](specs/2026-08-20-rung1-design.md)
- Up to 5 turns; a turn carries the compiler's or the harness's complaint about the
  model's own code and nothing of ours (`rung1.feedback`)
- Live numbers: `runs/rung1/gpt-5.6-luna/SUMMARY.md`

PLAN.md §2 asks of this rung: **does help with correctness produce mechanism?** The
answer is **no**, on a complete measurement: correctness reached 100% and mechanism
engagement reached 0 — and one attempt shows the loop actively working against it.

## The mechanism result

| | rung 0 | rung 1 |
|---|---|---|
| attempts scanned | 636 | 640 |
| attempts where an entitled mechanism **fired** | 3 | **0** |
| — of those, model-written intrinsics | 0 | 0 |
| — of those, compiler auto-vectorisation | 3 | 0 |
| attempts **naming** any `Q6_` intrinsic (any turn) | 0 | **1** |
| hallucinated intrinsic names | 0 | **2** |

**Given the answer and five attempts to improve it, mechanism engagement went down.**
Rung 0's three `genuine` hits were hexagon-clang auto-vectorising the model's own scalar
loops at T3. At rung 1 the model transcribes the *reference's* loop structure instead,
and that structure does not auto-vectorise — so the accidental hits disappear.

## The one attempt that reached for the hardware, and what the loop did to it

`b37i3_i32_bitwise_not` seed 5 (T1) is the only attempt in **1,280** across both rungs
to name a Hexagon intrinsic. Its first turn:

```cpp
#include <hexagon_types.h>
#include <hexagon_protos.h>
extern "C" void candidate_kernel(const int32_t *v_args_0, int32_t *out0) {
  constexpr int kVectorElements = sizeof(HVX_Vector) / sizeof(int32_t);
  for (int i = 0; i < 48 * 128; i += kVectorElements) {
    HVX_Vector input = Q6_V_ld(v_args_0 + i);
    HVX_Vector output = Q6_V_vnot_V(input);
    Q6_V_st(out0 + i, output);
  }
}
```

It has the *shape* of HVX programming exactly right. But of the three intrinsics named,
**only the arithmetic one exists**:

| name | occurrences in the vendor headers |
|---|---|
| `Q6_V_vnot_V` | 2 |
| `Q6_V_ld` | **0** — fabricated |
| `Q6_V_st` | **0** — fabricated |

So the model holds the *compute* vocabulary and invents the *memory* vocabulary. It
cannot load or store a vector, which is precisely the knowledge `prompt.retry_facts`
exists to supply at rung 2 ("`Q6_Vsf_vmax_VsfVsf` being selectable while `V6_vadd_sf` is
not cannot be derived from anything").

The lint gate rejected both fabricated names before the compiler ran, so turn 2's
feedback was exactly that rejection. **The model responded by abandoning the accelerator
entirely:**

```cpp
extern "C" void candidate_kernel(const int32_t *v_args_0, int32_t *out0) {
  for (int i = 0; i < 6144; ++i) { out0[i] = ~v_args_0[i]; }
}
```

Correct, and completely scalar. **The correctness loop did not merely fail to produce
mechanism — it removed the only attempt at it.** A loop optimises for what it measures,
and at this rung the accelerator is not what is measured.

This also makes the hallucinated-intrinsic metric informative for the first time.
`rung0-results.md` says it "only becomes informative at rungs 1–3" because rung 0's zero
followed mechanically from never naming an intrinsic. It has now become informative, and
what it reports is a *load/store* gap rather than an arithmetic one.

**Open item, and it changes a table.** Mechanism engagement can be counted on the FINAL
candidate or on ANY turn, and rung 1 is the first rung where they differ: 0 attempts name
an intrinsic finally, 1 does at some point. Both are defensible; reporting only the final
count would erase the single most informative attempt in the study. Recommended: report
the final count as the headline and the any-turn count beside it. Note this is visible
only because superseded candidates are kept as `candidate_turn<N>.cpp` — with an
overwrite-in-place design the finding would not exist.

## The correctness result

| tier | rung 0 correct | rung 1 turn 1 | rung 1 final |
|---|---|---|---|
| T0 | 159/160 (99.4%) | — | **160/160 (100%)** |
| T1 | 160/160 (100%) | — | **160/160 (100%)** |
| T0+T1 pooled | 319/320 (99.7%) | 311/320 (97.2%) | **320/320 (100%)** |
| T2 | 145/145 (97.9%) | — | **145/145 (100%)** |
| T3 | 105/105 (96.2%) | — | **105/105 (100%)** |
| **pooled** | **562/570 (98.6%)** | — | **570/570 (100%)** |

**Every graded rung-1 attempt compiled and was correct, at every tier, and not one of
them touched the accelerator.** The per-seed spread is 114/114/114/114/114 — range
**zero**, which is the first time any metric in this study has been seed-invariant.
Mechanism engagement is 0 against every entitlement: hvx 0/570, hmx 0/105, dma 0/248,
vtcm 0/248, l2fetch 0/408.

Two things worth separating:

1. **The reference made the first turn WORSE.** 311/320 at turn 1 against rung 0's
   319/320 on the same tasks, and compile rate fell to 314/320 from 319/320. Handing over
   a correct implementation cost accuracy on the first attempt — the transcription
   introduces errors the model's own simpler code did not have.
2. **The turn loop then fixed all of it, in one wave.** 9 failures at turn 1, 9 correct
   at turn 2, and the wave settled before turn 3 (nothing left to retry).

Turn histogram over all graded attempts: **556 ended at turn 1, 12 at turn 2, none
needed a third.** The five-turn budget was never approached; two turns sufficed
everywhere, at every tier.

## One verdict correction, and what caused it

`scripts/finish_rung1.sh` passed `--timeout 3600` to the main T2/T3 grading pass but
**not** to the re-grade inside each retry wave, which therefore fell back to `grade`'s
900 s default. Two attempts of `b74i0_fp32__euclidean_dist__all` (seeds 1 and 3) hit that
cap and came out **ungraded**, leaving T3 at 103 against rung 0's 105.

Two things worth recording:

1. **The timeout guard did its job.** Those two were recorded as *ungraded because the
   budget expired*, not as wrong answers. Without the 2026-08-19 fix they would have been
   two `correct: false` verdicts, and rung 1 would have reported 568/570 (99.6%) with the
   two failures attributed to the model rather than to my script.
2. **A wave is the worst place to inherit a short default**, because it grades precisely
   the attempts a shorter budget has already failed. Fixed by threading a `SIM_TIMEOUT`
   (default 9000 s) through both call sites.

Re-graded at 9000 s: both **correct**. 9000 s rather than 3600 s because these are the two
most expensive attempts in the eval core — 9.98 B and 7.82 B instructions at rung 0, which
is ~64 and ~50 min at the measured 2.6 M instructions/s *before* rung 1's overhead, so
even an hour would have clipped them.

## No speed benefit either

Paired over the **319 attempts correct at BOTH rungs** (same task, same seed), rung-1
instruction counts against rung-0:

| tier | pairs | median | mean | >5% slower | >5% faster |
|---|---|---|---|---|---|
| T0 | 159 | 1.00× | 1.10× | 52 | 21 |
| T1 | 160 | 1.01× | 1.13× | 42 | 21 |
| T2 | 142 | 1.02× | 1.03× | 38 | 18 |
| T3 | 101 | 1.00× | 1.36× | 20 | 10 |
| **pooled** | **562** | **1.01×** | **1.14×** | 132 | 70 |

The median is unchanged at every tier and the mean is worse at every tier. T3's
mean of 1.36× against a median of 1.00× says the damage is a tail, not a shift: most
kernels cost the same and a few cost far more. Twice as many attempts got materially
slower (132) as got faster (70). Being shown a correct reference
made the model's kernels neither faster nor more hardware-engaged — only more likely to
be right on the second try.

## pass@k

Same estimator as rung 0, over the same 114 tasks with n = 5.

| k | correct | genuine | rung 0 correct | rung 0 genuine |
|---|---|---|---|---|
| 1 | 100.0% | **0.0%** | 98.6% | 0.5% |
| 2 | 100.0% | 0.0% | 99.9% | 1.0% |
| 3 | 100.0% | 0.0% | 100.0% | 1.3% |
| 4 | 100.0% | 0.0% | 100.0% | 1.6% |
| 5 | **100.0%** | **0.0%** | 100.0% | 1.8% |

**pass@5 genuine goes 1.8% -> 0.0%.** More scaffolding, more turns, and the metric the
thesis rests on went to zero at every k.

**Rung 1's pass@1 is NOT a single-shot number** and must not be tabulated against rung
0's without saying so: the record holds the verdict after up to 5 turns. The like-for-like
comparison is turn 1 alone:

| | rung 0 | rung 1 turn 1 | rung 1 final |
|---|---|---|---|
| pass@1 correct | 98.6% | **97.9%** | 100.0% |

Single-shot, rung 1 is *worse* than rung 0. The reference costs accuracy on the first
attempt and the loop then more than recovers it.

## Token cost, per task and per tier

640 attempts plus 12 retry turns: **776,613 in / 710,338 out**, of which **385,150 input
tokens (49.6%) were cache hits** and 546,205 output tokens (76.9%) were reasoning.

| | input | output |
|---|---|---|
| per task (5 seeds) | 6,067 | 5,550 |
| per attempt | 1,213 | 1,110 |
| per-attempt median | 1,036 (624-4,188) | 846 (221-3,186) |

| tier | in/task | out/task |
|---|---|---|
| T0 | 5,777 | 4,092 |
| T1 | 5,779 | 4,936 |
| T2 | 6,552 | 6,382 |
| T3 | 6,162 | 6,787 |

**Rung 1 costs 2.5x the input and LESS output than rung 0** (834,466 -> 710,338, with
reasoning down 20%). Handed a correct reference, the model reasons less. That is the
token-level signature of the same effect the mechanism result reports: it transcribes
rather than derives.

## Sampling: the five seeds are not five independent samples

Measured 2026-08-21, and it qualifies the seed-spread claim above.

| | duplicate candidates | wasted simulation |
|---|---|---|
| rung 0 | 72/640 (11.3%) -- T0 34, T1 23, T2 5, T3 10 | 2.68 B insns, ~17 min |
| rung 1 | 95/640 (14.8%) -- T0 30, T1 34, T2 10, T3 21 | 5.68 B insns, ~36 min |

"Duplicate" means byte-identical `candidate.cpp` for two seeds of the same task. All five
seeds of `fp32_acos` and `i32___and___Tensor` returned the same file. Seeds 1001-1005 ARE
sent, and the calls are genuinely separate (latency differs per call, so these are not
cached response replays) -- but `seed_honored` is `None` on all 1,280 attempts, so the
provider never confirmed honouring them.

Consequences, both worth stating in any paper:

1. **Effective sample size is below 5 for 11-15% of attempts**, so rung 1's "per-seed
   spread range 0" is partly identical outputs rather than purely stability. PLAN.md §2
   rests on "5 seeds minimum -- a single-seed number is not a result"; report distinct
   samples per task alongside the spread.
2. **A verdict cache keyed on candidate sha256 is free correctness** and reclaims ~36 min
   per rung, including 21 redundant T3 runs here. This corrects an earlier claim in this
   project that candidate dedup had zero headroom -- that measurement hashed a record
   field containing the per-seed file path, so identical candidates looked distinct.

## Two accounting defects, found the same day

Neither changes a verdict; both would corrupt a reported figure.

1. **`rung0.cost_usd` charges full price on cached input.** It computes
   `prompt_tokens * price_in`, so rung 1's 385,150 cached tokens are billed at the
   uncached rate. Needs a third rate: `(prompt - cached)*in + cached*cached_rate +
   completion*out`. The function's own docstring explains why this matters -- a wrong
   cost "looks measured".
2. **Multi-turn usage is undercounted at the top level.** `rung1.cmd_turn` records each
   retry turn's usage inside `turns[]` and does not fold it into the record's `usage`
   field, so anything summing `usage` misses it -- 24,533 in / 10,305 out here. Harmless
   at 12 turns, material at rungs 2-3 where turns are the point. Either accumulate into
   the top-level field or document that `usage` means turn 1.

**Prompt caching itself does not threaten the measurement.** It is prefix-KV reuse;
sampling happens after the prefix, sample independence is unaffected, and token counts
are identical either way. It does make LATENCY incomparable across rungs (rung 0 cached
0%, rung 1 49.6%) -- which bears on open item 3 in `rung0-results.md`, where the
`gpt-5.6-luna` variant was chosen partly on "lowest latency" in a 3-task pilot.

## What these numbers do NOT say

- **Scalar similarity is not comparable across rungs.** Rung 1's median is 0.2565 (max
  0.9022) against rung 0's 0.136. That is expected and uninteresting: **rung 1 shows the
  reference.** Rung 0's low similarity was the recorded answer to the "you showed it
  scalar code" objection, and that answer belongs to rung 0 **only**. Any table printing
  both must say which rung each describes.
- **The T2/T3 correctness rows are unfinished**, not failed. `aggregate.summarise`
  refuses to fold ungraded attempts into a rate.
- **One model, two rungs.** Only an OpenAI key is available. There is no rung-2/3 delta
  yet, and no cross-model comparison.
- **No cycle claim.** Grading runs without `--timing`; the paired instruction counts
  above are simulated instruction totals, not cycles.

## Cost

640 attempts, 0 generation failures: **752,080 prompt + 700,033 completion tokens**.
Prompt tokens are 2.5× rung 0's 304,825 — the reference is in every prompt, and a retry
turn restates the ask plus the previous kernel. Completion tokens are slightly *lower*
than rung 0's 834,466, consistent with a model transcribing a given reference rather
than deriving one.

## Reproducing it

```bash
python -m hexkernels.forge.rung0 generate --rung 1 --model gpt-5.6-luna --seeds 5 --jobs 8
bash scripts/finish_rung1.sh gpt-5.6-luna 5 5    # grade, waves, sweep, report
python -m hexkernels.forge.rung1 histogram --model gpt-5.6-luna
```

`finish_rung1.sh` sweeps **last**, deliberately: the sweep reads each attempt's current
`candidate.cpp`, and a retry wave rewrites that file. An interim sweep taken before the
waves describes candidates that no longer exist — which is how the `b37i3` intrinsic
attempt above first surfaced as an apparent contradiction between the scan and the
verdict.
