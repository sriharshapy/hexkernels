# Rung 2 — measured results

**Status: COMPLETE** (2026-08-26). 570 of 640 attempts graded — the same 570 as rungs 0
and 1, so every tier is paired across all three rungs. The 70 ungraded are the tasks
whose reference never passed stage (g), identical to both earlier rungs.

- Model `gpt-5.6-luna` · eval set `core128` · schema v1 · 5 seeds · rung 2
- Generated 2026-08-25 (128/128 tasks, 640 attempts, 0 failures)
- Prompt: rung 1's ask **plus hardware facts, the MLIR Linalg, the loop schedule with
  `vectorizable_loop`, the mechanism budget, and the full `CONTRACT`** — 1,827 → 2,812 →
  **7,488** bytes for the same task across rungs 0, 1 and 2 (`prompt.build_rung2`)
- Up to 5 turns; a turn carries `model_client.feedback` **in full** — provenance leak →
  lint errors + `ISA_FACTS` → compiler output → simulator fault → wrong values +
  `verify.diagnose` hypotheses + `ISA_FACTS`. This is everything rung 1 was denied
- Design and seven unratified calls:
  [`superpowers/specs/2026-08-25-rung2-design.md`](specs/2026-08-25-rung2-design.md)

PLAN.md §2 asks of this rung: **does verification-grounded generation reach the
hardware?** The answer is **yes, and only through the retry channel.** PLAN.md §6's
rung-2 kill criterion does not fire.

## The mechanism result

| | rung 0 | rung 1 | **rung 2, turn 1** | **rung 2, final** |
|---|---|---|---|---|
| attempts scanned | 636 | 640 | 640 | 640 |
| attempts **naming** any `Q6_` intrinsic | 0 | 0 final / 1 any-turn | **52** | **20** |
| attempts where an entitled mechanism **fired** | 3 | 0 | 1 | **22** |
| — of those, model-written intrinsics | **0** | **0** | **0** | **20** |
| — of those, compiler auto-vectorisation | 3 | 0 | 1 | 2 |
| attempts that would not compile | 4 | 0 | **64** | 4 |
| hallucinated intrinsic names | 0 | 2 | **121 instances / 29 distinct** | **0** |

Two separate findings live in that table, and they should not be merged.

**The first prompt produces intent, and intent alone.** Rung 2's opening ask — with the
cache sizes, the VTCM base, the entitled mechanisms and the vectorisable axis, and still
no intrinsic named anywhere in it — moved *naming* an intrinsic from 0 attempts in 1,276
across rungs 0 and 1 to **52 of 640**. It converted none of them: model-written mechanism
engagement at turn 1 is **0**, exactly as at rungs 0 and 1, and 64 attempts would not even
compile. Scaffolding alone makes the model reach for the accelerator and miss.

**The retry channel converts it.** By the final candidate, 22 attempts fire an entitled
mechanism and **20 of those contain intrinsics the model wrote**. Every one arrived at
turn 2 or later — that is, after `ISA_FACTS`. The measured-ISA note is the difference
between reaching and arriving.

### The load/store gap, confirmed and then closed

Rung 1 had exactly one attempt that named an intrinsic, and it predicted this rung's
failure mode from a sample of one: `b37i3_i32_bitwise_not` held the *compute* vocabulary
(`Q6_V_vnot_V`, real) and invented the *memory* vocabulary (`Q6_V_ld`, `Q6_V_st`, neither
in the vendor headers). At 640 attempts the prediction holds and the fabrications are
overwhelmingly loads and stores:

| fabricated name | instances at turn 1 |
|---|---|
| `Q6_V_vldu_A` | 36 |
| `Q6_V_vstu_AV` | 28 |
| `Q6_V_vstu` | 8 |
| `Q6_V_vst` | 7 |
| `Q6_V_vldu` | 7 |
| `Q6_V_vld` | 4 |

121 instances over 29 distinct names, and **the top six are all vector loads or stores.**
The model can name what to compute and cannot name how to move the data.

**How it resolves is the interesting part: the model stops trying.** Not one of the 20
model-written genuine kernels uses a load or store intrinsic. They use ordinary pointer
access for memory and intrinsics only for compute — `Q6_V_vand_VV`,
`Q6_Vqf32_vsub_VsfVsf`, `Q6_Vsf_vmax_VsfVsf`, `Q6_V_vmux_QVV`,
`Q6_Q_vcmp_gt_VsfVsf`. The fabricated-name count goes **121 → 0**: every invented name is
gone by the final candidate, and the kernels that survive are the ones that stopped
asking for a vector load at all.

`b37i3_i32_bitwise_not` seed 4 is among the 20, firing `hvx` with `Q6_V_vsplat_R` and
`Q6_V_vxor_VV`. The single attempt that defined rung 1's failure is a rung-2 success on
the same task.

### The retreat, measured on a population

Rung 1's one intrinsic attempt was rejected by the lint gate and the model responded by
abandoning the accelerator entirely. That behaviour is now measurable:

| | count |
|---|---|
| attempts naming an intrinsic at turn 1 | 52 |
| still naming one at the final candidate | **20** |
| retreated to scalar | **32** |
| of the 20 that persisted, how many fire a mechanism | **20** |

**Every attempt that kept its intrinsics ended with a mechanism firing.** The rung's
failure mode is not intrinsics that do not work — it is intrinsics abandoned after the
first rejection, and it costs 32 of 52. Handing over ~15 correct intrinsic names does not
prevent the retreat; it is still the majority response.

### Per-turn curve

Only attempts that took more than one turn appear beyond turn 1 (`sweep --all-turns`,
reading the preserved `candidate_turn<N>.cpp` files).

| turn | candidates scanned | named a `Q6_` intrinsic | mechanism fired |
|---|---|---|---|
| 1 | 640 | 52 | 1 |
| 2 | 68 | 25 | 10 |
| 3 | 18 | 11 | 8 |
| 4 | 5 | 4 | 4 |
| 5 | 1 | 0 | 0 |

Turn 1 is the only measurement in this rung taken **before** any intrinsic name is
supplied, which is what makes the 52-with-0-firing row a clean statement about the first
prompt.

### Genuine, by tier

| tier | graded | genuine | of which model-written |
|---|---|---|---|
| T0 | 160 | 10 | **10** |
| T1 | 160 | 7 | **7** |
| T2 | 145 | 1 | 0 |
| T3 | 105 | 4 | **3** |
| **pooled** | **570** | **22 (3.9%)** | **20 (3.5%)** |

The two auto-vectorisation confounds are `b54i1_fp32_pairwise_distance` seed 2 (T2, turn
1) and `b66i1_i32___xor___Tensor` seed 2 (T3, turn 2) — correct, firing `hvx`, containing
no intrinsic. They are reported separately for the reason `rung0-results.md` gives: the
anti-cheat reads the ELF and cannot tell who caused the instruction to exist, and the
paper's claim is about the model.

`b66i1_i32___xor___Tensor` is the cleanest illustration in the corpus of why the split
matters: **seed 2 fired `hvx` by auto-vectorisation and seed 4 fired it with a real
intrinsic — same task, same tier, same turn.** Merged, they are two identical successes.

**T2 is the one tier with no model-written mechanism.** T0 and T1 account for 17 of the
20. Whether that is a size effect or a task-mix effect is not answerable from this run.

### The genuine count is NOT seed-stable, and this is the rung's weakest number

| seed | 1 | 2 | 3 | 4 | 5 | range |
|---|---|---|---|---|---|---|
| genuine | 3 | 6 | 3 | 8 | 2 | **6** |
| model-written | 3 | 4 | 3 | 8 | 2 | **6** |

Correctness is nearly seed-invariant (113–114 of 114 per seed, range 1). Mechanism
engagement is not: **2 to 8 per seed, a fourfold spread on a mean of 4.4.** PLAN.md §2
warns that identical re-runs have moved a single mechanism ±3, and this rung reproduces
exactly that instability at exactly that magnitude.

Two consequences, and neither is optional:

1. **No single-seed mechanism number from this rung is reportable.** The pooled 22 rests
   on 5 seeds and should always be quoted with the per-seed range beside it.
2. **The rung-1 → rung-2 comparison survives this and the rung-0 → rung-2 one is thinner
   than it looks.** Rung 1's genuine count was 0 on every seed, so 2-to-8 against 0-to-0
   is a real difference at any seed. Against rung 0's 3 auto-vectorisations, the
   *model-written* comparison (0 on every rung-0 seed) is the one that holds; the raw
   genuine counts overlap.

## The correctness result

| tier | rung 0 correct | rung 1 final | **rung 2 turn 1** | **rung 2 final** |
|---|---|---|---|---|
| T0 | 159/160 (99.4%) | 160/160 | — | **160/160 (100%)** |
| T1 | 160/160 (100%) | 160/160 | — | **160/160 (100%)** |
| T0+T1 pooled | 319/320 (99.7%) | 320/320 (100%) | **274/320 (85.6%)** | **320/320 (100%)** |
| T2 | 142/145 (97.9%) | 145/145 | — | **144/145 (99.3%)** |
| T3 | 101/105 (96.2%) | 105/105 | — | **105/105 (100%)** |
| **pooled** | **562/570 (98.6%)** | **570/570 (100%)** | — | **569/570 (99.8%)** |

**Rung 2 pays for its mechanism engagement in first-attempt correctness and gets almost
all of it back.** T0+T1 at turn 1 is 274 of 320 — against rung 0's 319 and rung 1's 311 on
the same attempts. Compile rate at turn 1 was 277/320 (86.6%) where rung 1 managed
314/320. The model is attempting something harder and failing at it, then the loop repairs
it.

**Why it fails is the finding, not that it fails.** Of the 46 T0/T1 failures at turn 1:

| stage | count |
|---|---|
| `lint` — refused by the static gate before compiling | **36** |
| `compile` | 7 |
| `wrong` — ran and produced wrong numbers | **3** |

Only 3 of 46 were wrong *answers*. 36 never ran at all, refused for fabricated intrinsic
names. Rung 2 does not fail at arithmetic; it fails at vocabulary — and vocabulary is the
one thing a retry can supply.

### Turn histogram and the full taxonomy

**572 attempts ended at turn 1, 50 at turn 2, 13 at turn 3, 4 at turn 4, 1 at turn 5.**
Rung 2 is the first rung to reach the five-turn cap, and the attempt that reached it is
the rung's only remaining failure.

Every retry turn ever spent (92 in total), by the stage that provoked it:

| stage | turns |
|---|---|
| `lint` | **68** |
| `wrong` | 13 |
| `compile` | 11 |

**74% of all retry effort in this rung went to vocabulary, not to correctness.** That
distinction has to be stated wherever "rung 2 needed more turns than rung 1" appears: a
`lint` rejection is *our* policy verdict on a fabricated name, not the model failing to
compute the right answer. Pooled, the two would read as one difficulty number.

The wave-by-wave shift is the repair in progress: T0/T1 went `lint=36, compile=7, wrong=3`
→ `lint=12, compile=2` → `lint=2, wrong=1`, with the failing set shrinking 46 → 14 → 3.

### The one attempt the loop could not fix

`b57i0_fp16_scaled_dot_product_attention` seed 4 (T2) spent all five turns and ended
incorrect. **All four of its retries were provoked by `wrong`** — never `lint`, never
`compile`. It is a genuine numerical failure (12,587 of 196,608 elements wrong, bad stride
26), and it is the only attempt in the rung whose problem was arithmetic all the way
down. The single failure in 570 is the one that had nothing to do with vocabulary.

## pass@k

Same estimator as rungs 0 and 1, over the same 114 tasks with n = 5.

| k | correct | genuine | rung 1 genuine | rung 0 genuine |
|---|---|---|---|---|
| 1 | 99.8% | **3.9%** | 0.0% | 0.5% |
| 2 | 100.0% | 5.7% | 0.0% | 1.0% |
| 3 | 100.0% | 6.9% | 0.0% | 1.3% |
| 4 | 100.0% | 7.9% | 0.0% | 1.6% |
| 5 | 100.0% | **8.8%** | 0.0% | 1.8% |

**pass@5 genuine goes 1.8% → 0.0% → 8.8%** across the three rungs. Unlike rung 0's, this
genuine column *is* a capability curve: rung 0's rise from 0.5% to 1.8% was three compiler
auto-vectorisations over two tasks, whereas 20 of rung 2's 22 are kernels the model wrote.

**Rung 2's pass@1 is not a single-shot number**, for the same reason rung 1's was not: the
record holds the verdict after up to 5 turns. The like-for-like single-shot comparison is
turn 1 alone, and on the tiers where a turn-1 figure exists (T0+T1) it is **85.6% correct
and 0.0% model-written genuine** — worse than either earlier rung on correctness, and
identical to both on mechanism.

## No speed benefit, and no speed cost

Paired over attempts correct at both rungs (same task, same seed), rung-2 instruction
counts against each earlier rung:

| | pairs | median | mean | >5% slower | >5% faster |
|---|---|---|---|---|---|
| vs rung 0 | 561 | **1.00×** | 1.17× | 147 | 85 |
| vs rung 1 | 569 | **1.00×** | 1.03× | 84 | 94 |

Against rung 1 the distribution is symmetric — 84 slower against 94 faster, mean 1.03×.
**Rung 2's kernels are not faster than rung 1's**, even though 20 of them now use vector
compute intrinsics. Two reasons this is unsurprising and one caveat:

1. The 20 genuine kernels are 3.5% of the population, far too few to move a pooled median.
2. Vector *compute* with scalar loads and stores is not the same optimisation as a
   vectorised loop; the memory traffic is unchanged.
3. **These are simulated instruction totals, not cycles.** Grading runs without
   `--timing` by the 2026-08-17 ratification. No cycle claim can be made here at all, and
   the mechanism result is precisely the thing instruction counts cannot see.

That last point is the thesis in miniature: a benchmark reporting only speed would see
rung 1 and rung 2 as indistinguishable (median 1.00×, mean 1.03×), while the mechanism
metric separates 0 model-written kernels from 20.

## Token cost, per task and per tier

640 attempts plus 92 retry turns: **2,165,774 in / 1,079,063 out**, of which **1,444,758
input tokens (66.7%) were cache hits** and 872,676 output tokens (80.9%) were reasoning.

| | rung 0 | rung 1 | rung 2 |
|---|---|---|---|
| input | 304,825 | 776,613 | **2,165,774** |
| — cached share | 0% | 49.6% | **66.7%** |
| output | 834,466 | 710,338 | **1,079,063** |
| — reasoning share | 82.0% | 76.9% | **80.9%** |
| per attempt | 476 / 1,304 | 1,213 / 1,110 | **3,384 / 1,686** |

| tier | in/task | out/task |
|---|---|---|
| T0 | 16,152 | 7,548 |
| T1 | 16,581 | 8,598 |
| T2 | 16,886 | 7,972 |
| T3 | 18,060 | 9,601 |

**Rung 2 reverses rung 1's token signature.** Rung 1 produced *less* output than rung 0
(834,466 → 710,338, reasoning down 20%), which that document reads as the trace of a model
transcribing a handed-over reference rather than deriving one. Rung 2 puts output back up
45% over rung 1 and reasoning share back to rung-0 levels. Given the apparatus, the model
derives again.

**68 records carry folded multi-turn usage.** Under rung 1's accounting those turns were
recorded inside `turns[]` and never added to the top-level `usage`, so anything summing
that field missed them. This rung fixes it (`rung2._add_usage`, `rung0.reprice`), and the
68 records are where the difference would have shown.

**No dollar figure.** This run was generated unpriced, as rungs 0 and 1 were: there is no
price table in this project by design, and inventing a rate for `gpt-5.6-luna` would be
the fabricated-number failure `cost_usd`'s own docstring refuses. Tokens are the currency
comparable across rungs. The cached-rate fix and `reprice` are tested and ready for when a
rate is known.

## Sampling

| | duplicate candidates | share |
|---|---|---|
| rung 0 | 72/640 | 11.3% |
| rung 1 | 95/640 | 14.8% |
| **rung 2** | **53/640** | **8.3%** |

"Duplicate" means byte-identical `candidate.cpp` for two seeds of the same task. Rung 2's
rate is the **lowest** of the three rungs: the larger prompt samples more diversely. This
weakens rather than strengthens the case for the sha256 verdict cache proposed in
`rung1-results.md` — the wasted simulation it would reclaim is smaller here than at either
earlier rung.

`seed_honored` remains `None` on every attempt, as at rungs 0 and 1: seeds 1001–1005 are
sent and the calls are genuinely separate, but the provider never confirms honouring them.
Report distinct samples per task alongside any seed-spread claim.

## Methods note: the first generation of this rung was discarded

The first 176 attempts of rung 2 were generated from a prompt that **withheld a fact rungs
0 and 1 both supply.** `RUNG0_CONTRACT` (used by rungs 0 and 1) states that every buffer is
dense, row-major and non-overlapping; `CONTRACT` (which `prompt.build()` uses, and which
rung 2 inherited) does not, and no other section of the rung-2 prompt said it. Rung 2 was
therefore not a superset of rung 1 — it granted an aliasing and layout licence *less*
freely, which is exactly the kind of fact that decides whether a model believes a
contiguous vector store is legal.

Caught by review before grading, fixed by carrying the guarantee explicitly
(`prompt.BUFFER_GUARANTEE`), and the affected generation was discarded and re-run from
scratch. The 37 prompts it had written are committed at
`runs/rung2.discarded-pre-C2/gpt-5.6-luna/prompts/` — they are the evidence, and a diff
against the re-run's prompt for the same task shows the missing sentence. The 176
candidates generated from them are retained locally but not committed; no number in this
document comes from any of it.

Two guards now exist that did not before:

1. `test_rung2.py::test_it_is_a_superset_of_rung1` asserts every semantic part rung 1
   carries is present in rung 2 — the signature, the operator identity, the whole buffer
   table, the guarantee, and the complete reference body. Its earlier form checked a
   length comparison plus hand-picked fields, which is why it missed this.
2. `test_rung2.py::test_the_first_prompt_names_no_intrinsic` is the guard `prompt.py` has
   always *cited* and this repository never had — PLAN.md §5.A left the old suites behind,
   and the name occurred nowhere but in two comments. It now covers all four first-prompt
   paths, including `build()`, which produces the corpus prompt.

## What these numbers do NOT say

- **The 22 firing attempts all fired `hvx`.** Not one fired `hmx`, `dma`, `vtcm` or
  `l2fetch`. Mechanism engagement at this rung means vector compute and nothing else, and
  the entitlements that exist only at the large tiers remain untouched: hmx 0/105, dma
  0/250, vtcm 0/250, l2fetch 0/410.
- **HMX still has no achievability witness** (0 of 15), so a rung-2 HMX attempt would have
  had no demonstrated ceiling to be measured against even if one had appeared.
- **No cycle claim.** Grading runs without `--timing`; the paired figures above are
  simulated instruction totals.
- **The T2/T3 rows rest on 145 and 105 attempts**, not 160. The same 70 attempts are
  ungraded at every rung, blocked on their reference rather than on simulator time.
- **One model, three rungs.** Only an OpenAI key is available. There is no cross-model
  comparison and no rung-3 delta.
- **`genuine` counted on the final candidate.** The any-turn count is identical for
  firing (22 either way) and differs sharply for naming (52 any-turn against 20 final);
  both are reported above because the gap between them *is* the retreat finding.

## Reproducing it

```bash
python -m hexkernels.forge.rung0 generate --rung 2 --model gpt-5.6-luna --seeds 5 --jobs 8
python -m hexkernels.forge.rung0 sweep    --rung 2 --model gpt-5.6-luna   # the turn-1 headline, no simulator
bash scripts/finish_rung2.sh gpt-5.6-luna 5 5                     # grade, waves, final sweep, report
python -m hexkernels.forge.rung0 sweep    --rung 2 --model gpt-5.6-luna --all-turns
python -m hexkernels.forge.rung2 histogram --model gpt-5.6-luna
```

`finish_rung2.sh` takes the mechanism sweep **before** any grading, because it needs no
simulator (~150 s for 640 attempts) and it is the headline — the rung's central number
exists before any simulator hours are committed. The **final** sweep passes `--redo`,
which is mandatory there and absent from the early one: `cmd_sweep` skips attempts that
already carry the field it writes, so without it the post-retry sweep would silently scan
nothing and `mechanism_scan` would describe turn-1 candidates while the verdicts described
post-retry ones.
