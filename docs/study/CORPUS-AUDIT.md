# Test set audit

320 tasks; 128-task evaluation core.

## Tier counts

- T0: 80
- T1: 80
- T2: 80
- T3: 80

## Mechanism counts

| mechanism | tasks | minimum |
|---|---|---|
| hvx | 320 | 320 |
| l2fetch | 240 | 60 |
| dma | 160 | 60 |
| vtcm | 160 | 60 |
| hmx | 32 | 24 |

## Composition (single-op vs fused)

This selection's fused share, computed from the data below, is 32% (104 of 320). That is a different measurement from the pool-level figure the fused top-up decision was made against: the mined pool's fused share was deliberately raised from ~34% to ~44% so the selection could reach its per-tier quota once the mechanism minimums were also satisfied. (That 34%/44% pair is historical -- the share of the pool at the time the top-up decision was made -- and is not derivable from today's pool or selection; do not mistake it for a computed figure.) A reader must be able to see the selection's own fused share and discount it. `fused` is derived from the presence of a `stages` key, never from the schema string.

By tier:

| tier | single-op | fused |
|---|---|---|
| T0 | 49 | 31 |
| T1 | 77 | 3 |
| T2 | 46 | 34 |
| T3 | 44 | 36 |

By mechanism:

| mechanism | single-op | fused |
|---|---|---|
| hmx | 32 | 0 |
| hvx | 216 | 104 |
| l2fetch | 167 | 73 |
| dma | 90 | 70 |
| vtcm | 90 | 70 |

## Depth (n_primitives) by tier

- T0: {1: 28, 2: 7, 3: 6, 4: 5, 5: 2, 6: 7, 7: 4, 8: 2, 9: 3, 10: 1, 11: 3, 12: 1, 13: 4, 14: 3, 16: 1, 18: 1, 20: 1, 164: 1}
- T1: {1: 42, 2: 1, 3: 5, 4: 3, 5: 3, 6: 7, 7: 2, 8: 3, 9: 5, 10: 1, 11: 2, 12: 1, 13: 1, 17: 1, 20: 1, 21: 1, 41: 1}
- T2: {1: 20, 2: 9, 3: 5, 4: 7, 5: 3, 6: 8, 7: 4, 8: 4, 9: 2, 10: 2, 11: 1, 12: 3, 13: 1, 14: 2, 15: 2, 16: 3, 17: 1, 18: 1, 20: 1, 21: 1}
- T3: {1: 18, 2: 12, 3: 7, 4: 3, 6: 9, 7: 4, 8: 2, 9: 6, 10: 2, 11: 3, 12: 2, 13: 3, 14: 2, 15: 2, 16: 1, 17: 1, 20: 1, 21: 1, 162: 1}

Note: n_plumbing is 0 on every spec by construction (mine.py:191 _would_be_destroyed rejects any candidate containing a non-primitive_admissible node before it can be picked). It carries no information and is not reported as a statistic here.

## Negative controls (fp32 contractions, hmx must be absent)

8: _scaled_dot_product_attention_math.default.float32.T0, rnn_tanh_cell.default.float32.T0, _convolution.default.float32.T1, rnn_relu_cell.default.float32.T1, scaled_dot_product_attention.default.float32.T2, kron.default.float32.T2.softplus.default, kron.default.float32.T2.softshrink.default, linear.default.float32.T3

## Known-absent required operators

Not the same kind of absence -- two are correct exclusions and one is a defect. See `hexkernels.forge.audit.KNOWN_ABSENT`.

- `amax` (absent): correctly excluded -- already in the corpus as dedicated hand-written kernels (kernels.py:573 fp32_amax_all, :1480 fp32_amax_hw); mine.py:770 rejects it as already-in-corpus, so re-mining it would duplicate an existing task.
- `amin` (absent): correctly excluded by the letter of the same rule, and worth flagging as arguable -- amin is 'used' only as half of fp32_aminmax (kernels.py:734), which traces to exactly two nodes. The dedup rule treats 'appears anywhere in an existing kernel's decomposition' the same as 'was mined as its own task', so no standalone amin task can ever exist.
- `gelu` (absent): a real defect, not a correct exclusion -- nothing rejects it: it is mechanism-eligible, absent from corpus_targets() (the existing fp16_gelu is CompositeImplicit and expands before any aten.gelu node exists), sizes at all four tiers with hvx granted, and collides with no signature. It is missing only because a mining run to recover it could not be completed in this environment. frac is absent with the same symptom, though frac is not a REQUIRED_OPS entry.

## Methodological note: mining is incremental, not idempotent

kernels.all_batches() is HANDWRITTEN_BATCHES union mined.MINED_BATCHES, and hexkernels/forge/mined.py builds MINED_BATCHES from benchmark/selection.json. corpus_targets() and corpus_signatures() both walk it -- so selection.json feeds back into the miner's own exclusion set. Once the selection exists, a re-mine skips everything already in it and yields only what is new: the pool is the union of successive mining runs under a growing exclusion set, and reproducing from scratch would give a different, larger first run. That is defensible, but it must be stated rather than discovered.

## Violations

None — the set is ready to freeze.
