Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *a, int8_t *out, int n);

Per-channel normalize + requantize (int8 -> int8). Layout is [NUM_CH][per_ch] with
per_ch = n / NUM_CH. All parameters are FIXED constants (bake them in; NOT passed):
    NUM_CH = 16
    NORM_MULT[16]  = { 3, 5, 7, 9, 11, 13, 15, 17, 3, 5, 7, 9, 11, 13, 15, 17 }
    NORM_SHIFT[16] = { 2, 3, 4, 5,  3,  4,  5,  6, 2, 3, 4, 5,  3,  4,  5,  6 }
    MULT = 5, SHIFT = 4, ZP = 0
For channel c (0..15) and element i (idx = c*per_ch + i):
    norm = round_half_away( a[idx]*NORM_MULT[c] >> NORM_SHIFT[c] )
    v    = norm * MULT
    half = (SHIFT > 0) ? (1 << (SHIFT-1)) : 0
    r    = round_half_away( v >> SHIFT ) + ZP
    out[idx] = clamp(r, -128, 127)
round_half_away means add (1<<(shift-1)) to the magnitude before the arithmetic shift.
a is int8 in [-128,127]; every intermediate fits int16.

n is large (a[] exceeds L2), so the task is DDR-bandwidth-bound. To go fast you
should hide DDR latency: stage tiles of a[] into VTCM with the uDMA engine and
compute on the on-chip copy, double-buffering so the next tile's DMA overlaps the
current tile's compute; DMA the int8 results back to DDR. VTCM is identity-mapped at
0xd8400000. uDMA: build a static/global Type-0 descriptor {next, ctrl=len, src, dst},
then Q6_dmstart_A(&desc) / Q6_R_dmwait(). The descriptor MUST be static/global (a
stack descriptor no-ops at -O2). n is a multiple of 128; per_ch = n/16 is a multiple
of your tile size, so each tile falls in one channel (its norm params are constant).

Sign-extend int8 -> int16 (Q6_Wh_vunpack_Vb) and compute in halfword lanes: multiply
by the channel's NORM_MULT (Q6_Vh_vmpyi_VhVh), round half-away (Q6_Vh_vabs_Vh; add
half; Q6_Vh_vasr_VhR by NORM_SHIFT; restore sign with Q6_Q_vcmp_gt_VhVh +
Q6_V_vmux_QVV), then the requant multiply by MULT with the same round-half-away >>
SHIFT, add ZP, and saturating-pack halfword -> int8 with Q6_Vb_vpack_VhVh_sat
(high-lane arg first for natural order). Include <hexagon_types.h> and
<hexagon_protos.h>. Do NOT write main(). Respond with a single complete C code block
and CLOSE the fence with ```.
