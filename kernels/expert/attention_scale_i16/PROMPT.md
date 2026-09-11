Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int32_t *raw, int16_t *out, int M, int N,
                          int32_t scale_mult, int scale_shift);

Elementwise fixed-point rescale of raw int32 attention scores to int16,
approximating multiplication by 1/sqrt(d). raw/out are [M x N] row-major,
but the operation is purely elementwise (no cross-element dependency).

Pinned formula, per element idx in [0, M*N):
  r    = (int64_t)raw[idx] * (int64_t)scale_mult
  half = (scale_shift > 0) ? (1LL << (scale_shift-1)) : 0
  q    = (r >= 0) ? ((r + half) >> scale_shift)
                  : -((-r + half) >> scale_shift)     (round-half-away-from-zero)
  out[idx] = clamp(q, -32768, 32767)

M=16, N=37 (592 elements total; N=37 is NOT a multiple of 32
int32-lanes-per-HVX-vector -- handle the tail). scale_mult (always positive)
and scale_shift (>=0) are RUNTIME parameters swept over multiple sets by the
harness, including scale_shift=0 (no rounding bias) -- read them at runtime,
do not hardcode. |raw[idx]| is bounded (<= 2,000,000) and scale_mult <= 500
by construction, so |raw[idx]|*scale_mult stays well within int32/int64
range.

Prefer HVX vector intrinsics; include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT use floating point. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
