Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *X, int8_t *Y, int H, int G, int M, int N,
                          int32_t scale_mult, int scale_shift);

Grouped-Query-Attention HEAD REDUCTION: H head score/output maps are reduced
down to G group-averaged (or weighted-reduced) maps by summing the HPG=H/G
heads within each group at every spatial position, then requantizing.

X: [H][M*N] int8, contiguous (X[h*M*N + idx], idx in [0, M*N)).
Y: [G][M*N] int8, contiguous (Y[g*M*N + idx]).
Group g covers heads [g*HPG, g*HPG+HPG). Read HPG = H/G at RUNTIME -- do not
hardcode a specific HPG value in the kernel logic.

Pinned formula, per group g and spatial position idx in [0, M*N):
  acc = sum_{t=0}^{HPG-1} (int32)X[(g*HPG+t)*M*N + idx]
  r    = (int64_t)acc * (int64_t)scale_mult
  half = (scale_shift > 0) ? (1LL << (scale_shift-1)) : 0
  q    = (r >= 0) ? ((r + half) >> scale_shift)
                  : -((-r + half) >> scale_shift)     (round-half-away-from-zero)
  Y[g*M*N + idx] = clamp(q, -128, 127)

H=8, G=2, HPG=4, M=8, N=17 (M*N=136, NOT a multiple of 128 -- handle the
tail). scale_mult (always positive) and scale_shift (>=0) are RUNTIME
parameters swept over multiple sets, including one that is exactly the
group AVERAGE and one that is a different weighted reduce -- read them at
runtime, do not hardcode the scale for a specific HPG.

NOTE: the head-plane stride is M*N=136 bytes, NOT a multiple of 128, so a
head plane's base pointer is generally not 128-byte aligned -- stage each
128-byte chunk through an aligned local buffer before an HVX vector load.

Prefer HVX vector intrinsics; include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT use floating point. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
