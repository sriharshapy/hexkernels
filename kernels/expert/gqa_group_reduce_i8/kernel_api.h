#ifndef KERNEL_API_H
#define KERNEL_API_H
#include <stdint.h>
/*
 * Grouped-Query-Attention HEAD REDUCTION: H head score/output maps are
 * reduced down to G group-averaged (or weighted-reduced) maps by summing
 * the HPG = H/G heads within each group at every spatial position, then
 * requantizing. This is the "head-merge" building block that sits before
 * (or after) the QKT/softmax/AV steps of a full GQA attention block --
 * it does NOT itself compute attention scores.
 *
 * X: [H][M*N] int8, contiguous (X[h*M*N + idx], idx in [0, M*N)).
 * Y: [G][M*N] int8, contiguous (Y[g*M*N + idx]).
 * Group g covers heads [g*HPG, g*HPG+HPG), HPG = H/G (H, G, M, N are all
 * RUNTIME parameters -- read HPG = H/G at runtime, do not hardcode any
 * specific HPG value).
 *
 * Pinned formula, per group g and spatial position idx in [0, M*N):
 *   acc = sum_{t=0}^{HPG-1} (int32)X[(g*HPG+t)*M*N + idx]
 *   r    = (int64_t)acc * (int64_t)scale_mult
 *   half = (scale_shift > 0) ? (1LL << (scale_shift-1)) : 0
 *   q    = (r >= 0) ? ((r + half) >> scale_shift)
 *                   : -((-r + half) >> scale_shift)     (round-half-away-from-zero)
 *   Y[g*M*N + idx] = clamp(q, -128, 127)
 *
 * scale_mult: int32, always POSITIVE for this task's runtime sweep (so
 * sign(q) always matches sign(acc)). scale_shift: int (>=0). Both are
 * RUNTIME parameters, swept over multiple sets by the harness -- including
 * one that is exactly the group AVERAGE (scale_mult=1, scale_shift=2 for
 * HPG=4) and one that is a different weighted reduce (scale_mult=3,
 * scale_shift=4). Do NOT hardcode the scale for a specific HPG -- read
 * scale_mult/scale_shift at runtime.
 *
 * H=8, G=2, HPG=4, M=8, N=17 (M*N=136, NOT a multiple of 128 -- tail path).
 * HPG*127 = 508 fits comfortably in int16, so no int8 overflow risk when
 * accumulating the HPG head-planes before requantizing.
 */
void candidate_kernel(const int8_t *X, int8_t *Y, int H, int G, int M, int N,
                      int32_t scale_mult, int scale_shift);
#endif /* KERNEL_API_H */
