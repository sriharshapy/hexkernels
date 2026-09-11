Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int16_t *x, int16_t *out, int R, int C, const uint16_t *exp_lut);

Row-wise softmax over a [R x C] int16 matrix, independently per row r:
  1. m_r    = max(x[r*C + j], j=0..C-1)                          (int16)
  2. idx_j  = clamp((int32_t)(x[r*C+j] - m_r), -255, 0) + 255     (in [0,255])
  3. e_j    = exp_lut[idx_j]                                      (uint16, runtime table -- do NOT hardcode)
  4. S_r    = sum of e_j over j in [0,C)                          (int64 accumulator -- NOT int32:
                                                                    C is large enough that S_r can reach
                                                                    ~65535*C, which OVERFLOWS int32)
  5. out[r*C+j] = (int16_t)(((int64_t)e_j*32767 + S_r/2) / S_r)   (round-half-down, int64 arithmetic throughout)

Fixed sizes: R=10, C=65500 (C is NOT a multiple of 128 -- tail path). x and
out are each ~1.31MB total (exceeds L2), so this is DDR-bandwidth-bound.

The LUT lookup is inherently scalar (a runtime table read per element --
vectorized memory gather is not available/safe here). To go fast: DMA the
WHOLE row (C*2 bytes) from DDR into a VTCM buffer ONCE, then do all three
passes (max-scan, sum+cache e_j, normalize) entirely on-chip, then DMA the
finished output row back to DDR ONCE. This turns ~4 DDR passes per row into 2
bulk DMA transfers per row (1 in, 1 out). Double-buffer across rows if you
can: prefetch row r+1's DMA-in while row r's on-chip passes + DMA-out run,
hiding DDR latency behind compute.

Scalar VTCM accesses cost ~48 cycles EACH in timing mode (far worse than a
scalar DDR access) -- so the three scalar compute passes must NEVER touch a
VTCM address directly. Bulk VECTOR-copy (128B at a time) each row's data
out of VTCM into a small cacheable scratch buffer first, run all three passes
against that cacheable buffer, then bulk VECTOR-copy the finished result back
into a VTCM output slot before DMA'ing it out.

VTCM is identity-mapped at 0xd8400000. uDMA: build a static/global Type-0
descriptor {next, ctrl=len, src, dst}, then Q6_dmstart_A(&desc) /
Q6_R_dmwait(). The descriptor MUST be static/global (a stack descriptor
no-ops at -O2).

Use HVX intrinsics where helpful; include <hexagon_types.h> and
<hexagon_protos.h>. Do NOT write main(). Respond with a single complete C
code block and CLOSE the fence with ```.
