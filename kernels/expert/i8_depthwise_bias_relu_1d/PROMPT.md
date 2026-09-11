Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *x, const int8_t *taps, const int32_t *bias,
                          int8_t *out, int L, int C, int ntaps);
Fused depthwise 1D conv (FIR correlation) + bias + ReLU, int8->int8.
C=8 independent channels, L=256 output samples/channel, ntaps=7 (also passed as runtime args).

Layout:
  x    : x[ch*(L+ntaps-1) + i],  ch in [0,C), i in [0, L+ntaps-1)
  taps : taps[ch*ntaps + j],      ch in [0,C), j in [0, ntaps)
  bias : bias[ch],                 ch in [0,C)  -- int32, per channel
  out  : out[ch*L + i],            ch in [0,C), i in [0, L)

For each output sample out[ch][i]:
  Step 1 -- FIR correlation (taps NOT reversed):
    acc = sum_{j=0}^{ntaps-1} x[ch*(L+ntaps-1)+i+j] * taps[ch*ntaps+j]   (int32)

  Step 2 -- bias:
    biased = acc + bias[ch]   (int32)

  Step 3 -- ReLU:
    after_relu = max(biased, 0)   (int32; clamp negatives to zero)

  Step 4 -- saturate to int8:
    out[ch*L + i] = clamp(after_relu, -128, 127)
    (after relu, output is always in [0, 127])

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
