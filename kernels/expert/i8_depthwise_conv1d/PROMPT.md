Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *x, const int8_t *taps, int32_t *out,
                          int L, int C, int ntaps);
Compute a depthwise 1D FIR as CORRELATION (taps NOT reversed):
    out[ch][i] = sum over j=0..ntaps-1 of x[ch*(L+ntaps-1)+i+j] * taps[ch*ntaps+j],
    for ch in [0,C) and i in [0,L).
Layout: x[ch * (L+ntaps-1) + i], taps[ch*ntaps + j], out[ch*L + i].
C=4 channels, L=250 output samples per channel, ntaps=8. L is not a multiple of 128 —
handle the tail. Each channel uses its own taps — DO NOT share taps across channels.
Use a 32-bit signed accumulator. Use HVX intrinsics; include <hexagon_types.h>
and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
