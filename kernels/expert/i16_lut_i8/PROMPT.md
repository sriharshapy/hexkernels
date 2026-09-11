Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int16_t *in, int8_t *out, int n,
                          const int8_t *lut, int lutsize);
Apply a LUT with int16 inputs. For each element:
  1. Clamp index: idx = clamp(in[i], 0, lutsize-1)
     (if in[i] < 0 → idx=0; if in[i] >= lutsize → idx=lutsize-1)
  2. Lookup: out[i] = lut[idx]
lut has lutsize entries (runtime param — do NOT hardcode lutsize or lut values).
The CLAMP is mandatory: inputs can be negative or exceed lutsize-1.
n=1000 (not a multiple of 128 — handle the tail).
Include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
