Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h);

Compute a 2x2 L2 POOL with stride 2. Output dimensions: (w/2) x (h/2), row-major.
Semantics:
 For each output pixel (ox, oy):
 a = in[2oy*w+2ox], b = in[2oy*w+2ox+1]
 c = in[(2oy+1)*w+2ox], d = in[(2oy+1)*w+2ox+1]
 sum_sq = a*a + b*b + c*c + d*d (int, max = 4*255^2 = 260100, fits int32)
 val = (uint32_t) sqrtf((float)sum_sq) ← floor via truncating cast
 out[oy*(w/2)+ox] = (uint8_t)(val > 255 ? 255 : val) ← saturate at 255
w and h are even; w/2 = 65 is NOT a multiple of 128 — handle the tail.
Use sqrtf from <math.h> in the scalar fallback / tail; for HVX bodies you may
use a fixed-point sqrt approximation that matches the above formula exactly.
Do NOT write main.
Respond with a single complete C code block and CLOSE the fence with ```.
