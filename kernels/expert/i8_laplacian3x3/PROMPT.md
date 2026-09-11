Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const uint8_t *in, int16_t *out, int w, int h);

Compute the 3x3 Laplacian edge filter:
 Kernel: [[ 0, 1, 0],
 [ 1, -4, 1],
 [ 0, 1, 0]]
 out[y*w+x] = in[clamp(y-1,0,h-1)*w + x]
 + in[clamp(y+1,0,h-1)*w + x]
 + in[y*w + clamp(x-1,0,w-1)]
 + in[y*w + clamp(x+1,0,w-1)]
 - 4 * in[y*w + x]
 Result is signed 16-bit (range [-1020, 1020], fits in int16_t). No saturation.
Border policy: CLAMP-TO-EDGE — out-of-range coords clamp to [0,h-1]/[0,w-1].
Output buffer: int16_t out[w*h], same w*h as input.
w is not a multiple of 128 — handle the tail.
Do NOT write main.
Respond with a single complete C code block and CLOSE the fence with ```.
