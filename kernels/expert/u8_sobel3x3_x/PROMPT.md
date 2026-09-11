Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const uint8_t *in, int16_t *out, int w, int h);

Compute the Sobel x-gradient filter:
  Kernel: [[-1, 0, 1],
           [-2, 0, 2],
           [-1, 0, 1]]
  out[y*w+x] = sum over (dy,dx) in [-1..1] x [-1..1] of:
               kernel[dy+1][dx+1] * in[clamp(y+dy,0,h-1)*w + clamp(x+dx,0,w-1)]
  Result is signed 16-bit (range [-1020, 1020], fits exactly in int16_t).
Border policy: CLAMP-TO-EDGE — out-of-range coords clamp to [0,h-1]/[0,w-1].
Output buffer: int16_t out[w*h], same w*h as input.
w is not a multiple of 128 — handle the tail.
Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
