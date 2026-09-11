Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h);

Compute the Sobel gradient magnitude (L1 approximation), clamped to uint8:
  Gx kernel: [[-1, 0, 1],[-2, 0, 2],[-1, 0, 1]]
  Gy kernel: [[-1,-2,-1],[ 0, 0, 0],[ 1, 2, 1]]
  Compute Gx and Gy using those kernels applied to the 3x3 neighbourhood.
  out[y*w+x] = min(|Gx| + |Gy|, 255)   (clamped L1 magnitude, uint8)
Border policy: CLAMP-TO-EDGE — out-of-range coords clamp to [0,h-1]/[0,w-1].
Output: uint8_t out[w*h], same size as input. w is not a multiple of 128 — handle the tail.
Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
