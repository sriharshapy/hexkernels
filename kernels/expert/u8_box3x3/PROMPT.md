Implement ONLY this function (Hexagon HVX C):
 void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h);
Compute a 3x3 box blur: out[y*w+x] = (sum of the 9 neighbours) / 9 (integer division,
truncated). Use CLAMP-TO-EDGE for borders (out-of-range coords clamp to [0,w-1]/[0,h-1]).
Output is the same w*h size. w is not a multiple of 128 — handle the tail. Use HVX
Respond with a single complete C code block and CLOSE the fence with ```.
