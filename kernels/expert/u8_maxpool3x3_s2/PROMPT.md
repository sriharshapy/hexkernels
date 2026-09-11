Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const uint8_t *in, uint8_t *out, int w, int h);

Compute a 3x3 overlapping MAX pool with stride 2, clamp-to-edge padding.
Output dimensions: ow = (w+1)/2, oh = (h+1)/2, stored row-major (stride ow).
Semantics:
    For each output pixel (ox, oy):
        cx = 2*ox, cy = 2*oy  (centre in input)
        out[oy*ow+ox] = max over dy in {-1,0,1}, dx in {-1,0,1} of:
            in[ clamp(cy+dy, 0, h-1) * w + clamp(cx+dx, 0, w-1) ]
Padding policy: CLAMP-TO-EDGE (replicate border pixels; NO zero-padding).
w=129, h=97 → ow=65, oh=49; ow=65 is NOT a multiple of 128 — handle the tail.
Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>.
Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
