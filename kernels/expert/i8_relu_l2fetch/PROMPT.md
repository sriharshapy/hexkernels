Implement ONLY this function (Hexagon HVX C):
    void candidate_kernel(const int8_t *x, int8_t *out, int n);
Compute out[i] = (x[i] > 0) ? x[i] : 0 for i in [0, n) (signed int8 ReLU).

n is large (working set exceeds L2), so the task is DDR-bandwidth-bound. To go fast, hide
DDR latency: unroll the vector loop for more in-flight loads and issue a rolling L2 prefetch
a fixed distance ahead of the load (l2fetch). l2fetch has no C intrinsic on this toolchain;
use inline asm with a register-pair descriptor ([15:0]=stride,[31:16]=width,[47:32]=height).
n is a multiple of 128; handle any tail.

Use HVX intrinsics; include <hexagon_types.h> and <hexagon_protos.h>. Do NOT write main().
Respond with a single complete C code block and CLOSE the fence with ```.
