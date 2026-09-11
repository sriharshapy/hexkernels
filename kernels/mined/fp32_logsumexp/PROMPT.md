Accelerate the kernel `fp32_logsumexp` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static float v_amax[32];
static float v_abs_1[32];
static unsigned char v_eq[32];
static float v_scalar_tensor[1];
static float v_where[32];
static float v_squeeze[32];
static float v_sub[32];
static float v_exp[32];
static float v_sum_1[32];
static float v_log[32];
static float v_add[32];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i1 = 0; i1 < 4; i1++) {
    for (int i2 = 0; i2 < 8; i2++) {
      {
        float acc = -INFINITY;
        for (int r0 = 0; r0 < 1; r0++) {
          acc = (v_args_0[i1*8 + i2*1] > acc ? v_args_0[i1*8 + i2*1] : acc);
        }
        v_amax[i1*8 + i2*1] = acc;
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        v_abs_1[i1*8 + i2*1] = (v_amax[i1*8 + i2*1] < 0 ? -v_amax[i1*8 + i2*1] : v_amax[i1*8 + i2*1]);
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        v_eq[i1*8 + i2*1] = (v_abs_1[i1*8 + i2*1] == INFINITY);
      }
    }
  }
  v_scalar_tensor[0] = (float)(0.0f);
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        v_where[i1*8 + i2*1] = (v_eq[i1*8 + i2*1] ? v_scalar_tensor[0] : v_amax[i1*8 + i2*1]);
      }
    }
  }
  for (int i = 0; i < 32; i++) v_squeeze[i] = v_where[i];
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        v_sub[i1*8 + i2*1] = (v_args_0[i1*8 + i2*1] - v_where[i1*8 + i2*1]);
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 4; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        v_exp[i1*8 + i2*1] = expf(v_sub[i1*8 + i2*1]);
      }
    }
  }
  for (int i1 = 0; i1 < 4; i1++) {
    for (int i2 = 0; i2 < 8; i2++) {
      {
        float acc = 0;
        for (int r0 = 0; r0 < 1; r0++) {
          acc = (acc + v_exp[i1*8 + i2*1]);
        }
        v_sum_1[i1*8 + i2*1] = acc;
      }
    }
  }
  for (int i0 = 0; i0 < 4; i0++) {
    for (int i1 = 0; i1 < 8; i1++) {
      v_log[i0*8 + i1*1] = logf(v_sum_1[i0*8 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 4; i0++) {
    for (int i1 = 0; i1 < 8; i1++) {
      v_add[i0*8 + i1*1] = (v_log[i0*8 + i1*1] + v_squeeze[i0*8 + i1*1]);
    }
  }
  for (int i = 0; i < 32; i++) { out0[i] = v_add[i]; }
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten.amax.default
  iterator_types = ['reduction', 'parallel', 'parallel']
  loops:
    d0: 1 (reduction)
    d1: 4 (parallel)
    d2: 8 (parallel)
  indexing_maps:
    args_0 (1, 4, 8) operand: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    amax (1, 4, 8) result: affine_map<(d0, d1, d2) -> (0, d1, d2)>
  vectorizable_loop: d2
aten.abs.default
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 4 (parallel)
    d2: 8 (parallel)
  indexing_maps:
    amax (1, 4, 8) operand: affine_map<(d0, d1, d2) -> (0, d1, d2)>
    abs_1 (1, 4, 8) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.eq.Scalar
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 4 (parallel)
    d2: 8 (parallel)
  indexing_maps:
    abs_1 (1, 4, 8) operand: affine_map<(d0, d1, d2) -> (0, d1, d2)>
    eq (1, 4, 8) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.scalar_tensor.default
  iterator_types = []
  loops:
  indexing_maps:
    scalar_tensor () result: affine_map<() -> ()>
  vectorizable_loop: none
aten.where.self
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 4 (parallel)
    d2: 8 (parallel)
  indexing_maps:
    eq (1, 4, 8) operand: affine_map<(d0, d1, d2) -> (0, d1, d2)>
    scalar_tensor () operand: affine_map<(d0, d1, d2) -> ()>
    amax (1, 4, 8) operand: affine_map<(d0, d1, d2) -> (0, d1, d2)>
    where (1, 4, 8) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.squeeze.dims
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 4 (parallel)
    d1: 8 (parallel)
  indexing_maps:
    where (1, 4, 8) operand: affine_map<(d0, d1) -> (0, d0, d1)>
    squeeze (4, 8) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 4 (parallel)
    d2: 8 (parallel)
  indexing_maps:
    args_0 (1, 4, 8) operand: affine_map<(d0, d1, d2) -> (0, d1, d2)>
    where (1, 4, 8) operand: affine_map<(d0, d1, d2) -> (0, d1, d2)>
    sub (1, 4, 8) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.exp.default
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 4 (parallel)
    d2: 8 (parallel)
  indexing_maps:
    sub (1, 4, 8) operand: affine_map<(d0, d1, d2) -> (0, d1, d2)>
    exp (1, 4, 8) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.sum.dim_IntList
  iterator_types = ['reduction', 'parallel', 'parallel']
  loops:
    d0: 1 (reduction)
    d1: 4 (parallel)
    d2: 8 (parallel)
  indexing_maps:
    exp (1, 4, 8) operand: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
    sum_1 (4, 8) result: affine_map<(d0, d1, d2) -> (d1, d2)>
  vectorizable_loop: d2
aten.log.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 4 (parallel)
    d1: 8 (parallel)
  indexing_maps:
    sum_1 (4, 8) operand: affine_map<(d0, d1) -> (d0, d1)>
    log (4, 8) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.add.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 4 (parallel)
    d1: 8 (parallel)
  indexing_maps:
    log (4, 8) operand: affine_map<(d0, d1) -> (d0, d1)>
    squeeze (4, 8) operand: affine_map<(d0, d1) -> (d0, d1)>
    add (4, 8) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1

Working set: 256 bytes -> tier T0.
Mechanisms this size justifies:
  hvx: at least one loop is parallel with unit/zero stride on every operand, so it maps to a vector loop

Provenance (this governs whether the result is usable at all):
  - Derive this kernel from the reference, the schedule, and the vendor's
    intrinsic headers ONLY.
  - NOTHING from this repository may appear in the translation unit -- no
    `harness_common.h`, no `hmx_helpers.h`, nothing else under
    hexbench/env/harness/. Those are R&D material. The directory is not on the
    compile's include path, so such an include is a build failure, and a COPIED
    body (the VTCM base, the alignment attribute, a crouton offset, a tolerance
    compare) is rejected before compiling. Hardware facts are yours to derive
    under your own names, with a comment saying where you got them.
  - Do not copy from, or pattern it on, any existing solution to this or a
    similar kernel -- including any private reference corpus and including other
    kernels generated alongside this one.
  - This pipeline's claim is that its corpus comes from public sources: an op
    list from the operator registry, sizes from the memory hierarchy, mechanisms
    computed from those sizes. A kernel copied from an existing solution breaks
    that claim no matter how well it performs, and cannot be used.

Rules:
  - Entry point must be exactly:  extern "C" void candidate_kernel(const float *v_args_0, float *out0)
    The "extern \"C\"" is required; without it the symbol mangles and the
    harness fails to link.
  - It is compiled as C++17 and run on the simulator against golden vectors.
    Producing the wrong numbers fails, however fast it is.
  - Do not change the signature, the parameter order, or the memory layout.
    Parameter order follows the traced graph's placeholders.
  - Include what you use. The vendor headers are available and are the only ones
    that resolve: <hexagon_types.h> (which also carries the user-DMA descriptor
    types), <hexagon_protos.h>, <hvx_hexagon_protos.h>, <hmx_hexagon_protos.h>,
    plus the C++ standard library. No repo-local header is on the include path.
  - The harness enables the HMX context before calling you, so HMX code needs no
    SSR setup (doing it anyway is harmless -- it is idempotent).
  - If you use the DMA engine, declare every descriptor
    `static hexagon_udma_descriptor_typeN_t d[...] __attribute__((aligned(64)));`.
    A descriptor is read by an EXTERNAL engine and does not tolerate the alignment
    a stack frame happens to give a 16- or 32-byte struct. Measured: the same
    kernel with function-local descriptors failed 2,979 of 3,211,264 elements and
    with static aligned ones passed 0 of 3,211,264, pipeline untouched. A second
    kernel crashed 0x28 the same way.
  - Also: a second `dmstart` may only be outstanding after the program's FIRST
    transfer has completed. Prime the engine with one `dmstart`/`dmwait` in a
    prologue and then overlap freely, or chain descriptors through `next` and issue
    ONE `dmstart` per tile. Two `dmstart`s as the engine's first act fault 0x28.
  - Return only the C++ translation unit, no prose and no markdown fence.
