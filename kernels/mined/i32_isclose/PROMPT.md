Accelerate the kernel `i32_isclose` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static unsigned char v_eq[98304];
static float v__to_copy[98304];
static float v_mul[98304];
static float v_abs_1[98304];
static float v_add[98304];
static float v_sub[98304];
static float v_abs_2[98304];
static float v_abs_3[98304];
static unsigned char v_ne[98304];
static unsigned char v_eq_1[98304];
static unsigned char v_mul_1[98304];
static unsigned char v_le[98304];
static unsigned char v_bitwise_and[98304];
static unsigned char v_bitwise_or[98304];

extern "C" void candidate_kernel(const int32_t *v_args_0, const int32_t *v_args_1, unsigned char *out0) {
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_eq[i0*384 + i1*1] = (v_args_0[i0*384 + i1*1] == v_args_1[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v__to_copy[i0*384 + i1*1] = v_args_1[i0*384 + i1*1];
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_mul[i0*384 + i1*1] = (v__to_copy[i0*384 + i1*1] * 1e-05f);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_abs_1[i0*384 + i1*1] = (v_mul[i0*384 + i1*1] < 0 ? -v_mul[i0*384 + i1*1] : v_mul[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_add[i0*384 + i1*1] = (v_abs_1[i0*384 + i1*1] + 1e-08f);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_sub[i0*384 + i1*1] = (v_args_0[i0*384 + i1*1] - v__to_copy[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_abs_2[i0*384 + i1*1] = (v_sub[i0*384 + i1*1] < 0 ? -v_sub[i0*384 + i1*1] : v_sub[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_abs_3[i0*384 + i1*1] = (v_abs_2[i0*384 + i1*1] < 0 ? -v_abs_2[i0*384 + i1*1] : v_abs_2[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_ne[i0*384 + i1*1] = (v_abs_3[i0*384 + i1*1] != INFINITY);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_eq_1[i0*384 + i1*1] = (v_abs_2[i0*384 + i1*1] == v_abs_2[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_mul_1[i0*384 + i1*1] = (v_eq_1[i0*384 + i1*1] * v_ne[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_le[i0*384 + i1*1] = (v_abs_2[i0*384 + i1*1] <= v_add[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_bitwise_and[i0*384 + i1*1] = (v_mul_1[i0*384 + i1*1] & v_le[i0*384 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_bitwise_or[i0*384 + i1*1] = (v_eq[i0*384 + i1*1] | v_bitwise_and[i0*384 + i1*1]);
    }
  }
  for (int i = 0; i < 98304; i++) { out0[i] = v_bitwise_or[i]; }
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten.eq.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    args_0 (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    args_1 (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    eq (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten._to_copy.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    args_1 (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    _to_copy (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.mul.Scalar
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    _to_copy (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    mul (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.abs.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    mul (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    abs_1 (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.add.Scalar
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    abs_1 (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    add (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    args_0 (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    _to_copy (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    sub (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.abs.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    sub (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    abs_2 (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.abs.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    abs_2 (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    abs_3 (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.ne.Scalar
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    abs_3 (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    ne (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.eq.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    abs_2 (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    abs_2 (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    eq_1 (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.mul.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    eq_1 (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    ne (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    mul_1 (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.le.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    abs_2 (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    add (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    le (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.bitwise_and.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    mul_1 (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    le (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    bitwise_and (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.bitwise_or.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    eq (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    bitwise_and (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    bitwise_or (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1

Working set: 1179648 bytes -> tier T2.
Mechanisms this size justifies:
  hvx: at least one loop is parallel with unit/zero stride on every operand, so it maps to a vector loop
  l2fetch: working set 1179648 B exceeds L1D 16384 B, so the stream misses L1 and prefetch has something to hide
  dma: working set 1179648 B exceeds L2 1048576 B, so the data does not stay resident and staging is what buys bandwidth
  vtcm: the staged tiles need a software-managed scratchpad

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
  - Entry point must be exactly:  extern "C" void candidate_kernel(const int32_t *v_args_0, const int32_t *v_args_1, unsigned char *out0)
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
