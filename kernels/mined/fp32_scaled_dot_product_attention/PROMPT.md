Accelerate the kernel `fp32_scaled_dot_product_attention` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static float v_mul[98304];
static float v_permute[98304];
static float v_mul_1[98304];
static float v_mm[65536];
static float v_amax[256];
static float v_sub[65536];
static float v_exp[65536];
static float v_sum_1[256];
static float v_div[65536];
static unsigned char v_eq[65536];
static unsigned char v_logical_not[65536];
static unsigned char v_any_1[256];
static unsigned char v_logical_not_1[256];
static float v_full_like[65536];
static float v_where[65536];
static float v_mm_1[98304];
static float v__to_copy_1[98304];

extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, const float *v_args_2, float *out0) {
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v_mul[i0*384 + i1*1] = (v_args_0[i0*384 + i1*1] * 0.22590050090246122f);
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_permute[i0*256 + i1*1] = v_args_1[i1*384 + i0];
    }
  }
  for (int i0 = 0; i0 < 384; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_mul_1[i0*256 + i1*1] = (v_permute[i0*256 + i1*1] * 0.22590050090246122f);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      float acc = 0;
      for (int k = 0; k < 384; k++) {
        acc = acc + v_mul[i0*384 + k] * v_mul_1[k*256 + i1];
      }
    v_mm[i0*256 + i1] = acc;
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    {
      float acc = -INFINITY;
      for (int r1 = 0; r1 < 256; r1++) {
        acc = (v_mm[i0*256 + r1*1] > acc ? v_mm[i0*256 + r1*1] : acc);
      }
      v_amax[i0*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_sub[i0*256 + i1*1] = (v_mm[i0*256 + i1*1] - v_amax[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_exp[i0*256 + i1*1] = expf(v_sub[i0*256 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    {
      float acc = 0;
      for (int r1 = 0; r1 < 256; r1++) {
        acc = (acc + v_exp[i0*256 + r1*1]);
      }
      v_sum_1[i0*1] = acc;
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_div[i0*256 + i1*1] = (v_exp[i0*256 + i1*1] / v_sum_1[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_eq[i0*256 + i1*1] = (v_mm[i0*256 + i1*1] == (-INFINITY));
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_logical_not[i0*256 + i1*1] = (!v_eq[i0*256 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    {
      int32_t acc = 0;
      for (int r1 = 0; r1 < 256; r1++) {
        acc = (acc || (v_logical_not[i0*256 + r1*1] != 0));
      }
      v_any_1[i0*1] = (unsigned char)(acc);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 1; i1++) {
      v_logical_not_1[i0*1] = (!v_any_1[i0*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_full_like[i0*256 + i1*1] = (float)(0);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 256; i1++) {
      v_where[i0*256 + i1*1] = (v_logical_not_1[i0*1] ? v_full_like[i0*256 + i1*1] : v_div[i0*256 + i1*1]);
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      float acc = 0;
      for (int k = 0; k < 256; k++) {
        acc = acc + v_where[i0*256 + k] * v_args_2[k*384 + i1];
      }
    v_mm_1[i0*384 + i1] = acc;
    }
  }
  for (int i0 = 0; i0 < 256; i0++) {
    for (int i1 = 0; i1 < 384; i1++) {
      v__to_copy_1[i0*384 + i1*1] = v_mm_1[i0*384 + i1*1];
    }
  }
  for (int i = 0; i < 98304; i++) { out0[i] = v__to_copy_1[i]; }
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten.mul.Scalar
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    args_0 (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    mul (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.permute.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 384 (parallel)
    d1: 256 (parallel)
  indexing_maps:
    args_1 (256, 384) operand: affine_map<(d0, d1) -> (d1, d0)>
    permute (384, 256) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: none
aten.mul.Scalar
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 384 (parallel)
    d1: 256 (parallel)
  indexing_maps:
    permute (384, 256) operand: affine_map<(d0, d1) -> (d0, d1)>
    mul_1 (384, 256) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.mm.default
  iterator_types = ['parallel', 'parallel', 'reduction']
  loops:
    d0: 256 (parallel)
    d1: 256 (parallel)
    d2: 384 (reduction)
  indexing_maps:
    mul (256, 384) operand: affine_map<(d0, d1, d2) -> (d0, d2)>
    mul_1 (384, 256) operand: affine_map<(d0, d1, d2) -> (d2, d1)>
    mm (256, 256) result: affine_map<(d0, d1, d2) -> (d0, d1)>
  vectorizable_loop: d1
aten.amax.default
  iterator_types = ['parallel', 'reduction']
  loops:
    d0: 256 (parallel)
    d1: 256 (reduction)
  indexing_maps:
    mm (256, 256) operand: affine_map<(d0, d1) -> (d0, d1)>
    amax (256, 1) result: affine_map<(d0, d1) -> (d0, 0)>
  vectorizable_loop: none
aten.sub.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 256 (parallel)
  indexing_maps:
    mm (256, 256) operand: affine_map<(d0, d1) -> (d0, d1)>
    amax (256, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    sub (256, 256) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.exp.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 256 (parallel)
  indexing_maps:
    sub (256, 256) operand: affine_map<(d0, d1) -> (d0, d1)>
    exp (256, 256) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.sum.dim_IntList
  iterator_types = ['parallel', 'reduction']
  loops:
    d0: 256 (parallel)
    d1: 256 (reduction)
  indexing_maps:
    exp (256, 256) operand: affine_map<(d0, d1) -> (d0, d1)>
    sum_1 (256, 1) result: affine_map<(d0, d1) -> (d0, 0)>
  vectorizable_loop: none
aten.div.Tensor
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 256 (parallel)
  indexing_maps:
    exp (256, 256) operand: affine_map<(d0, d1) -> (d0, d1)>
    sum_1 (256, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    div (256, 256) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.eq.Scalar
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 256 (parallel)
  indexing_maps:
    mm (256, 256) operand: affine_map<(d0, d1) -> (d0, d1)>
    eq (256, 256) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.logical_not.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 256 (parallel)
  indexing_maps:
    eq (256, 256) operand: affine_map<(d0, d1) -> (d0, d1)>
    logical_not (256, 256) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.any.dim
  iterator_types = ['parallel', 'reduction']
  loops:
    d0: 256 (parallel)
    d1: 256 (reduction)
  indexing_maps:
    logical_not (256, 256) operand: affine_map<(d0, d1) -> (d0, d1)>
    any_1 (256, 1) result: affine_map<(d0, d1) -> (d0, 0)>
  vectorizable_loop: none
aten.logical_not.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    any_1 (256, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    logical_not_1 (256, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.full_like.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 256 (parallel)
  indexing_maps:
    full_like (256, 256) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.where.self
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 256 (parallel)
  indexing_maps:
    logical_not_1 (256, 1) operand: affine_map<(d0, d1) -> (d0, 0)>
    full_like (256, 256) operand: affine_map<(d0, d1) -> (d0, d1)>
    div (256, 256) operand: affine_map<(d0, d1) -> (d0, d1)>
    where (256, 256) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.mm.default
  iterator_types = ['parallel', 'parallel', 'reduction']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
    d2: 256 (reduction)
  indexing_maps:
    where (256, 256) operand: affine_map<(d0, d1, d2) -> (d0, d2)>
    args_2 (256, 384) operand: affine_map<(d0, d1, d2) -> (d2, d1)>
    mm_1 (256, 384) result: affine_map<(d0, d1, d2) -> (d0, d1)>
  vectorizable_loop: d1
aten._to_copy.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 256 (parallel)
    d1: 384 (parallel)
  indexing_maps:
    mm_1 (256, 384) operand: affine_map<(d0, d1) -> (d0, d1)>
    _to_copy_1 (256, 384) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1

Working set: 1572864 bytes -> tier T2.
Mechanisms this size justifies:
  hvx: at least one loop is parallel with unit/zero stride on every operand, so it maps to a vector loop
  l2fetch: working set 1572864 B exceeds L1D 16384 B, so the stream misses L1 and prefetch has something to hide
  dma: working set 1572864 B exceeds L2 1048576 B, so the data does not stay resident and staging is what buys bandwidth
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
  - Entry point must be exactly:  extern "C" void candidate_kernel(const float *v_args_0, const float *v_args_1, const float *v_args_2, float *out0)
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
