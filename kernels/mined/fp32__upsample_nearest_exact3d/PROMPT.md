Accelerate the kernel `fp32__upsample_nearest_exact3d` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static float v__to_copy[524288];
static float v_arange[16];
static float v_add[16];
static float v_mul[16];
static int64_t v__to_copy_1[16];
static int64_t v_unsqueeze[16];
static int64_t v_unsqueeze_1[16];
static float v_arange_1[16];
static float v_add_1[16];
static float v_mul_1[16];
static int64_t v__to_copy_2[16];
static int64_t v_unsqueeze_2[16];
static float v_arange_2[16];
static float v_add_2[16];
static float v_mul_2[16];
static int64_t v__to_copy_3[16];
static float v_index[65536];
static float v__to_copy_4[65536];

extern "C" void candidate_kernel(const float *v_args_0, float *out0) {
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 32; i2++) {
        for (int i3 = 0; i3 < 32; i3++) {
          for (int i4 = 0; i4 < 32; i4++) {
            v__to_copy[i1*32768 + i2*1024 + i3*32 + i4*1] = v_args_0[i1*32768 + i2*1024 + i3*32 + i4*1];
          }
        }
      }
    }
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v_arange[i0*1] = (float)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v_add[i0*1] = (v_arange[i0*1] + 0.5f);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v_mul[i0*1] = (v_add[i0*1] * 2.0f);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v__to_copy_1[i0*1] = v_mul[i0*1];
  }
  for (int i = 0; i < 16; i++) v_unsqueeze[i] = v__to_copy_1[i];
  for (int i = 0; i < 16; i++) v_unsqueeze_1[i] = v_unsqueeze[i];
  for (int i0 = 0; i0 < 16; i0++) {
    v_arange_1[i0*1] = (float)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v_add_1[i0*1] = (v_arange_1[i0*1] + 0.5f);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v_mul_1[i0*1] = (v_add_1[i0*1] * 2.0f);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v__to_copy_2[i0*1] = v_mul_1[i0*1];
  }
  for (int i = 0; i < 16; i++) v_unsqueeze_2[i] = v__to_copy_2[i];
  for (int i0 = 0; i0 < 16; i0++) {
    v_arange_2[i0*1] = (float)(0 + i0 * 1);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v_add_2[i0*1] = (v_arange_2[i0*1] + 0.5f);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v_mul_2[i0*1] = (v_add_2[i0*1] * 2.0f);
  }
  for (int i0 = 0; i0 < 16; i0++) {
    v__to_copy_3[i0*1] = v_mul_2[i0*1];
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 16; i2++) {
        for (int i3 = 0; i3 < 16; i3++) {
          for (int i4 = 0; i4 < 16; i4++) {
            v_index[i1*4096 + i2*256 + i3*16 + i4*1] = v__to_copy[(i0)*524288 + (i1)*32768 + (v_unsqueeze_1[(i2)])*1024 + (v_unsqueeze_2[(i3)])*32 + (v__to_copy_3[(i4)])];
          }
        }
      }
    }
  }
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 16; i1++) {
      for (int i2 = 0; i2 < 16; i2++) {
        for (int i3 = 0; i3 < 16; i3++) {
          for (int i4 = 0; i4 < 16; i4++) {
            v__to_copy_4[i1*4096 + i2*256 + i3*16 + i4*1] = v_index[i1*4096 + i2*256 + i3*16 + i4*1];
          }
        }
      }
    }
  }
  for (int i = 0; i < 65536; i++) { out0[i] = v__to_copy_4[i]; }
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten._to_copy.default
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 32 (parallel)
    d3: 32 (parallel)
    d4: 32 (parallel)
  indexing_maps:
    args_0 (1, 16, 32, 32, 32) operand: affine_map<(d0, d1, d2, d3, d4) -> (0, d1, d2, d3, d4)>
    _to_copy (1, 16, 32, 32, 32) result: affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
  vectorizable_loop: d4
aten.arange.start_step
  iterator_types = ['parallel']
  loops:
    d0: 16 (parallel)
  indexing_maps:
    arange (16,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.add.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 16 (parallel)
  indexing_maps:
    arange (16,) operand: affine_map<(d0) -> (d0)>
    add (16,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.mul.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 16 (parallel)
  indexing_maps:
    add (16,) operand: affine_map<(d0) -> (d0)>
    mul (16,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten._to_copy.default
  iterator_types = ['parallel']
  loops:
    d0: 16 (parallel)
  indexing_maps:
    mul (16,) operand: affine_map<(d0) -> (d0)>
    _to_copy_1 (16,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.unsqueeze.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 16 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    _to_copy_1 (16,) operand: affine_map<(d0, d1) -> (d1)>
    unsqueeze (16, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.unsqueeze.default
  iterator_types = ['parallel', 'parallel', 'parallel']
  loops:
    d0: 16 (parallel)
    d1: 1 (parallel)
    d2: 1 (parallel)
  indexing_maps:
    unsqueeze (16, 1) operand: affine_map<(d0, d1, d2) -> (d1, 0)>
    unsqueeze_1 (16, 1, 1) result: affine_map<(d0, d1, d2) -> (d0, d1, d2)>
  vectorizable_loop: d2
aten.arange.start_step
  iterator_types = ['parallel']
  loops:
    d0: 16 (parallel)
  indexing_maps:
    arange_1 (16,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.add.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 16 (parallel)
  indexing_maps:
    arange_1 (16,) operand: affine_map<(d0) -> (d0)>
    add_1 (16,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.mul.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 16 (parallel)
  indexing_maps:
    add_1 (16,) operand: affine_map<(d0) -> (d0)>
    mul_1 (16,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten._to_copy.default
  iterator_types = ['parallel']
  loops:
    d0: 16 (parallel)
  indexing_maps:
    mul_1 (16,) operand: affine_map<(d0) -> (d0)>
    _to_copy_2 (16,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.unsqueeze.default
  iterator_types = ['parallel', 'parallel']
  loops:
    d0: 16 (parallel)
    d1: 1 (parallel)
  indexing_maps:
    _to_copy_2 (16,) operand: affine_map<(d0, d1) -> (d1)>
    unsqueeze_2 (16, 1) result: affine_map<(d0, d1) -> (d0, d1)>
  vectorizable_loop: d1
aten.arange.start_step
  iterator_types = ['parallel']
  loops:
    d0: 16 (parallel)
  indexing_maps:
    arange_2 (16,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.add.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 16 (parallel)
  indexing_maps:
    arange_2 (16,) operand: affine_map<(d0) -> (d0)>
    add_2 (16,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.mul.Tensor
  iterator_types = ['parallel']
  loops:
    d0: 16 (parallel)
  indexing_maps:
    add_2 (16,) operand: affine_map<(d0) -> (d0)>
    mul_2 (16,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten._to_copy.default
  iterator_types = ['parallel']
  loops:
    d0: 16 (parallel)
  indexing_maps:
    mul_2 (16,) operand: affine_map<(d0) -> (d0)>
    _to_copy_3 (16,) result: affine_map<(d0) -> (d0)>
  vectorizable_loop: d0
aten.index.Tensor
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 16 (parallel)
    d3: 16 (parallel)
    d4: 16 (parallel)
  indexing_maps:
    _to_copy (1, 16, 32, 32, 32) operand: affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, 0, 0, 0)>
    unsqueeze_1 (16, 1, 1) operand: affine_map<(d0, d1, d2, d3, d4) -> (d2, 0, 0)>
    unsqueeze_2 (16, 1) operand: affine_map<(d0, d1, d2, d3, d4) -> (d3, 0)>
    _to_copy_3 (16,) operand: affine_map<(d0, d1, d2, d3, d4) -> (d4)>
    index (1, 16, 16, 16, 16) result: affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
  vectorizable_loop: d4
aten._to_copy.default
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel', 'parallel']
  loops:
    d0: 1 (parallel)
    d1: 16 (parallel)
    d2: 16 (parallel)
    d3: 16 (parallel)
    d4: 16 (parallel)
  indexing_maps:
    index (1, 16, 16, 16, 16) operand: affine_map<(d0, d1, d2, d3, d4) -> (0, d1, d2, d3, d4)>
    _to_copy_4 (1, 16, 16, 16, 16) result: affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2, d3, d4)>
  vectorizable_loop: d4

Working set: 2359296 bytes -> tier T2.
Mechanisms this size justifies:
  hvx: at least one loop is parallel with unit/zero stride on every operand, so it maps to a vector loop
  l2fetch: working set 2359296 B exceeds L1D 16384 B, so the stream misses L1 and prefetch has something to hide
  dma: working set 2359296 B exceeds L2 1048576 B, so the data does not stay resident and staging is what buys bandwidth
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
