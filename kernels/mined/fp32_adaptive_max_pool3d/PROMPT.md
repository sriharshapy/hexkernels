Accelerate the kernel `fp32_adaptive_max_pool3d` for the Hexagon NSP.

Target: Hexagon v75 (v75na_1), compiled with hexagon-clang++ -std=c++17.
  HVX vector width : 128 bytes
  L1 data cache    : 16384 bytes
  L2 cache         : 1048576 bytes
  VTCM             : 8388608 bytes at 0xd9000000 (software-managed scratchpad, not part of the automatic hierarchy)

The reference below is correct and deliberately slow. It is the specification: your kernel must compute exactly the same values.

```cpp
#include <stdint.h>
#include <math.h>

static float v_adaptive_max_pool3d_0[4096];
static int64_t v_adaptive_max_pool3d_1[4096];

extern "C" void candidate_kernel(const float *v_args_0, float *out0, int64_t *out1) {
  for (int i0 = 0; i0 < 1; i0++) {
    for (int i1 = 0; i1 < 8; i1++) {
      for (int i2 = 0; i2 < 8; i2++) {
        for (int i3 = 0; i3 < 8; i3++) {
          for (int i4 = 0; i4 < 8; i4++) {
            const int s0 = (i2 * 16) / 8;
            const int e0 = ((i2 + 1) * 16 + 8 - 1) / 8;
            const int s1 = (i3 * 16) / 8;
            const int e1 = ((i3 + 1) * 16 + 8 - 1) / 8;
            const int s2 = (i4 * 16) / 8;
            const int e2 = ((i4 + 1) * 16 + 8 - 1) / 8;
            float best = (float)-INFINITY;
            int64_t arg = 0;
            for (int k0 = s0; k0 < e0; k0++) {
              for (int k1 = s1; k1 < e1; k1++) {
                for (int k2 = s2; k2 < e2; k2++) {
                  if (v_args_0[(i0)*32768 + (i1)*4096 + (k0)*256 + (k1)*16 + (k2)] > best) { best = v_args_0[(i0)*32768 + (i1)*4096 + (k0)*256 + (k1)*16 + (k2)]; arg = (int64_t)((k0)*256 + (k1)*16 + (k2)); }
                }
              }
            }
            v_adaptive_max_pool3d_0[i1*512 + i2*64 + i3*8 + i4*1] = best;
            v_adaptive_max_pool3d_1[i1*512 + i2*64 + i3*8 + i4*1] = arg;
          }
        }
      }
    }
  }
  for (int i = 0; i < 4096; i++) { out0[i] = v_adaptive_max_pool3d_0[i]; }
  for (int i = 0; i < 4096; i++) { out1[i] = v_adaptive_max_pool3d_1[i]; }
}
```

The same computation as MLIR Linalg, emitted by torch-mlir from this graph. This is the compiler's own structured form: `iterator_types` marks each loop parallel or reduction, and the `#map` aliases are the affine indexing maps -- one result per operand axis, so an axis pinned to a constant (`(d0, 0)`) is a broadcast that does not advance. Named ops such as `linalg.matmul` carry their iterator types in the op definition rather than spelling them out.

```mlir
#map = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d5, d6, d7)>
#map1 = affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
func.func @main(%arg0: tensor<1x8x16x16x16xf32>) -> (tensor<1x8x8x8x8xf32>, tensor<1x8x8x8x8xi64>) {
  %3 = linalg.fill ins(%cst : f32) outs(%2 : tensor<1x8x8x8x8xf32>) -> tensor<1x8x8x8x8xf32>
  %4:2 = linalg.generic {indexing_maps = [#map, #map1, #map1], iterator_types = ["parallel", "parallel", "parallel", "parallel", "parallel", "reduction", "reduction", "reduction"]} ins(%0 : tensor<3x3x3xi1>) outs(%3, %1 : tensor<1x8x8x8x8xf32>, tensor<1x8x8x8x8xi64>) {
    ^bb0(%in: i1, %out: f32, %out_0: i64):
    %5 = linalg.index 1 : index
    %6 = linalg.index 2 : index
    %7 = linalg.index 3 : index
    %8 = linalg.index 4 : index
    %9 = linalg.index 5 : index
    %10 = linalg.index 6 : index
    %11 = linalg.index 7 : index
    %12 = arith.muli %6, %c16 : index
    %13 = arith.floordivsi %12, %c8 : index
    %14 = arith.addi %6, %c1 : index
    %15 = arith.muli %14, %c16 : index
    %16 = arith.subi %15, %c1 : index
    %17 = arith.floordivsi %16, %c8 : index
    %18 = arith.addi %17, %c1 : index
    %19 = arith.muli %7, %c16 : index
    %20 = arith.floordivsi %19, %c8 : index
    %21 = arith.addi %7, %c1 : index
    %22 = arith.muli %21, %c16 : index
    %23 = arith.subi %22, %c1 : index
    %24 = arith.floordivsi %23, %c8 : index
    %25 = arith.addi %24, %c1 : index
    %26 = arith.muli %8, %c16 : index
    %27 = arith.floordivsi %26, %c8 : index
    %28 = arith.addi %8, %c1 : index
    %29 = arith.muli %28, %c16 : index
    %30 = arith.subi %29, %c1 : index
    %31 = arith.floordivsi %30, %c8 : index
    %32 = arith.addi %31, %c1 : index
    %33 = arith.addi %13, %9 : index
    %34 = arith.addi %20, %10 : index
    %35 = arith.addi %27, %11 : index
    %extracted = tensor.extract %padded[%c0, %5, %33, %34, %35] : tensor<1x8x17x17x17xf32>
    %36 = arith.cmpi ult, %33, %18 : index
    %37 = arith.select %36, %extracted, %cst : f32
    %38 = arith.cmpi ult, %34, %25 : index
    %39 = arith.select %38, %37, %cst : f32
    %40 = arith.cmpi ult, %35, %32 : index
    %41 = arith.select %40, %39, %cst : f32
    %42 = arith.cmpf ogt, %41, %out : f32
    %43 = arith.select %42, %41, %out : f32
    %44 = arith.muli %33, %c16 : index
    %45 = arith.addi %44, %34 : index
    %46 = arith.muli %45, %c16 : index
    %47 = arith.addi %46, %35 : index
    %48 = arith.index_cast %47 : index to i64
    %49 = arith.select %42, %48, %out_0 : i64
    linalg.yield %43, %49 : f32, i64
    } -> (tensor<1x8x8x8x8xf32>, tensor<1x8x8x8x8xi64>)
}
```

Loop schedule of each primitive. `parallel` iterations are independent; `reduction` iterations accumulate and must not be reordered across the accumulator. The indexing maps give each operand's access pattern -- an operand whose map pins an axis to a constant does not advance along it (a broadcast, stride 0). `vectorizable_loop` is the axis along which every operand is unit- or zero-stride, so a contiguous vector load and store are legal:

aten.adaptive_max_pool3d.default
  iterator_types = ['parallel', 'parallel', 'parallel', 'parallel', 'parallel', 'reduction', 'reduction', 'reduction']
  loops:
    d0: 1 (parallel)
    d1: 8 (parallel)
    d2: 8 (parallel)
    d3: 8 (parallel)
    d4: 8 (parallel)
    d5: 3 (reduction)
    d6: 3 (reduction)
    d7: 3 (reduction)
  indexing_maps:
    args_0 (1, 8, 16, 16, 16) operand: affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d5, d6, d7)>
    adaptive_max_pool3d#0 (1, 8, 8, 8, 8) result: affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
    adaptive_max_pool3d#1 (1, 8, 8, 8, 8) result: affine_map<(d0, d1, d2, d3, d4, d5, d6, d7) -> (d0, d1, d2, d3, d4)>
  vectorizable_loop: d4

Working set: 163840 bytes -> tier T1.
Mechanisms this size justifies:
  hvx: at least one loop is parallel with unit/zero stride on every operand, so it maps to a vector loop
  l2fetch: working set 163840 B exceeds L1D 16384 B, so the stream misses L1 and prefetch has something to hide

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
  - Entry point must be exactly:  extern "C" void candidate_kernel(const float *v_args_0, float *out0, int64_t *out1)
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
