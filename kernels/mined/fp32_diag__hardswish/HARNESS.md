# `harness.c` is not checked in for this kernel

Its golden vectors are base64-embedded, and at this task's working set that
makes the file 12.0 MB -- over the 1 MB cap this corpus applies.

The harness is a **build cache, not a result**: it regenerates deterministically
from the task definition in `benchmark/selection.json`, a whole batch at a time,
in about 14 seconds.

```bash
python -m hexkernels.forge.run_batch --batch <N> --out witness_build/batch<N> --skip-verify
cp witness_build/batch<N>/fp32_diag__hardswish/harness.cpp kernels/mined/fp32_diag__hardswish/harness.c
```

Everything that is *not* regenerable -- the accelerated kernel, the scalar
reference, the prompt and the metadata -- is checked in beside this file.
