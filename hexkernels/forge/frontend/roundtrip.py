"""Compile the emitted kernel plus its generated golden harness with the real
Hexagon toolchain, run it under `hexagon-sim`, and report the verdict.

This is where the emitter (`emit.py`) and the oracle (`oracle.py`) validate
each other: until a kernel passes its own generated harness on the real
simulator, nothing has proven the emitted C is *correct* -- only that it
looks right.

`emit_c` is imported INSIDE `roundtrip()`, not at module import time.
`hexkernels.forge.frontend.emit` may be built by another agent in parallel and
not exist yet; deferring the import means this module -- and every
non-`@pytest.mark.slow` test that only exercises `oracle.py` -- still works
with no toolchain and no emitter present.
"""
import os
import re
import shutil
import tempfile

from hexkernels.core import target as _target
from hexkernels.core.toolchain import (
    COMPILER,
    HVX_CFLAGS,
    SIM_TIMEOUT_S,
    _exe,
    default_sdk_root,
    find_toolchain_bin,
    run,
    toolchain_env,
)
from hexkernels.forge.frontend.oracle import golden, harness_c
from hexkernels.forge.frontend.trace import trace

# A hexagon-clang++ compile normally finishes in a few seconds; bound it so a
# pathological emitted kernel can't hang the test run indefinitely.
COMPILE_TIMEOUT_S = 180

# harness_common.h (for hvx_close_f32 / the float tolerance) lives here.
_COMMON_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "env", "harness",
)

_FIRST_STORE = re.compile(r"(\][^=;\n]*=\s*)([^;]+)(;)")


def _corrupt(kernel_src: str) -> str:
    """Perturb the first array store in `kernel_src` by ``+ 1.0f``.

    Used only by the rejection test: a harness that passes every candidate
    -- including a wrong one -- is worse than no harness at all, because it
    would certify every future candidate. This proves the harness can fail.
    """
    m = _FIRST_STORE.search(kernel_src)
    if not m:
        return kernel_src
    start, end = m.span()
    corrupted = f"{m.group(1)}({m.group(2)}) + 1.0f{m.group(3)}"
    return kernel_src[:start] + corrupted + kernel_src[end:]


def roundtrip(module, args, name, corrupt: bool = False) -> dict:
    """Trace `module`, emit its kernel, build the golden oracle + harness,
    compile both with `hexagon-clang++`, run under `hexagon-sim`, and report
    the result as ``{"compiled": bool, "correct": bool, "error_text": str}``.

    `corrupt=True` perturbs the emitted kernel's first store so the
    rejection test can prove the generated harness actually rejects a wrong
    answer rather than passing everything.
    """
    from hexkernels.forge.frontend.emit import emit_c  # deferred: see module docstring

    g = trace(module, args, name)
    kernel_src = emit_c(g)
    if corrupt:
        kernel_src = _corrupt(kernel_src)
    gold = golden(module, args)
    harness_src = harness_c(g, gold)

    sdk_root = default_sdk_root()
    bin_dir = find_toolchain_bin(sdk_root)
    env = toolchain_env(bin_dir)
    env["HEXAGON_SDK_ROOT"] = sdk_root

    clang = os.path.join(bin_dir, _exe(COMPILER))
    sim = os.path.join(bin_dir, _exe("hexagon-sim"))
    arch = _target.current().arch  # never a literal -- see CLAUDE.md retarget history

    workdir = tempfile.mkdtemp(prefix="forge_roundtrip_")
    try:
        kernel_path = os.path.join(workdir, "kernel.cpp")
        harness_path = os.path.join(workdir, "harness.cpp")
        elf_path = os.path.join(workdir, f"{name}.elf")

        with open(kernel_path, "w", encoding="utf-8") as f:
            f.write(kernel_src)
        with open(harness_path, "w", encoding="utf-8") as f:
            f.write(harness_src)

        compile_cmd = [clang, *HVX_CFLAGS, f"-I{_COMMON_DIR}",
                        "-o", elf_path, harness_path, kernel_path]
        rc, cout, cerr, ctimed = run(compile_cmd, env, timeout=COMPILE_TIMEOUT_S)
        if ctimed or rc != 0 or not os.path.exists(elf_path):
            error_text = (f"compile timed out after {COMPILE_TIMEOUT_S}s" if ctimed
                          else ((cerr or "").strip() or (cout or "").strip()
                                or "compilation failed (no diagnostics captured)"))
            return {"compiled": False, "correct": False, "error_text": error_text}

        sim_cmd = [sim, f"-m{arch}", elf_path]
        rc, sout, serr, timed_out = run(sim_cmd, env, timeout=SIM_TIMEOUT_S)
        combined = (sout or "") + "\n" + (serr or "")
        if timed_out:
            return {"compiled": True, "correct": False,
                    "error_text": f"simulator timed out after {SIM_TIMEOUT_S}s"}

        correct = "HVXENV_CORRECT" in combined
        return {"compiled": True, "correct": correct,
                "error_text": "" if correct else combined.strip()}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)
