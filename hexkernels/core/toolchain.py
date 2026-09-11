"""Hexagon SDK toolchain discovery + invocation.

Ported byte-identical (logic-frozen) from the old m1_driver/evaluate.py:
locating the Hexagon SDK toolchain bin dir, building a subprocess env with
it on PATH/LD_LIBRARY_PATH, running compiler/sim commands without raising
on nonzero exit or timeout, and the pinned timing-model constants + the
per-task capability -> compiler/sim flag helpers.
"""
import glob
import os
import subprocess

from hexkernels.core import target as _target
from typing import Optional, Union

_WIN_SDK_DEFAULT = r"C:\Hexagon_SDK\6.4.0.2"


def default_sdk_root() -> str:
    """HEXAGON_SDK_ROOT env var if set, else the Windows install default."""
    return os.environ.get("HEXAGON_SDK_ROOT", _WIN_SDK_DEFAULT)


def _exe(name: str) -> str:
    """Append .exe on Windows; bare name on Linux/Mac."""
    return name + (".exe" if os.name == "nt" else "")


def toolchain_env(bin_dir: str, base_env: Optional[dict] = None) -> dict:
    """Subprocess env with the toolchain bin on PATH (all OSes), plus, on
    non-Windows, its sibling lib dirs on LD_LIBRARY_PATH (the Linux ISS needs them)."""
    env = dict(base_env if base_env is not None else os.environ)
    env["PATH"] = bin_dir + os.pathsep + env.get("PATH", "")
    if os.name != "nt":
        tools = os.path.dirname(bin_dir)  # .../Tools
        libs = [d for d in (os.path.join(tools, "lib"), os.path.join(tools, "lib", "iss"))
                if os.path.isdir(d)]
        if libs:
            existing = env.get("LD_LIBRARY_PATH", "")
            parts = libs + ([existing] if existing else [])
            env["LD_LIBRARY_PATH"] = os.pathsep.join(parts)
    return env


DEFAULT_SDK_ROOT = default_sdk_root()

# --- Pinned timing target -------------------------------------------------
# The target now comes from hexkernels.core.target (HEXBENCH_TARGET selects it,
# default v75 -> rev_id v75na_1). Selecting the arch selects that chip's
# cache/pipeline model, so we don't hand-set cache sizes.
#
# RETARGETED 2026-08-03, v68n_1024 -> v75na_1. The two are cache-config-identical
# (6 HW threads, 32KB L1-I, 16KB L1-D, 1024KB L2, H3 coprocessor, HVX 128B);
# VTCM differs (4096KB @ 0xd8400000 -> 8192KB @ 0xd9000000) and is read from the
# config table, not hardcoded. v75 is a supported target for Qualcomm's
# hexagon-mlir (which tests v73/v75/v79, NOT v68), which is what makes vendor
# compiler output usable as a mechanism oracle and performance ceiling.
#
# CYCLE COUNTS ARE NOT COMPARABLE ACROSS THIS CHANGE -- different
# microarchitecture. Correctness and the anti-cheat verdicts are unaffected.
# Set HEXBENCH_TARGET=v68 to reproduce prior measurements.
#
# TIMING_MODE engages the cycle-approximate microarchitectural model, which
# accounts for caches + bus/DDR latency. With it OFF, the sim idealizes memory
# (much lower, compute-only Pcycles) -- misleading for bandwidth-bound kernels.
#
# BUS_PENALTY/BUS_RATIO are the SDK's representative defaults, pinned EXPLICITLY
# so the cycle reward stays reproducible across SDK upgrades. dsp_clock is NOT
# set: it does not affect Pcycles (only wall-time = Pcycles / clock).
DSP_ARCH = _target.current().arch
TIMING_MODE = True
BUS_PENALTY = 75
BUS_RATIO = 2

# C++ (switched 2026-08-03). Kernels are C++17 compiled with hexagon-clang++.
# `candidate_kernel` must be declared `extern "C"` -- without it the symbol
# mangles to `_Z16candidate_kernelPKaPai`, which breaks harness linkage and makes
# the anti-cheat's symbol scoping unreadable. Verified: HVX intrinsics,
# l2fetch, VTCM absolute-address scratch, templates and alignas all compile
# under -std=c++17 on both v68 and v75.
CXX_STD = "c++17"
COMPILER = "hexagon-clang++"

# Shared compile flags for both the harness+candidate link and the candidate-only
# (-c) object used for ELF-level HVX detection -- kept in one place so the two
# compiles can never drift (e.g. -O level) and disagree on what code was judged.
HVX_CFLAGS = [f"-m{DSP_ARCH}", "-mhvx", "-mhvx-length=128B",
              f"-std={CXX_STD}", "-O2"]

SIM_TIMEOUT_S = 60  # default/base sim timeout; also the (size-independent) objdump guard
SIM_TIMEOUT_MAX_S = 900  # hard ceiling for the size-aware sim timeout — an XL kernel gets
                         # more time, but this still kills genuine infinite-loop candidates.
                         # Raised 300 -> 900 (2026-08-01): the achievability sweep found four
                         # DMA/VTCM experts that are CORRECT but need 204-406s under timing
                         # (dma_2d_transpose_i8 405.6s, dma_nchw_nhwc_i8 400.2s,
                         # vtcm_softmax_row_i16 323.9s, dma_ping_pong_conv1d_i8 204.5s). At a
                         # 300s ceiling those tasks are unsolvable BY CONSTRUCTION — no model
                         # could ever pass them — which is a worse defect than a slow eval.
                         # Cost: a genuinely hung candidate now burns 900s instead of 300s.


# --- Per-task capability flags (v4) --------------------------------------
# A task's spec.json may declare an optional "caps" list, e.g. ["hmx"].
# Absent/empty => current HVX-only behavior (every v3 task is unchanged).
def cflags_for_caps(caps: list) -> list[str]:
    """Compile flags = base HVX flags + capability extensions."""
    flags = list(HVX_CFLAGS)
    if "hmx" in caps:
        flags.append("-mhmx")
    return flags


def sim_flags_for_caps(caps: list) -> list[str]:
    """Extra hexagon-sim flags for the declared capabilities."""
    extra = []
    if "hmx" in caps:
        extra += ["--mhmx", "2"]
    return extra


def find_toolchain_bin(sdk_root: str) -> str:
    """Locate the HEXAGON_Tools .../Tools/bin directory inside the SDK."""
    pattern = os.path.join(sdk_root, "tools", "HEXAGON_Tools", "*", "Tools", "bin")
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(
            f"No Hexagon toolchain bin found under {sdk_root} (looked for {pattern})"
        )
    return matches[-1]  # highest version


def run(cmd: list, env: dict, timeout: Optional[float] = None) -> tuple[Union[int, None], str, str, bool]:
    """Run a command, capture stdout+stderr, never raise on nonzero exit.

    DECODING (fixed 2026-08-02). Text mode previously used the locale codec --
    cp1252 on Windows -- with no error handler, so a single byte outside that
    codepage in compiler or simulator output raised UnicodeDecodeError *inside
    subprocess's reader thread*. The thread died, ``p.stderr`` came back None,
    and evaluate.py then crashed with ``AttributeError: 'NoneType' object has no
    attribute 'strip'`` -- reported as a compile failure for a kernel that
    compiled fine. Observed twice on real v6 experts.

    UTF-8 with ``errors="replace"`` instead: diagnostics are for humans and for
    substring matching, so a replacement character is always better than losing
    the process result. This also makes behaviour identical across platforms,
    which the locale default did not.
    """
    try:
        p = subprocess.run(
            cmd, env=env, timeout=timeout,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, encoding="utf-8", errors="replace",
        )
        return p.returncode, p.stdout, p.stderr, False
    except subprocess.TimeoutExpired as e:
        out = e.stdout or ""
        err = e.stderr or ""
        if isinstance(out, bytes):
            out = out.decode(errors="replace")
        if isinstance(err, bytes):
            err = err.decode(errors="replace")
        return None, out, err, True
