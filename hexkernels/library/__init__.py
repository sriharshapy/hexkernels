"""Load and query the kernel corpus in `kernels/`.

    from hexkernels.library import load, find

    load()                              # every kernel
    find(origin="expert", hmx=True)     # HMX kernels with measured cycles
    find(elf_confirmed=True)            # the detector saw the mechanism in the ELF
    find(tier="T2", buildable=True)     # complete bundles at one working-set tier

Every entry is a `Kernel`, and every `Kernel` has the same shape regardless of
which corpus it came from -- that uniformity is the point of `kernels/`, so
nothing here should ever need to branch on `origin`.
"""

import functools
import json
import os

__all__ = ["Kernel", "load", "find", "root", "summary"]


def root():
    """Absolute path of `kernels/`, resolved relative to the installed package."""
    env = os.environ.get("HEXKERNELS_ROOT")
    if env:
        return os.path.abspath(env)
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(os.path.dirname(os.path.dirname(here)), "kernels")


class Kernel:
    """One corpus entry: its metadata, and lazy access to its sources."""

    def __init__(self, spec, path):
        self.spec = spec
        self.path = path

    def __repr__(self):
        return f"<Kernel {self.spec['origin']}/{self.spec['name']} {self.spec['bundle']}>"

    # -- identity -----------------------------------------------------------
    @property
    def name(self):
        return self.spec["name"]

    @property
    def origin(self):
        """`expert`, `mined` or `model`. See LIBRARY.md for what each attests."""
        return self.spec["origin"]

    @property
    def dtype(self):
        return self.spec.get("dtype")

    @property
    def tier(self):
        return self.spec.get("tier")

    # -- mechanisms ---------------------------------------------------------
    @property
    def uses_hmx(self):
        return bool(self.spec["mechanisms_in_source"].get("hmx"))

    @property
    def uses_hvx(self):
        return bool(self.spec["mechanisms_in_source"].get("hvx"))

    @property
    def elf_confirmed(self):
        """Tri-state. True/False are detector verdicts; None means never scanned.

        Do not coerce this to a bool: "the detector found nothing" and "no
        detector was run" are different claims, and collapsing them is exactly
        the overstatement `hexkernels.anticheat` exists to prevent.
        """
        return self.spec.get("verified", {}).get("elf_confirmed")

    # -- buildability -------------------------------------------------------
    @property
    def buildable(self):
        """True when `reference.c` and `harness.c` are both present."""
        return self.spec["bundle"] == "complete"

    # -- sources ------------------------------------------------------------
    def _read(self, filename):
        path = os.path.join(self.path, filename)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()

    @property
    def source(self):
        """The accelerated kernel. Always present."""
        return self._read("kernel.c")

    @property
    def reference(self):
        """Scalar ground truth, or None when the bundle is kernel-only."""
        return self._read("reference.c")

    @property
    def harness(self):
        return self._read("harness.c")

    @property
    def prompt(self):
        return self._read("PROMPT.md")

    @property
    def speedup(self):
        """Measured expert speedup over the scalar baseline, where one exists."""
        return self.spec.get("verified", {}).get("expert_speedup")


@functools.lru_cache(maxsize=1)
def _index():
    path = os.path.join(root(), "index.json")
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def summary():
    """The counts printed by `tools/assemble_library.py` at build time."""
    return _index()["summary"]


@functools.lru_cache(maxsize=1)
def load():
    """Every kernel in the corpus, as a tuple of `Kernel`."""
    base = root()
    return tuple(
        Kernel(spec, os.path.join(base, spec["origin"], spec["name"]))
        for spec in _index()["kernels"]
    )


def find(origin=None, dtype=None, tier=None, hvx=None, hmx=None,
         elf_confirmed=None, buildable=None, name=None):
    """Filter the corpus. Every argument left as None is simply not applied.

    `elf_confirmed` matches the tri-state exactly, so `find(elf_confirmed=None)`
    does NOT mean "unscanned" -- it means "do not filter on this". Pass the
    string "unscanned" to select the never-scanned entries.
    """
    out = []
    for k in load():
        if origin is not None and k.origin != origin:
            continue
        if dtype is not None and k.dtype != dtype:
            continue
        if tier is not None and k.tier != tier:
            continue
        if hvx is not None and k.uses_hvx != hvx:
            continue
        if hmx is not None and k.uses_hmx != hmx:
            continue
        if buildable is not None and k.buildable != buildable:
            continue
        if name is not None and name not in k.name:
            continue
        if elf_confirmed is not None:
            want = None if elf_confirmed == "unscanned" else elf_confirmed
            if k.elf_confirmed is not want:
                continue
        out.append(k)
    return out
