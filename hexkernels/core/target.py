"""The Hexagon target we compile and simulate for.

One place for every architecture-dependent constant, so that retargeting is a
config change rather than a hunt through literals. Before this module existed,
the VTCM aperture was hardcoded in three places (`anticheat._VTCM_ADDR`,
`harness_common.h`, `anticheat_runtime`), and moving to any newer part would
have silently reported `used_vtcm=False` for every kernel -- fail-closed, but
completely blind, with nothing to flag it.

VALUES ARE PROBED, NOT ASSUMED. Every field below was read from the target's own
configuration table via `__rdcfg()` (offsets from the SDK's
`hexagon_standalone.h`), plus the simulator's own core table. That discipline
exists because the VTCM figure was wrong twice in this project from inference --
first 24 KB, then 256 KB, which is the minimum allocation/page size mistaken for
capacity, an error of 16x.

  HVX PRM 80-N2040-47 Rev. E section 3.2 p15: "The size of the memory is
  implementation-defined. The size is discoverable from the configuration table
  defined in the V68 system architecture specification."

Probe transcript (`hexagon-sim --mvXX`, tools 19.0.04):

    v68  rev_id 0x00008d68 (v68n_1024)  __vtcm_base 0xd840  __tcm_size 4096
    v75  rev_id 0x00008c75 (v75na_1)    __vtcm_base 0xd900  __tcm_size 8192

Both carry the H3 coprocessor (HMX) in the simulator's core table. Only the
`n*`/`q*` core families do; the `m*` (modem), `l*` (low-power) and `h*` families
do not, which is what identifies these as NSP/cDSP parts rather than another DSP
role. Note that identification is inferred from the coprocessor column and the
v79 core spread -- the SDK does not document the family letters.

Select with the ``HEXBENCH_TARGET`` environment variable. Default is ``v75``.
``v68`` remains available because prior cycle measurements were taken on it and
are not comparable across microarchitectures.
"""
import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Target:
    arch: str          # compiler/sim flag stem: -mv75
    core: str          # rev_id name the simulator reports
    revid: str
    l1d_bytes: int
    l2_bytes: int
    vtcm_bytes: int
    vtcm_base: int
    hvx_bytes: int
    note: str

    @property
    def vtcm_end(self) -> int:
        """One past the last VTCM byte."""
        return self.vtcm_base + self.vtcm_bytes


TARGETS = {
    "v68": Target(
        arch="v68", core="v68n_1024", revid="0x00008d68",
        l1d_bytes=16 * 1024, l2_bytes=1024 * 1024,
        vtcm_bytes=4096 * 1024, vtcm_base=0xd8400000, hvx_bytes=128,
        note="Hexagon 780 / Snapdragon 888-class. Retained because cycle "
             "measurements taken on it are not comparable to v75.",
    ),
    "v75": Target(
        arch="v75", core="v75na_1", revid="0x00008c75",
        l1d_bytes=16 * 1024, l2_bytes=1024 * 1024,
        vtcm_bytes=8192 * 1024, vtcm_base=0xd9000000, hvx_bytes=128,
        note="Cache-config-identical to v68n_1024 (6 threads, 32K L1-I, 16K "
             "L1-D, 1024K L2, H3, 128B HVX); VTCM is 2x and at a different "
             "base. Supported target for Qualcomm's hexagon-mlir, which tests "
             "v73/v75/v79 and not v68.",
    ),
}

DEFAULT_TARGET = "v75"


def current() -> Target:
    """The active target. ``HEXBENCH_TARGET`` overrides the default."""
    name = os.environ.get("HEXBENCH_TARGET", DEFAULT_TARGET)
    try:
        return TARGETS[name]
    except KeyError:
        raise ValueError(
            f"unknown HEXBENCH_TARGET={name!r}; known: {sorted(TARGETS)}"
        ) from None


def vtcm_immediate_pattern(t: Target) -> str:
    """Regex matching an immediate inside this target's VTCM aperture.

    VTCM is an address aperture, not an instruction set: code "uses VTCM" by
    naming an address inside it, so unlike DMA/HMX there is no mnemonic to
    match. Hexagon materialises such a 32-bit constant through the immediate
    extender, e.g. ``immext(#0xd9002000)``.

    Built from the probed base and size rather than written by hand, because a
    hand-written aperture is exactly what would break on retarget. Requires the
    aperture to be nibble-aligned, which both known targets are:
    v68 0xd8400000+4M -> 0xd84.....-0xd87.....,
    v75 0xd9000000+8M -> 0xd90.....-0xd97.....
    """
    lo, hi = t.vtcm_base, t.vtcm_end - 1
    # Shared leading hex digits, then a range over the first differing digit.
    lo_s, hi_s = f"{lo:08x}", f"{hi:08x}"
    common = 0
    while common < 8 and lo_s[common] == hi_s[common]:
        common += 1
    if common >= 8:
        raise ValueError("degenerate VTCM aperture")
    prefix = lo_s[:common]
    lo_d, hi_d = lo_s[common], hi_s[common]
    rest = 8 - common - 1
    if lo_s[common + 1:] != "0" * rest or hi_s[common + 1:] != "f" * rest:
        raise ValueError(
            f"VTCM aperture {lo:#x}-{hi:#x} is not nibble-aligned; the regex "
            "form cannot represent it. Use an explicit range check instead."
        )
    prefix_re = "".join(f"[{c}{c.upper()}]" if c.isalpha() else c for c in prefix)
    digit_re = f"[{lo_d}-{hi_d}]" if lo_d != hi_d else lo_d
    return rf"#0x{prefix_re}{digit_re}[0-9a-fA-F]{{{rest}}}\b"
