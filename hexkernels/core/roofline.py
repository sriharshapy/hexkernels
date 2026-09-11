"""Pure roofline-efficiency computation. No I/O. eta = achieved/peak in [0,1],
where peak = min(compute_peak[dtype,engine], AI * bandwidth[binding_level]) and
the binding level is where work_bytes fits in the cache hierarchy.

Spec dtype strings (e.g. "int8->int32", "i8->i8", "uint8", "int8+uint8->int32")
are canonicalized to bare element-class keys ("int8", "int16", "fp16", "fp32")
before the compute-peak lookup via _canonical_compute_dtype()."""

import re as _re


def _canonical_compute_dtype(dtype: str) -> str:
    """Map a spec dtype string to a canonical compute-peak key.

    Rules:
    - Take the substring before '->' if present (operand/input side).
    - Find the first type token in that substring.
    - Map by element byte width:
        int8 / i8 / uint8 / u8          -> "int8"
        int16 / i16 / uint16 / u16       -> "int16"
        fp16 / float16 / __fp16          -> "fp16"
        fp32 / float32 / float           -> "fp32"
        int32 / i32                      -> "int32"  (no model key -> bw ceiling)
        unknown / unparseable            -> "int8"   (default, dominant dtype)
    """
    # Take operand (left) side of any '->'
    operand = dtype.split("->")[0].strip()
    # Find the first type token (word characters, possibly with digits)
    tokens = _re.findall(r'[a-zA-Z_][a-zA-Z0-9_]*', operand)
    first = tokens[0].lower() if tokens else ""
    if first in ("int8", "i8", "uint8", "u8"):
        return "int8"
    if first in ("int16", "i16", "uint16", "u16"):
        return "int16"
    if first in ("fp16", "float16", "__fp16"):
        return "fp16"
    if first in ("fp32", "float32", "float"):
        return "fp32"
    if first in ("int32", "i32"):
        return "int32"
    # Default for unknown/unparseable dtypes
    return "int8"


def _binding_level(work_bytes, model):
    """Return the cache/memory level where work_bytes fits.
    L1D and L2 are hardware caches (automatic). VTCM is software-managed
    scratchpad — not part of the automatic hierarchy; use DDR for anything
    that doesn't fit in L2."""
    sizes = model["level_size_bytes"]
    if work_bytes <= sizes["L1D"]:
        return "L1D"
    if work_bytes <= sizes["L2"]:
        return "L2"
    return "DDR"


def effective_bandwidth(work_bytes, level, model):
    """Size-aware (Hockney r_inf/n_half) achievable bandwidth at a footprint, B/cycle.

    eff_bw(B) = r_inf * B / (B + n_half), so B->inf gives r_inf and B=n_half gives r_inf/2.
    Falls back to the constant asymptotic bandwidth_bytes_per_cycle[level] when the model
    has no bandwidth_model block (backward compatibility)."""
    bm = model.get("bandwidth_model", {}).get(level)
    if not bm:
        return model["bandwidth_bytes_per_cycle"][level]
    r_inf = bm["r_inf_bytes_per_cycle"]
    n_half = bm["half_bytes"]
    return r_inf * work_bytes / (work_bytes + n_half)


def roofline_efficiency(*, useful_ops, work_bytes, kernel_cycles, dtype, engine, model):
    """Return {achieved, AI, binding_level, ceiling, eta, eta_raw, eff_bw, bw_asymptotic}.

    Keys:
      - achieved: ops/cycle; None if kernel_cycles is falsy.
      - AI: arithmetic intensity (useful_ops / work_bytes).
      - binding_level: cache level where work_bytes fits (L1D, L2, or DDR).
      - ceiling: peak achievable ops/cycle = min(compute_peak[dtype,engine], AI * bandwidth[level]).
        dtype is canonicalized via _canonical_compute_dtype() before the compute-peak lookup so
        rich spec strings like "int8->int32", "i8->i8", "uint8", "int8+uint8->int32" all resolve
        to the model's bare keys (hvx_int8, hvx_int16, hvx_fp16, ...).
        If compute_peak key is absent (e.g. int32 operand), ceiling = bandwidth ceiling alone.
        None if kernel_cycles is falsy.
      - eta: efficiency ratio (achieved / ceiling) clamped to [0, 1]; None if ceiling ≤ 0 or kernel_cycles is falsy.
      - eta_raw: unclamped achieved / ceiling ratio; None if ceiling ≤ 0 or kernel_cycles is falsy.
      - eff_bw: size-aware effective bandwidth used (B/cycle); None if kernel_cycles is falsy.
      - bw_asymptotic: asymptotic bandwidth ceiling for the binding level (B/cycle); None if kernel_cycles is falsy.

    Contract: model["bandwidth_bytes_per_cycle"] MUST contain keys L1D, L2, DDR.
    Missing level is a programmer error and raises KeyError by design."""
    if not kernel_cycles or not work_bytes:
        return {"achieved": None, "AI": None, "binding_level": None,
                "ceiling": None, "eta": None, "eta_raw": None,
                "eff_bw": None, "bw_asymptotic": None}
    level = _binding_level(work_bytes, model)
    bw = effective_bandwidth(work_bytes, level, model)
    bw_asymptotic = model["bandwidth_bytes_per_cycle"][level]
    ai = useful_ops / work_bytes
    canonical = _canonical_compute_dtype(dtype)
    key = f"{engine}_{canonical}"
    compute_peak = model["compute_peak_ops_per_cycle"].get(key)
    bw_ceiling = ai * bw
    ceiling = bw_ceiling if compute_peak is None else min(compute_peak, bw_ceiling)
    achieved = useful_ops / kernel_cycles
    eta_raw = achieved / ceiling if ceiling > 0 else None
    eta = None if eta_raw is None else max(0.0, min(1.0, eta_raw))
    return {"achieved": achieved, "AI": ai, "binding_level": level,
            "ceiling": ceiling, "eta": eta, "eta_raw": eta_raw,
            "eff_bw": bw, "bw_asymptotic": bw_asymptotic}
