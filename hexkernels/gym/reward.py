"""Tiered reward for HexagonGym, from RELIABLE evaluate() signals only.
0 uncompiled < 1 compiled-incorrect < 2 correct-but-not-genuine < 3 correct+genuine.
Speed is deliberately EXCLUDED from the tier (box-gated; reported separately)."""

MECH_FLAG = {
    "hvx": "used_hvx_compute",
    "hmx": "used_hmx",
    "dma": "used_dma",
    "vtcm": "used_vtcm",
    "l2fetch": "used_l2fetch",
}


def target_mechanisms(spec):
    mechs = spec.get("mechanisms") or []
    return [m for m in MECH_FLAG if m in mechs]


def is_genuine(feedback, spec):
    if not feedback.get("correct"):
        return False
    # NOTE: all([]) is vacuously True -- a spec with NO target mechanisms makes any correct
    # kernel "genuine" by definition (nothing left to check mechanism-wise), so it's tier 3
    # and the gym episode ends on correct. This is intentional, not a bug -- do not "fix" it.
    return all(feedback.get(MECH_FLAG[m]) for m in target_mechanisms(spec))


def reward_tier(feedback, spec):
    if not feedback.get("compiled"):
        return 0
    if not feedback.get("correct"):
        return 1
    return 3 if is_genuine(feedback, spec) else 2
