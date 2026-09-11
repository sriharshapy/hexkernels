"""Pure per-mechanism aggregation of gym records -> the reliable headline scoreboard."""
from hexkernels.gym import metrics as MX
from hexkernels.gym import reward as R


def scoreboard(records):
    per = {}
    correct = 0
    for rec in records:
        fb, spec = rec["feedback"], rec["spec"]
        if fb.get("correct"):
            correct += 1
        for m in R.target_mechanisms(spec):
            slot = per.setdefault(m, {"n": 0, "genuine": 0, "rate": 0.0})
            slot["n"] += 1
            if fb.get("correct") and fb.get(R.MECH_FLAG[m]):
                slot["genuine"] += 1
    for slot in per.values():
        slot["rate"] = slot["genuine"] / slot["n"] if slot["n"] else 0.0
    n = len(records)
    # Additive: every existing key keeps its meaning, so prior artifacts stay comparable.
    # `conversion` is the headline axis -- genuine <= correct by construction, so raw
    # genuine counts conflate coding ability with reaching for the accelerator.
    return {"per_mechanism": per, "correct_rate": (correct / n if n else 0.0), "n": n,
            "conversion": MX.conversion(records),
            "mechanism_weighted": MX.mechanism_weighted(records),
            "isa_quality": MX.isa_quality(records)}
