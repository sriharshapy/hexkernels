"""The audit must FAIL on each defect it claims to catch. An audit that only
ever returns clean is indistinguishable from one that checks nothing."""
from hexkernels.forge import audit, testset


def _spec(op, tier, mechs, dtype="float32", n_prim=2):
    return {"op": op, "overload": "", "sig": "binary", "dtype": dtype,
            "dtype_bytes": 4 if dtype == "float32" else 2, "shape": [64, 64],
            "tier": tier, "mechanisms": list(mechs), "working_set": 16384,
            "n_primitives": n_prim,
            "schema": f"aten::{op}(Tensor self) -> Tensor",
            "accessor": "torch._C._jit_get_all_schemas()"}


def _clean_set():
    specs = []
    mech = {"T0": ["hvx"], "T1": ["hvx", "l2fetch"],
            "T2": ["hvx", "l2fetch", "vtcm", "dma"],
            "T3": ["hvx", "l2fetch", "vtcm", "dma"]}
    for tier in testset.TIERS:
        for i in range(74):
            specs.append(_spec(f"f_{tier}_{i}", tier, mech[tier],
                               n_prim=1 + (i % 4)))
        for i in range(6):
            specs.append(_spec(f"c_{tier}_{i}", tier, mech[tier] + ["hmx"],
                               "float16", n_prim=1 + (i % 4)))
        specs.append(_spec("linear", tier, mech[tier]))  # fp32 control
    return specs


def test_clean_set_has_no_violations():
    specs = _clean_set()
    core = testset.designate_core(specs[:320]) if len(specs) >= 320 else []
    rep = audit.audit(specs, core)
    assert rep["violations"] == [], rep["violations"]


def test_flags_a_starved_mechanism():
    specs = [s for s in _clean_set() if "hmx" not in s["mechanisms"]]
    rep = audit.audit(specs, [])
    assert any("hmx" in v for v in rep["violations"])


def test_flags_an_empty_mechanism_set():
    specs = _clean_set()
    specs[0]["mechanisms"] = []
    rep = audit.audit(specs, [])
    assert any("no mechanisms" in v for v in rep["violations"])


def test_flags_missing_negative_controls():
    specs = [s for s in _clean_set() if s["op"] != "linear"]
    rep = audit.audit(specs, [])
    assert any("negative control" in v for v in rep["violations"])


def test_flags_hmx_granted_at_fp32():
    specs = _clean_set()
    for s in specs:
        if s["op"] == "linear":
            s["mechanisms"] = list(s["mechanisms"]) + ["hmx"]
    rep = audit.audit(specs, [])
    assert any("fp32" in v and "hmx" in v for v in rep["violations"])


def test_flags_degenerate_depth_within_a_tier():
    """If every T3 task is depth 4 and every T0 is depth 1, 'big is hard' and
    'deep is hard' cannot be separated."""
    specs = _clean_set()
    for s in specs:
        s["n_primitives"] = {"T0": 1, "T1": 1, "T2": 4, "T3": 4}[s["tier"]]
    rep = audit.audit(specs, [])
    assert any("depth" in v for v in rep["violations"])


def test_report_renders_every_section():
    rep = audit.audit(_clean_set(), [])
    text = audit.render(rep)
    for heading in ("Tier counts", "Mechanism counts", "Depth", "Negative controls"):
        assert heading in text


# -- Amendment-added behaviour (composition, known-absent, mining note) --
# The audit's own docstring argues "an audit that only ever returns clean is
# indistinguishable from one that checks nothing." That applies to the
# audit's own amendment-added code paths too, not just the brief's checks.

def test_is_fused_derives_from_stages_key_not_schema_string():
    """Discriminating on purpose: a spec whose SCHEMA says 'fused:' but has
    no `stages` key must count as single-op, and a spec with `stages` whose
    schema does not mention 'fused' must count as fused. A test built only
    from well-formed specs would pass against a schema-string implementation
    just as well as against the real one, so it proves nothing."""
    masquerading_single = _spec("weird", "T0", ["hvx"])
    masquerading_single["schema"] = "fused: aten::weird then aten::other"
    assert "stages" not in masquerading_single
    assert audit._is_fused(masquerading_single) is False

    plainly_schemad_fused = _spec("weird2", "T0", ["hvx"])
    plainly_schemad_fused["stages"] = [["weird2", ""], ["other", ""]]
    plainly_schemad_fused["schema"] = "aten::weird2(Tensor self) -> Tensor"
    assert audit._is_fused(plainly_schemad_fused) is True


def test_composition_counts_correct_for_a_small_mixed_set():
    a = _spec("a", "T0", ["hvx"])                       # T0 single
    b = _spec("b", "T0", ["hvx", "l2fetch"])             # T0 fused
    b["stages"] = [["b", ""], ["c", ""]]
    d = _spec("d", "T1", ["hvx", "vtcm"])                # T1 fused
    d["stages"] = [["d", ""], ["e", ""]]
    f = _spec("f", "T1", ["hvx"])                        # T1 single

    rep = audit.audit([a, b, d, f], [])
    comp = rep["composition"]
    assert comp["by_tier"]["T0"] == {"single": 1, "fused": 1}
    assert comp["by_tier"]["T1"] == {"single": 1, "fused": 1}
    assert comp["by_tier"]["T2"] == {"single": 0, "fused": 0}
    assert comp["by_tier"]["T3"] == {"single": 0, "fused": 0}
    assert comp["by_mechanism"]["hvx"] == {"single": 2, "fused": 2}
    assert comp["by_mechanism"]["l2fetch"] == {"single": 0, "fused": 1}
    assert comp["by_mechanism"]["vtcm"] == {"single": 0, "fused": 1}


def test_known_absent_reports_stale_waiver_when_op_reappears():
    """A waiver that cannot expire is a permanent blind spot. If a
    KNOWN_ABSENT op reappears in the specs, the report must say so instead
    of continuing to describe it as absent."""
    specs = _clean_set()
    specs[0]["op"] = "gelu"
    rep = audit.audit(specs, [])
    assert rep["known_absent"]["gelu"]["absent"] is False
    # the other two known-absent ops are genuinely still absent
    assert rep["known_absent"]["amax"]["absent"] is True
    assert rep["known_absent"]["amin"]["absent"] is True

    text = audit.render(rep)
    assert "stale" in text.lower() and "gelu" in text


def test_known_absent_ops_never_produce_violations():
    """Deliberate and worth pinning: KNOWN_ABSENT is disclosure, not
    enforcement. A later change must not silently start failing the audit
    on a disclosed gap."""
    specs = _clean_set()
    rep = audit.audit(specs, [])
    assert rep["violations"] == []
    for op in ("amax", "amin", "gelu"):
        assert not any(op in v for v in rep["violations"])

    # even when a known-absent op reappears, that is disclosure only
    specs2 = _clean_set()
    specs2[0]["op"] = "gelu"
    rep2 = audit.audit(specs2, [])
    assert not any("gelu" in v for v in rep2["violations"])
