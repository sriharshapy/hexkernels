"""Achievability-witness kernels in benchmark/candidates/ must be auditable.

These kernels were ported from an external pipeline (see
benchmark/candidates/PROVENANCE.md). Three properties keep the import honest:
every file names a real task in this repo's frozen set, the manifest exactly
covers what's on disk with correct hashes, and none of them reference the
source project's forbidden R&D helper symbols.
"""
import hashlib
import json
import pathlib

from hexkernels.forge import mined

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
CANDIDATES_DIR = REPO_ROOT / "benchmark" / "candidates"
MANIFEST_PATH = CANDIDATES_DIR / "MANIFEST.json"

# Symbols that identify R&D helper material the source project forbade
# candidates from touching (hmx_helpers.h / harness_common.h and their
# internals). A hit here means a kernel leaked non-vendor internals.
LEAKED_SYMBOLS = (
    "HVX_VTCM_BASE",
    "HVX_ALIGN",
    "hvx_crouton_off",
    "hvx_hmx_enable",
    "hvx_close_f32",
    "hvx_report",
    "hmx_tile_matmul",
    "hvx_hmx_i8_",
    "harness_common.h",
    "hmx_helpers.h",
)


def _cpp_files():
    return sorted(CANDIDATES_DIR.glob("*.cpp"))


def _task_names():
    return {s.name for specs in mined.MINED_BATCHES.values() for s in specs}


def test_candidates_dir_is_populated():
    assert CANDIDATES_DIR.is_dir()
    assert _cpp_files(), "expected at least one imported candidate kernel"


def test_every_candidate_matches_a_real_task_name():
    names = _task_names()
    for path in _cpp_files():
        assert path.stem in names, (
            f"{path.name} does not match any task name in mined.MINED_BATCHES"
        )


def test_manifest_exists_and_is_valid_json():
    assert MANIFEST_PATH.is_file()
    json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def test_manifest_covers_every_cpp_file_present():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    on_disk = {path.stem for path in _cpp_files()}
    assert set(manifest.keys()) == on_disk, (
        set(manifest.keys()) ^ on_disk
    )


def test_manifest_sha256_matches_file_on_disk():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for name, entry in manifest.items():
        path = CANDIDATES_DIR / f"{name}.cpp"
        data = path.read_bytes()
        actual_sha = hashlib.sha256(data).hexdigest()
        assert actual_sha == entry["sha256"], (
            f"{name}: manifest sha256 does not match file on disk"
        )
        assert entry["name"] == name
        assert entry["source_path"], f"{name}: manifest missing source_path"


def test_no_candidate_leaks_an_rd_helper_symbol():
    for path in _cpp_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for symbol in LEAKED_SYMBOLS:
            assert symbol not in text, (
                f"{path.name} references forbidden R&D symbol {symbol!r}"
            )


# --- Tier-match bookkeeping (see PROVENANCE.md's "NOT 171 witnesses" section) ---
#
# Every manifest entry should carry task_tier / source_tier / tier_match, and
# those three must not silently drift apart from each other or from what the
# rest of this repo currently believes a task's tier is.
#
# The full check described in the task -- recomputing source_tier from the
# source repo's own run_artifacts/forge2/batch*/<kernel>/provenance.json and
# comparing the live-computed tier-match count against what's recorded here --
# would require this test to read files outside this repository (an external
# path on the machine that produced the import, not something checked into
# hexbench, and not guaranteed to exist wherever these tests run). That
# conflicts with the project rule that nothing in this repo reads from outside
# it, so this test does NOT do that. Instead it asserts everything that CAN be
# verified from inside the repo:
#   1. every entry carries the three fields, with the right shape/typing;
#   2. tier_match is not a free variable -- it is exactly the equality of the
#      recorded task_tier/source_tier pair (or None when either is unresolved);
#   3. task_tier itself has not drifted from what hexkernels.forge.mined.MINED_BATCHES
#      (i.e. benchmark/selection.json) says today, since that part IS derivable
#      from inside the repo and is exactly the kind of drift this test exists
#      to catch.
# What this deliberately does NOT re-verify is source_tier's fidelity to the
# external provenance.json files -- that half of the claim is only as good as
# the one-time import script that populated it, audited by a human at import
# time, not by this test suite on every run.


def _live_task_tier_by_name():
    """This repo's current per-task tier, keyed by task name.

    Built the same way benchmark/candidates were matched to tasks: flatten
    hexkernels.forge.mined.MINED_BATCHES (which itself loads benchmark/selection.json)
    into a name -> expect_tier mapping. A handful of task names recur across
    batches with different tiers (fused compositions reused at multiple
    sizes); none of those ambiguous names are among the imported candidates,
    so a plain last-write-wins dict is unambiguous for every name this test
    actually looks up.
    """
    out = {}
    for specs in mined.MINED_BATCHES.values():
        for s in specs:
            out[s.name] = s.expect_tier
    return out


def test_manifest_entries_carry_tier_match_fields():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for name, entry in manifest.items():
        for field in ("task_tier", "source_tier", "tier_match"):
            assert field in entry, f"{name}: manifest entry missing {field!r}"
        assert entry["task_tier"] is None or isinstance(entry["task_tier"], str)
        assert entry["source_tier"] is None or isinstance(entry["source_tier"], str)
        assert entry["tier_match"] is None or isinstance(entry["tier_match"], bool)


def test_tier_match_boolean_agrees_with_recorded_tier_pair():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    for name, entry in manifest.items():
        task_tier = entry["task_tier"]
        source_tier = entry["source_tier"]
        if task_tier is None or source_tier is None:
            assert entry["tier_match"] is None, (
                f"{name}: tier_match should be null when a tier is unresolved"
            )
        else:
            assert entry["tier_match"] == (task_tier == source_tier), (
                f"{name}: tier_match does not match the recorded tier pair "
                f"({task_tier!r} vs {source_tier!r})"
            )


def test_manifest_task_tier_matches_live_selection_json():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    live = _live_task_tier_by_name()
    for name, entry in manifest.items():
        assert name in live, f"{name}: no longer a task in benchmark/selection.json"
        assert entry["task_tier"] == live[name], (
            f"{name}: manifest task_tier {entry['task_tier']!r} has drifted from "
            f"the live tier {live[name]!r} in benchmark/selection.json"
        )
