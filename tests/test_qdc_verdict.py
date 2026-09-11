"""Every shape that must NOT be reported as a device pass.

Each test here corresponds to something that actually slipped through at some
point: a job that ran zero tests, a report whose skips were never parsed, a run
in which the DSP measured nothing. They are regression pins, not hypotheticals.
"""

import pytest

from hexkernels.device.qdc.verdict import (QdcResultsError, check_results,
                                           cycles_total_verdict,
                                           parse_results_xml)

GOOD_SUITE = '<testsuite tests="5" failures="0" errors="0" skipped="0"/>'
GOOD_LOG = "hexlib: --self-test: PASS\ncycles_total=41234\n"


@pytest.fixture
def job(tmp_path):
    """Build a fetched-log set. `suite=None` omits the report entirely."""
    def build(suite=GOOD_SUITE, log=GOOD_LOG, extra_suite=None):
        paths = []
        if suite is not None:
            p = tmp_path / "TestLogs" / "results.xml"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(suite)
            paths.append(str(p))
        if extra_suite is not None:
            p = tmp_path / "other" / "TestLogs" / "results.xml"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(extra_suite)
            paths.append(str(p))
        p = tmp_path / "run.log"
        p.write_text(log)
        paths.append(str(p))
        return paths
    return build


def test_a_genuine_run_passes(job):
    ok, msg = check_results(job())
    assert ok, msg
    assert "cycles_total=41234" in msg
    # The skip count is on the PASS line too, so a reader can tell "0 skipped"
    # from "skips were never looked at".
    assert "0 skipped" in msg


def test_zero_tests_is_never_a_pass(job):
    ok, msg = check_results(job('<testsuite tests="0" failures="0" errors="0" skipped="0"/>'))
    assert not ok
    assert "ran no tests" in msg


def test_cycles_total_zero_is_not_a_measurement(job):
    """The DSP reporting 0 cycles is the expected silicon failure, not an edge case."""
    ok, msg = check_results(job(log="hexlib: --self-test: PASS\ncycles_total=0\n"))
    assert not ok
    assert "measured NOTHING" in msg


def test_a_skip_is_not_a_pass(job):
    ok, msg = check_results(job('<testsuite tests="5" failures="0" errors="0" skipped="5"/>'))
    assert not ok
    assert "skipped" in msg
    assert "never run" in msg


def test_a_missing_attribute_is_not_a_zero(job):
    ok, msg = check_results(job('<testsuite tests="5" errors="0" skipped="0"/>'))
    assert not ok
    assert "parse" in msg


def test_a_negative_count_is_refused(job):
    ok, _ = check_results(job('<testsuite tests="-1" failures="0" errors="0" skipped="0"/>'))
    assert not ok


def test_two_reports_are_ambiguous_not_a_pass(job):
    ok, msg = check_results(job(extra_suite=GOOD_SUITE))
    assert not ok
    assert "refusing to guess" in msg


def test_no_report_at_all(job):
    ok, msg = check_results(job(suite=None))
    assert not ok
    assert "no results is a failure" in msg


def test_missing_measurement_lines(job):
    ok, msg = check_results(job(log="nothing useful here\n"))
    assert not ok
    assert "measurement line" in msg


def test_failures_are_reported_with_their_counts(job):
    ok, msg = check_results(job('<testsuite tests="5" failures="2" errors="1" skipped="0"/>'))
    assert not ok
    assert "2 failure(s)" in msg and "1 error(s)" in msg


@pytest.mark.parametrize("log,expected", [
    ("cycles_total=100\ncycles_total=0\n", True),    # at least one positive
    ("cycles_total=0\ncycles_total=0\n", False),     # PCYCLE never advanced
    ("cycles_total=abc\n", False),                   # malformed != absent
    ("no marker at all\n", False),
])
def test_cycles_verdict_states(log, expected):
    ok, _ = cycles_total_verdict(log)
    assert ok is expected


def test_nested_suites_are_refused_not_summed(tmp_path):
    """Summing nested totals would double-count a parent's rolled-up counts."""
    p = tmp_path / "results.xml"
    p.write_text('<testsuites><testsuite tests="5" failures="0" errors="0" '
                 'skipped="0"><testsuite tests="5" failures="0" errors="0" '
                 'skipped="0"/></testsuite></testsuites>')
    with pytest.raises(QdcResultsError, match="nested"):
        parse_results_xml(str(p))
