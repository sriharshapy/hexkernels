"""Decide whether a device job actually MEASURED something, or merely finished.

`job.wait()` is a completion detector: it returns True when a file named
`TestLogs/results.xml` appears among a job's logs. It does not open that file.
That is deliberate and documented in `job.py` -- but it means `wait() -> True`
is a PRECONDITION for a verdict, never the verdict itself.

This module is the verdict. It exists because of one observed failure on this
account: **a job that ran zero tests reported passing.** Every check below was
added after something in this shape slipped through, and each has its own
message so a real failure is diagnosable from which check tripped:

  * a results.xml was fetched, is unambiguous, and PARSES as XML;
  * every `<testsuite>` carries all four of `REQUIRED_SUITE_ATTRS` -- a
    missing `failures` is a malformed report, NEVER a zero;
  * `tests > 0`. Zero tests is a failure; so is a negative count;
  * `failures == 0` and `errors == 0`;
  * `skipped == 0`. **A SKIP IS NOT A PASS.** Every test on the device path is
    unconditional, so a skip means pytest collected a test and never ran it --
    the device was unreachable or collection half-failed. Reported separately
    from failures because "5 skipped" and "5 failed" call for completely
    different next actions;
  * the logs contain the measurement lines a genuine run prints;
  * and `cycles_total=` carries a value **greater than zero**.

That last one is not pedantry. The DSP reads PCYCLE inside a user-mode unsigned
PD, where `SYSCFG.PCYCLEEN` cannot be set -- so `cycles_total=0` is exactly what
a silicon job prints when the counter never advances. For a while the check was
only `"cycles_total=" in logs`, and `cycles_total=0` satisfied it: a run in
which the DSP measured literally nothing printed "measurement lines present"
and exited 0.

Vendored from `hexlib/cli.py`, where the budget-guard and submit paths still
live. The parsing and verdict logic is reproduced here in full because it is the
part that keeps a false pass off the money path.
"""

from __future__ import annotations

import os
import re
import xml.etree.ElementTree as ET
from typing import NamedTuple

__all__ = [
    "CYCLES_TOTAL_MARKER", "REQUIRED_SUITE_ATTRS", "JUnitCounts",
    "QdcResultsError", "parse_results_xml", "cycles_total_verdict",
    "check_results",
]

CYCLES_TOTAL_MARKER = "cycles_total="
SELFTEST_PASS_MARKER = "--self-test: PASS"

# `cycles_total=0` CONTAINS `cycles_total=`. That is the whole reason this regex
# exists rather than a substring test. `(\S*)` deliberately captures whatever
# follows, valid or not, so a MALFORMED value stays distinguishable from an
# ABSENT line instead of both reading as "no match". Built from the marker
# rather than respelling it, so presence and value checks cannot drift apart.
_CYCLES_TOTAL_RE = re.compile(re.escape(CYCLES_TOTAL_MARKER) + r"(\S*)")
_CYCLES_TOTAL_DIGITS = re.compile(r"[0-9]+")

# pytest's own `--junitxml` always writes all four. Refusing a report that lacks
# any of them cannot reject a report our own job produced; it rejects a
# truncated or foreign one, which is the safe direction to fail in.
REQUIRED_SUITE_ATTRS = ("tests", "failures", "errors", "skipped")

RESULTS_MARKER = os.path.join("TestLogs", "results.xml").replace("\\", "/")


class QdcResultsError(Exception):
    """A results.xml that must NOT be treated as a pass.

    Unparseable, a shape this project does not produce, missing any of
    `REQUIRED_SUITE_ATTRS` (a missing one is never a zero), or carrying a
    negative count. Caught by `check_results`, never allowed to propagate.
    """


class JUnitCounts(NamedTuple):
    """All four counts, all required.

    `skipped` is here because it was for a while parsed NOWHERE, which made a
    collected-but-never-run test indistinguishable from a passing one: a suite
    reporting `tests="5" failures="0" errors="0" skipped="5"` exited 0 and
    never mentioned the skips. On a device, a skip overwhelmingly means the
    test could not run at all -- absence read as success, wearing a different
    attribute name.
    """

    tests: int
    failures: int
    errors: int
    skipped: int


def results_filename(path: str) -> bool:
    """True for the real report, false for a framework placeholder.

    Matched on the full `TestLogs/results.xml` suffix, NOT the basename. Two
    files named `results.xml` can legitimately exist once the fetch mirrors
    QDC's directory layout, and a placeholder listed first would otherwise be
    parsed as the verdict.
    """
    return path.replace("\\", "/").endswith(RESULTS_MARKER)


def parse_results_xml(path: str) -> JUnitCounts:
    """Parse a JUnit results.xml, summing across `<testsuite>` elements.

    ONLY ONE SHAPE IS ACCEPTED, pinned to what this project actually produces
    rather than guessed at as a general JUnit parser. pytest's `--junitxml`
    emits exactly one `<testsuite>`, either as the root or as the sole child of
    a `<testsuites>` wrapper; it never nests one inside another. An earlier
    version summed `.//testsuite` at any depth, which would silently
    DOUBLE-COUNT a report whose parent totals already include a child's. This
    refuses that shape outright instead of guessing how to sum it.
    """
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        raise QdcResultsError(f"{path} did not parse as XML: {e}") from e

    if root.tag == "testsuite":
        suites = [root]
    elif root.tag == "testsuites":
        suites = root.findall("testsuite")      # DIRECT children only.
    else:
        raise QdcResultsError(
            f"{path} root is <{root.tag}>, not <testsuite> or <testsuites> -- "
            "not a JUnit report shape this project recognizes"
        )
    if not suites:
        raise QdcResultsError(
            f"{path} contains no <testsuite> element -- not a JUnit report "
            "this project recognizes"
        )

    for suite in suites:
        if suite.findall(".//testsuite"):
            raise QdcResultsError(
                f"{path} has a <testsuite> nested inside another -- not the "
                "flat shape pytest's --junitxml produces, and summing nested "
                "totals would double-count them; refusing rather than guessing"
            )

    totals = dict.fromkeys(REQUIRED_SUITE_ATTRS, 0)
    for suite in suites:
        for attr in REQUIRED_SUITE_ATTRS:
            raw = suite.get(attr)
            if raw is None:
                raise QdcResultsError(
                    f"{path} has a <testsuite> with no {attr!r} attribute -- "
                    "pytest always writes all of "
                    f"{', '.join(REQUIRED_SUITE_ATTRS)}, so an absent one is a "
                    "malformed report and must NEVER be read as zero"
                )
            try:
                value = int(raw)
            except ValueError as e:
                raise QdcResultsError(
                    f"{path} has a non-integer {attr!r} attribute: {e}") from e
            if value < 0:
                raise QdcResultsError(
                    f"{path} has {attr}={raw!r}, a NEGATIVE count -- pytest "
                    "cannot produce that, and a negative must not be summed "
                    "into a total that then compares as 'no failures'"
                )
            totals[attr] += value
    return JUnitCounts(**totals)


def cycles_total_verdict(combined: str) -> tuple[bool, str]:
    """`(ok, detail)` for the `cycles_total=` measurement in the fetched logs.

    `ok` only if at least one line carries a non-negative decimal integer
    strictly greater than zero. `detail` always names which of four states was
    found, so the caller's message states the real problem rather than
    "missing": no line at all; a value that is not an integer; every line
    reporting 0 (the DSP measured nothing); or at least one positive value.

    AT LEAST ONE, not all: a job's logs legitimately contain several
    `cycles_total=` lines and not every mode need have measured something. If
    PCYCLE is dead in the unsigned PD then EVERY line reads 0, so taking the
    maximum cannot hide the failure this exists to catch.
    """
    found = _CYCLES_TOTAL_RE.findall(combined)
    if not found:
        return False, f"no `{CYCLES_TOTAL_MARKER}` line anywhere in the fetched logs"

    values, malformed = [], []
    for raw in found:
        if _CYCLES_TOTAL_DIGITS.fullmatch(raw):
            values.append(int(raw, 10))
        else:
            malformed.append(raw)

    positive = [v for v in values if v > 0]
    if positive:
        return True, f"{CYCLES_TOTAL_MARKER}{max(positive)}"
    if values:
        return False, (
            f"every `{CYCLES_TOTAL_MARKER}` line reports 0 ({len(values)} "
            "such line(s)) -- the DSP measured NOTHING, which is what PCYCLE "
            "returns when SYSCFG.PCYCLEEN is clear, and a user-mode unsigned "
            "PD cannot set it"
        )
    return False, (
        f"`{CYCLES_TOTAL_MARKER}` is present but its value is not a decimal "
        f"integer: {malformed[0]!r}"
    )


def check_results(paths, job_id="?"):
    """`(ok, message)`. True only if the job genuinely ran and measured.

    `paths` is every fetched log file. Never raises: a parse failure is
    reported as a failure, because a verdict read from a guess is worse than
    no verdict.
    """
    matches = [p for p in paths if results_filename(p)]
    if len(matches) > 1:
        return False, (
            f"job {job_id}: {len(matches)} files match {RESULTS_MARKER} "
            f"({', '.join(sorted(matches))}) -- refusing to guess which is the "
            "verdict. A job whose report cannot be identified unambiguously is "
            "a failure, never a pass."
        )
    if not matches:
        return False, (
            f"job {job_id}: no {RESULTS_MARKER} among the fetched log files -- "
            "a job with no results is a failure, never a pass"
        )

    try:
        tests, failures, errors, skipped = parse_results_xml(matches[0])
    except QdcResultsError as e:
        return False, (
            f"job {job_id}: could not parse results.xml as a JUnit report -- a "
            f"truncated or unparseable results file is a failure, never a "
            f"pass: {e}"
        )

    # `<= 0`, not `== 0`: a comparison that only catches the exact value it was
    # written for is not a bound.
    if tests <= 0:
        return False, (f"job {job_id}: results.xml reports {tests} tests -- a "
                       "job that ran no tests is a failure, never a pass")
    if failures or errors:
        return False, (f"job {job_id}: results.xml reports {failures} "
                       f"failure(s) and {errors} error(s) across {tests} "
                       f"test(s) ({skipped} skipped)")
    if skipped:
        return False, (
            f"job {job_id}: results.xml reports {skipped} skipped test(s) out "
            f"of {tests} -- every test on the device path is unconditional, so "
            "a skip means a test was collected and never run. "
            "Collected-but-not-run is not passed."
        )

    combined = ""
    for p in paths:
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                combined += fh.read()
        except OSError:
            continue

    missing = [m for m in (CYCLES_TOTAL_MARKER, SELFTEST_PASS_MARKER)
               if m not in combined]
    if missing:
        return False, (
            f"job {job_id}: results.xml reports {tests} test(s) with no "
            "failures, but the fetched logs are missing the expected "
            f"measurement line(s): {', '.join(repr(m) for m in missing)} -- a "
            "pass with no measurements behind it is the exact failure mode "
            "this check exists to rule out"
        )

    # PRESENCE IS NOT MEASUREMENT. The check above only proved the substring
    # appears; this one reads the number after it.
    ok, detail = cycles_total_verdict(combined)
    if not ok:
        return False, (
            f"job {job_id}: results.xml reports {tests} test(s) with no "
            f"failures, but the DSP's own cycle measurement is not usable: "
            f"{detail} -- a pass whose only measurement is zero is still a "
            "pass with no measurements behind it"
        )

    # The skip count is printed on the PASS line too, not only when nonzero: a
    # success line that omits a count it checked leaves a reader unable to tell
    # "0 skipped" from "skips were never looked at".
    return True, (f"job {job_id}: {tests} test(s), 0 failures, 0 errors, "
                  f"{skipped} skipped, {detail}")
