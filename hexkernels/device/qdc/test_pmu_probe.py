# hexlib/device/qdc/test_pmu_probe.py
"""STAGE-1 PROBE: can the CDSP's PMU counters be read on THIS QDC device?

This file answers ONE question and deliberately measures nothing else. A
silicon-only experiment (no `hexagon-sim`) gets every cache, DRAM, l2fetch,
VTCM and HMX number from the DSP's PMU or not at all, so whether the PMU is
reachable decides what that experiment can contain. It is not a metric; it is
the precondition for all of them.

WHY THIS IS NOT `HAP_user_pmu`. `$HEXAGON_SDK_ROOT/incs/HAP_user_pmu.md` says
two things that rule it out here: "HAP PMU APIs only work on debug-enabled
devices" and "The HAP PMU APIs are not accessible from unsigned PD."
`hexlib/runtime/host/session.c` opens an UNSIGNED PD by construction --
`enable_unsigned_pd()` sends DSPRPC_CONTROL_UNSIGNED_MODULE with enable=1 and
`hexlib_open` refuses a device reporting UNSIGNED_PD_SUPPORT = 0 -- and
`test_on_device.py` asserts `unsigned_pd_support = 1` on this very device. So
the HAP path is closed on the path hexlib actually runs, and confirming that
by writing skel code would spend device minutes proving a documented no.

WHY itrace INSTEAD. The itrace library reaches the same counters and is
documented to work in the mode hexlib is already in:
`libs/itrace/docs/README.md`'s `itrace_monitor` interface carries
`-U/--unsigned  Run on specified PD: 1 for Unsigned PD and 0 for Signed PD.
Default value is 1 for CDSP`. Unsigned CDSP is itrace's DEFAULT, not a
concession. It also beats the HAP path on three axes that matter downstream:
8 PMU slots at a time rather than 4 (`HAP_pmu_group_config_t` is
`pmu_events[4]`, "Only four unique PMU events can be tracked"), automatic
rotation through more events than fit (README: "designed to rotate through all
events"), and `itrace_start_section`/`_end_section` for counts scoped to a code
region rather than the whole program.

NO CODE IS COMPILED FOR THIS PROBE. `itrace_monitor` ships PREBUILT for
android_aarch64 (`libs/itrace/prebuilt/android_aarch64/`), so the probe pushes
a vendor binary and reads its output. Nothing in hexlib's IDL, skel or host is
touched, which is the point: a negative result here must be a fact about the
DEVICE, not about code written to ask the question.

EVENTS ARE NAMED, NEVER NUMBERED, AND THAT IS LOAD-BEARING. The
`ITRACE_DSP_EVENT_PMU_*` constants are NOT the hardware opcodes, and the
offset is not constant, so a number copied from the header selects the wrong
counter silently:

    ITRACE_DSP_EVENT_PMU_COMMITTED_PKT_ANY          0x8002  -> opcode 0x03
    ITRACE_DSP_EVENT_PMU_AXI_LINE128_READ_REQUEST   0x8037  -> opcode 0x3f
    ITRACE_DSP_EVENT_PMU_AXI_WRITE_REQUEST          0x803a  -> opcode 0x42
    ITRACE_DSP_EVENT_PMU_COMMITTED_FPS              0x8046  -> opcode 0x50

(header values from `libs/itrace/inc/itrace_dsp_events_pmu.h`; opcodes from the
annotated example `examples/itrace/src_app/advanced_run.c:471-479`). The deltas
are +1, +8, +8, +10 -- there is a mapping table, not an offset. README says
each event may be given "by its name or opcode", so this file gives names and
never numbers.

THE WORKLOAD IS `hexlib_run --self-test`, ALREADY PROVEN ON THIS DEVICE. A
counter framework that works reads zero against an idle DSP, which is
indistinguishable from a counter framework that does not work. So the probe
runs real CDSP work inside the monitoring window and the load-bearing
assertion is that COMMITTED_PKT_ANY moved -- the one event that MUST advance
if the DSP executed anything at all.

WHAT A PASS AND A FAIL EACH MEAN. Pass: the PMU is readable from an unsigned
PD on this device, and stage 4 can collect cache/DRAM/HVX/coproc counters.
Fail: it is not, and a silicon-only experiment is limited to PCYCLE, QTIMER
and their ratio -- which is a real answer and must be reported as one, not
retried until it passes.
"""
import re

from .utils import push, sh, write_qdc_log

DEV = "/data/local/tmp/pmuprobe"

MON = "itrace_monitor"
MON_LIBS = ("libitrace.so", "libc++_shared.so")
RUN = "hexlib_run"
SKEL_SO = "libhexlib_iface_skel.so"

# Four events, one per question, all inside itrace's 8 slots so nothing is
# rotated out and a zero cannot be blamed on time-slicing:
#   COMMITTED_PKT_ANY   did the DSP execute at all      (the control)
#   DCACHE_DEMAND_MISS  L1 data
#   L2_DU_READ_MISS     L2 data
#   AXI_READ_REQUEST    traffic that left the DSP for DDR
EVENTS = (
    "ITRACE_DSP_EVENT_PMU_COMMITTED_PKT_ANY",
    "ITRACE_DSP_EVENT_PMU_DCACHE_DEMAND_MISS",
    "ITRACE_DSP_EVENT_PMU_L2_DU_READ_MISS",
    "ITRACE_DSP_EVENT_PMU_AXI_READ_REQUEST",
)

_CONTROL_EVENT = "COMMITTED_PKT_ANY"

# Monitor for 12 s; start the workload 3 s in so the window brackets it on
# both sides. `p 1000` samples every 1 ms, `n 8` asks for all 8 slots.
_WINDOW_S = 12
_WORKLOAD_DELAY_S = 3


def _env():
    """LD_LIBRARY_PATH for itrace's own shared objects, ADSP_LIBRARY_PATH so
    FastRPC finds the skel. Both point at DEV because everything was pushed
    flat into it."""
    return "LD_LIBRARY_PATH=" + DEV + " ADSP_LIBRARY_PATH=" + DEV


def test_probe_binaries_land_and_are_executable():
    """Push the vendor monitor, its libs, and the known-good CDSP workload.

    Asserted PRESENT rather than "not absent": an exec-format error on the
    runner vs the device is the failure `utils.push` exists to prevent, and it
    only shows up as a file that is there but will not run."""
    sh("mkdir -p " + DEV)
    for name in (MON,) + MON_LIBS + (RUN, SKEL_SO):
        push(name, DEV)
    sh("chmod 755 " + DEV + "/" + MON + " " + DEV + "/" + RUN)
    out = sh("ls -l " + DEV)
    write_qdc_log("pmuprobe_ls.log", out)
    for name in (MON,) + MON_LIBS + (RUN, SKEL_SO):
        assert name in out, name + " did not land in " + DEV + ":\n" + out


def test_itrace_monitor_runs_at_all():
    """Start and immediately quit the monitor.

    Separated from the measurement so a missing `libc++_shared.so`, a wrong
    machine type, or a linker error is diagnosed as itself rather than
    reported as "the PMU is unreadable"."""
    out = sh("cd " + DEV + " && " + _env() + " ./" + MON + " </dev/null; echo RC=$?")
    write_qdc_log("pmuprobe_monitor_start.log", out)
    assert "RC=" in out, "no exit status from " + MON + " at all:\n" + out
    assert ("itrace" in out or "Options" in out or "RC=0" in out), (
        MON + " produced nothing recognisable -- it may not have executed:\n" + out
    )


def test_pmu_counters_are_readable_from_an_unsigned_pd():
    """THE PROBE. Monitor four CDSP PMU events across a real CDSP workload.

    The monitor runs in the FOREGROUND (so the shell waits for it) and the
    workload in the background, delayed, so the sampling window brackets the
    work. `printf` feeds the monitor's REPL: slots, events, period, start,
    then -- after the window -- end and quit."""
    events = ",".join(EVENTS)

    inner = (
        "( sleep " + str(_WORKLOAD_DELAY_S) + "; ./" + RUN + " --self-test "
        ">selftest.log 2>&1; echo SELFTEST_RC=$? >>selftest.log ) & "
        '( printf "n 8\\nd CDSP,' + events + '\\np 1000\\ns\\n"; '
        "sleep " + str(_WINDOW_S) + '; printf "e\\nq\\n" ) | ./' + MON
    )
    cmd = ("cd " + DEV + " && " + _env() + " sh -c " + repr(inner)
           + " 2>&1; echo RC=$?")
    out = sh(cmd)
    write_qdc_log("pmuprobe_monitor.log", out)

    selftest = sh("cat " + DEV + "/selftest.log 2>&1; echo RC=$?")
    write_qdc_log("pmuprobe_selftest.log", selftest)

    # itrace writes its reports under `itrace_results/` beside the exe
    # (README: default output `@itrace_results/itrace_monitor`). Collect the
    # whole tree -- which files appear is itself evidence.
    tree = sh("ls -lR " + DEV + "/itrace_results 2>&1; echo RC=$?")
    write_qdc_log("pmuprobe_results_tree.log", tree)

    csvs = sh(
        "for f in " + DEV + "/itrace_results/*.csv " + DEV
        + '/itrace_results/*/*.csv; do [ -f "$f" ] && echo "===== $f" '
        '&& cat "$f"; done 2>&1; echo RC=$?'
    )
    write_qdc_log("pmuprobe_csv.log", csvs)

    # Everything above is evidence and is logged unconditionally, because a
    # negative result is a real finding that must survive the assertions below.
    combined = out + "\n" + tree + "\n" + csvs

    assert "SELFTEST_RC=0" in selftest, (
        "the CDSP workload did not succeed inside the monitoring window, so a "
        "zero counter proves nothing about the PMU. hexlib_run --self-test "
        "output:\n" + selftest
    )

    assert _CONTROL_EVENT in combined, (
        _CONTROL_EVENT + " never appears in itrace's output at all -- the "
        "events were not registered, so this device does not expose the PMU "
        "to an unsigned PD by this route.\nmonitor:\n" + out + "\ntree:\n" + tree
    )

    counts = [int(m) for m in
              re.findall(_CONTROL_EVENT + r"\D{0,40}?(\d+)", combined)]
    assert any(c > 0 for c in counts), (
        _CONTROL_EVENT + " was registered but every sample read 0, while the "
        "workload demonstrably ran. That is a counter that is present and not "
        "advancing.\nvalues seen: " + repr(counts) + "\ncsv:\n" + csvs
    )
