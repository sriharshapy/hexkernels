# Running on real silicon: Qualcomm Device Cloud

`hexkernels/device/qdc/` runs kernels on a real Snapdragon rather than the simulator.
Three facts about it are measured, not assumed, and contradicting them costs either
money or a false result.

## The SSH is a tunnel, not a shell

This was a correction. The module first assumed QDC published a remote shell to `exec`
`adb` in. It does not. What QDC publishes is a template for a **port forward**:

```
ssh -i <PRIVATE_KEY_FILE_PATH> \
    -L <ADB_PORT>:<host>:5037 \
    -N sshtunnel@ssh.qdc.qualcomm.com
```

`-N` means there is no remote command to append at all, and the forwarded target is
port **5037** — the adb *server* port. So the usable shape is: hold the forward open
(`tunnel()`), then run a **local** `adb -P <local_port>` that talks to the remote adb
daemon. One happy consequence: `adb push` is the file-transfer path, so no scp or sftp
is needed.

The key must be QDC-issued. Your own SSH key will not authenticate.

## An abandoned session bills its full timeout

A session holds the device until it is COMPLETED or its timeout expires. **There is no
implicit release** when your script exits, crashes, or is killed.

`complete()` is therefore not a courtesy. Every caller must call it from a `finally` —
`session.reserved()` is a context manager that does exactly that, so no caller has to
remember.

Session states, from the SDK's own enum rather than guessed:
`SUBMITTED → DISPATCHED → SETUP → RUNNING → COMPLETED`, plus `CANCELED` and
`UNDEFINED`. `wait_ready()` waits for RUNNING **and** a non-empty `ssh_configs`,
because RUNNING alone arrives before the SSH endpoint is published — and a caller that
connects then gets a connection-refused it will blame on itself.

## Interactive vs. batch

| | `job.py` | `session.py` |
|---|---|---|
| shape | one zip, one pytest run, one verdict | reserves a device, hands back an SSH tunnel |
| cost of a question | a full submit/dispatch/setup/run/upload cycle | one reservation, many tries |
| use it for | a repeatable measurement | finding out what the measurement should be |

## Completion is not a verdict

**Never poll `get_job_status` or `get_jobs_list` for completion.** On this account
`get_job_status` returns `state=None`, and `get_jobs_list` lagged more than 30 minutes
on both jobs observed. Completion is detected by the *appearance* of
`TestLogs/results.xml` among a job's log files.

But that is all `wait()` does. It does not open the file, so it cannot distinguish a
real report from a zero-byte placeholder. **`wait() -> True` is a precondition for a
verdict, never the verdict.**

The verdict is `verdict.py`, and it exists because of one observed failure:

> **A job that ran zero tests reported passing.**

So `check_results()` refuses a pass unless the report parses, is unambiguous, reports
`tests > 0`, has no failures, no errors and **no skips**, the logs carry the
measurement lines a genuine run prints, and `cycles_total=` carries a value **greater
than zero**.

That last check is the one worth dwelling on. For a while the check was only
`"cycles_total=" in logs` — and `cycles_total=0` satisfies that. A run in which the DSP
measured literally nothing printed "measurement lines present" and exited 0. Zero is
not a pedantic edge case here: the DSP reads PCYCLE inside a user-mode unsigned PD
where `SYSCFG.PCYCLEEN` cannot be set, so `cycles_total=0` is *precisely* what the
first silicon job prints if the counter never advances. It was the single most
important thing that job could have told us, and the check used to swallow it.

A skip is refused separately from a failure, not folded into it. Every test on the
device path is unconditional, so a skip means pytest collected a test and never ran
it — the device was unreachable, or collection half-failed. "5 skipped" and "5 failed"
call for completely different next actions.

```python
from hexkernels.device.qdc.verdict import check_results

ok, message = check_results(fetched_log_paths, job_id=12345)
```

## Credentials

The API key resolves from `QDC_API_KEY` or `~/.qdc_api_key`, in `job.py`.
`session.py` deliberately does not re-implement this: a second copy of credential
handling is a second place to leak one. Nothing is embedded in this repo.

## What stayed behind

The budget guard and submit CLI (`_qdc_budget_guard`, `_qdc_submit`) live in the
`hexlib` project and are not vendored here — they are tied to that project's runtime
and its own kernel entry points. `verdict.py` is vendored in full because it is the
part that keeps a false pass off the money path.
