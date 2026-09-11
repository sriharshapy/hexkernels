# hexlib/device/qdc/session.py
"""Create, drive and RELEASE a QDC INTERACTIVE session over SSH.

WHY THIS EXISTS ALONGSIDE `job.py`. `job.py` submits a batch Appium job: one
zip, one pytest run, one verdict, and the device is gone. Every question costs
a full submit/dispatch/setup/run/upload cycle. An interactive session reserves
the same device and hands back an SSH command, so a probe that needs three
tries at an `itrace_monitor` invocation costs one reservation instead of three
jobs. `job.py` is the right tool for a repeatable measurement; this is the
right tool for finding out what the measurement should be.

THE ONE THING THAT COSTS MONEY IF YOU GET IT WRONG. A session holds the device
until it is COMPLETED or its timeout expires -- there is no implicit release
when your script exits, crashes, or is killed. `complete()` is therefore not
an optional courtesy; every caller must call it from a `finally`, and
`reserved()` below is a context manager that does exactly that so no caller
has to remember. A session created and abandoned bills its full
`timeout_min`.

STATES, FROM THE SDK'S OWN ENUM (`models/session_state.py`), NOT GUESSED:
SUBMITTED -> DISPATCHED -> SETUP -> RUNNING -> COMPLETED, plus CANCELED and
UNDEFINED. `wait_ready()` waits for RUNNING **and** a non-empty `ssh_configs`,
because RUNNING alone arrives before the SSH endpoint is published and a
caller that connects then gets a connection refused it will blame on itself.

THE SSH IS A TUNNEL, NOT A SHELL, AND THAT IS A CORRECTION. This module first
assumed QDC published a remote shell to exec `adb` in, and provided a
`host_sh()` that appended a command to it. That was wrong. What QDC publishes
is a TEMPLATE for a port forward, verbatim from session 770980:

    ssh -i <PRIVATE_KEY_FILE_PATH> \\
        -L <ADB_PORT>:sa775049.sa.svc.cluster.local:5037 \\
        -N sshtunnel@ssh.qdc.qualcomm.com

`-N` means there is no remote command to append at all, and the forwarded
target is port **5037** -- the ADB SERVER port. So the usable shape is: hold
the forward open (`tunnel()`), then run a LOCAL `adb -P <local_port>` which
talks to the REMOTE adb daemon (`adb()`, `adb_shell()`). One happy consequence:
`adb push` is the file-transfer path, so no scp or sftp is needed.

The runner/phone distinction `utils.py` records for the batch path still
applies, just one level out: `adb_shell()` runs on the phone, `adb push` moves
files to it, and everything else is local.

CREDENTIALS ARE `job.py`'S, DELIBERATELY NOT RE-IMPLEMENTED. `_client()`,
the API key resolution (`QDC_API_KEY` or `~/.qdc_api_key`), the base-URL
override and `TARGET_ID` all come from `job.py`. A second copy of credential
handling is a second place to leak one.

qualcomm_device_cloud_sdk IS IMPORTED LAZILY, function-by-function, for the
same reason `job.py` does it: importing this module (and running its offline
tests) must work on a machine where the SDK is not installed.
"""
from __future__ import annotations

import contextlib
import os
import shlex
import socket
import subprocess
import time

from hexkernels.device.qdc import job

# Poll interval while waiting for a session to become usable. Sessions reach
# RUNNING in ~1-2 min on this account (16 prior sessions, all SM8650), so this
# is frequent enough to not waste reserved minutes waiting and slow enough to
# not hammer the API.
POLL_S = 10

# Terminal states: polling past one of these can never succeed.
_DEAD = ("Completed", "Canceled")


class SessionError(Exception):
    """Anything that went wrong creating, polling or releasing a session."""


def _qdc():
    from qualcomm_device_cloud_sdk.api import qdc_api
    return qdc_api


def default_public_key(path: str | None = None) -> str:
    """Read an SSH public key. Explicit `path` wins; otherwise
    `~/.ssh/id_ed25519.pub`.

    The key TEXT is sent (submit_session URL-quotes it itself); no private key
    is read here, ever. A caller wanting a key already registered with QDC
    passes `ssh_public_key_id` to `create()` instead and leaves this alone.
    """
    p = path or os.path.expanduser(os.path.join("~", ".ssh", "id_ed25519.pub"))
    if not os.path.isfile(p):
        raise SessionError(
            f"no SSH public key at {p}. Pass ssh_public_key_path=... or "
            "ssh_public_key_id=... explicitly."
        )
    with open(p) as f:
        text = f.read().strip()
    if not text:
        raise SessionError(f"SSH public key at {p} is empty")
    return text


def create(
    name: str,
    *,
    timeout_min: int,
    ssh_public_key: str | None = None,
    ssh_public_key_id: str | None = None,
    artifacts: list | None = None,
    target_id: int | None = None,
) -> int:
    """Reserve a device and return the session id.

    `timeout_min` IS REQUIRED AND RANGE-CHECKED, matching `job.submit`'s
    discipline: an unreserved default here bills real device time, and the
    SDK's own docstring example passes `timeout=600` (ten hours), which is
    exactly the mistake a default invites.
    """
    if not isinstance(timeout_min, int) or isinstance(timeout_min, bool):
        raise SessionError(f"timeout_min must be an int, got {timeout_min!r}")
    if not job.MIN_TIMEOUT_MIN <= timeout_min <= job.MAX_TIMEOUT_MIN:
        raise SessionError(
            f"timeout_min must be {job.MIN_TIMEOUT_MIN}..{job.MAX_TIMEOUT_MIN}, "
            f"got {timeout_min}"
        )
    if ssh_public_key is None and ssh_public_key_id is None:
        raise SessionError(
            "an interactive session needs ssh_public_key or ssh_public_key_id "
            "-- without one QDC publishes no SSH endpoint and the session can "
            "only be driven from the web console."
        )

    q = _qdc()
    sid = q.submit_session(
        public_api_client=job._client(),
        target_id=target_id if target_id is not None else job.TARGET_ID,
        session_name=name,
        timeout=timeout_min,
        session_artifacts=artifacts,
        ssh_public_key=ssh_public_key,
        ssh_public_key_id=ssh_public_key_id,
    )
    if not sid:
        raise SessionError("submit_session returned no session id")
    return int(sid)


def describe(session_id: int) -> dict:
    """The session's own record as a plain dict (state, result, sshConfigs).

    PARSES `Response.content`, AND THAT IS THE FIX FOR A REAL BUG. This used to
    try `dict(r)` then `r.to_dict()` and fall back to `{"raw": r}`.
    `get_session_by_id` returns `qualcomm_device_cloud_sdk.types.Response`,
    which is NEITHER a dict NOR has `to_dict` -- it carries `.parsed`,
    `.content`, `.status_code`. So every call took the fallback, `sshConfigs`
    read as None forever, `ssh_commands()` always returned [], and
    `wait_ready()` polled its whole 900 s cap against a session that had
    published its endpoint within a minute. `.content` is the raw JSON and is
    parsed here rather than walking `.parsed`'s attrs, because the JSON uses
    the wire names (`sshConfigs`, `sshCommand`) that this module already reads.
    """
    q = _qdc()
    r = q.get_session_by_id(job._client(), session_id)
    if isinstance(r, dict):
        return r
    content = getattr(r, "content", None)
    if content:
        import json as _json
        try:
            d = _json.loads(content.decode() if isinstance(content, bytes) else content)
            if isinstance(d, dict):
                return d
        except ValueError:
            pass
    to_dict = getattr(r, "to_dict", None)
    return to_dict() if callable(to_dict) else {}


def state(session_id: int) -> str:
    """Current SessionState as a string, or "" if the API did not say."""
    q = _qdc()
    try:
        s = q.get_session_status(job._client(), session_id)
    except Exception:  # noqa: BLE001 -- a status hiccup must not kill a poll loop
        return ""
    return str(s) if s else ""


def ssh_commands(session_id: int) -> list[str]:
    """Every `ssh_command` / `qualnet_ssh_command` QDC published, in order.

    Returned as the raw strings QDC produced rather than parsed into
    host/port/user: the command is what QDC documents and what a human pastes,
    so keeping it verbatim means this function cannot mis-parse a form it has
    never seen.
    """
    d = describe(session_id)
    cfgs = d.get("sshConfigs") or d.get("ssh_configs") or []
    out: list[str] = []
    for c in cfgs:
        if not isinstance(c, dict):
            continue
        for key in ("ssh_command", "sshCommand",
                    "qualnet_ssh_command", "qualnetSshCommand"):
            v = c.get(key)
            if isinstance(v, str) and v.strip():
                out.append(v.strip())
    return out


def wait_ready(session_id: int, cap_s: int = 900) -> list[str]:
    """Block until the session is RUNNING *and* has published an SSH command.

    Returns the SSH commands. Raises SessionError on a terminal state or on
    running out of `cap_s` -- never returns an empty list, so a caller cannot
    mistake "not ready yet" for "ready with no endpoint".
    """
    deadline = time.time() + cap_s
    last = ""
    while time.time() < deadline:
        st = state(session_id)
        if st:
            last = st
        if any(d in st for d in _DEAD):
            raise SessionError(
                f"session {session_id} reached terminal state {st!r} before "
                "publishing an SSH endpoint"
            )
        cmds = ssh_commands(session_id)
        if cmds and "Running" in st:
            return cmds
        time.sleep(POLL_S)
    raise SessionError(
        f"session {session_id} did not become usable within {cap_s}s "
        f"(last state {last!r})"
    )


def complete(session_id: int) -> bool:
    """Release the device. True if QDC accepted the completion.

    NEVER RAISES. It is called from `finally` blocks whose whole purpose is to
    stop billing; an exception here would mask the original error AND leave the
    device held. A False return with the device still reserved is the caller's
    to notice, and `reserved()` prints it.
    """
    try:
        _qdc().complete_session(job._client(), session_id)
        return True
    except Exception:  # noqa: BLE001 -- see docstring
        return False


@contextlib.contextmanager
def reserved(name: str, *, timeout_min: int, **kw):
    """Reserve a device for the block and release it no matter how we leave.

        with session.reserved("pmu probe", timeout_min=15) as (sid, cmds):
            ...

    THE `finally` IS THE POINT. Every early return, assertion, exception and
    KeyboardInterrupt inside the block still releases the device. Without it
    an abandoned session bills its whole `timeout_min`.
    """
    sid = create(name, timeout_min=timeout_min, **kw)
    print(f"session {sid} created (timeout {timeout_min} min)")
    try:
        cmds = wait_ready(sid)
        print(f"session {sid} is RUNNING with {len(cmds)} ssh endpoint(s)")
        yield sid, cmds
    finally:
        ok = complete(sid)
        print(f"session {sid} released: {ok}"
              if ok else
              f"WARNING: session {sid} may still be RESERVED and BILLING -- "
              f"complete it by hand")


# --- driving the reserved device -----------------------------------------

# Options injected into every ssh invocation. The bastion is a fresh host on
# every session, so a known_hosts entry would be a prompt on a non-interactive
# pipe -- which hangs rather than fails, and hangs while the device bills.
_SSH_OPTS = [
    "-o", "StrictHostKeyChecking=no",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
    "-o", "BatchMode=yes",
    "-o", "ConnectTimeout=30",
]


def _ssh_base(ssh_command: str, identity: str | None = None) -> list[str]:
    """Split QDC's published SSH command into argv and inject our options.

    QDC publishes the command WITHOUT an `-i`, because it assumes the key is
    the caller's default identity. It usually is not: QDC generates the
    keypair and hands back a downloaded `.pem`, so the identity has to be
    named explicitly. The `-i` and the options go directly after `ssh` --
    before the destination -- because ssh stops parsing options at the first
    non-option argument, and appending them would make them part of the
    REMOTE command instead.

    Splitting with shlex rather than str.split keeps a quoted path intact.
    """
    argv = shlex.split(ssh_command)
    if not argv:
        raise SessionError("empty ssh command from QDC")
    head, rest = argv[0], argv[1:]
    opts = list(_SSH_OPTS)
    if identity:
        opts += ["-i", identity]
    return [head] + opts + rest


def render_ssh_command(template: str, key_path: str, local_port: int) -> list[str]:
    """Fill QDC's published template and return it as argv.

    QDC PUBLISHES A TEMPLATE, NOT A COMMAND, and it is a PORT FORWARD rather
    than a remote shell. Observed verbatim on session 770980:

        ssh -i <PRIVATE_KEY_FILE_PATH> \\
            -L <ADB_PORT>:sa775049.sa.svc.cluster.local:5037 \\
            -N sshtunnel@ssh.qdc.qualcomm.com

    Three things follow from that shape and none of them are optional:

      * `-N` means NO REMOTE COMMAND. There is no shell on the far side to run
        `adb` in; appending a command to this is meaningless. An earlier
        version of this module did exactly that.
      * The forwarded target port is **5037**, the ADB SERVER port. So the
        thing on the other end is an adb *daemon*, and the way to use it is a
        LOCAL `adb -P <local_port>` talking to a REMOTE server. That also
        means `adb push` is the file-transfer mechanism and no scp/sftp is
        needed.
      * The placeholders are literal `<PRIVATE_KEY_FILE_PATH>` and
        `<ADB_PORT>` and must be substituted, not passed through.

    SPLIT FIRST, SUBSTITUTE SECOND, AND THAT ORDER IS THE FIX FOR A REAL BUG.
    This used to substitute into the template string and then `shlex.split()`
    the result. `shlex` in POSIX mode treats backslash as an escape, so a
    Windows key path went in as
    `C:\\Users\\user\\AppData\\...` and came out as `C:UsersuserAppData...`
    -- ssh then reported "Identity file ... not accessible: No such file or
    directory" followed by "Permission denied (publickey)", which reads like a
    key problem and is actually a quoting one. Session 771027 died on it.

    The template itself contains no backslashes and its placeholders are whole
    tokens, so splitting it first is safe; the substituted values then never
    pass through shlex at all. `posix=False` was the other option and is worse
    -- it would preserve quote characters inside the tokens.
    """
    if "<PRIVATE_KEY_FILE_PATH>" not in template or "<ADB_PORT>" not in template:
        raise SessionError(
            "ssh template is missing the placeholders this function "
            f"substitutes; QDC may have changed its shape: {template!r}"
        )
    tokens = shlex.split(template)
    argv = [
        t.replace("<PRIVATE_KEY_FILE_PATH>", key_path)
         .replace("<ADB_PORT>", str(local_port))
        for t in tokens
    ]
    head, rest = argv[0], argv[1:]
    return [head] + list(_SSH_OPTS) + rest


@contextlib.contextmanager
def tunnel(template: str, key_path: str, local_port: int, ready_timeout: int = 60):
    """Hold the SSH port forward open for the duration of the block.

    Yields the local port. The ssh process is terminated on exit -- including
    on exception -- because a leaked `ssh -N` keeps a listener bound and the
    next run picks the stale tunnel instead of a fresh one, which looks exactly
    like a device that stopped responding.

    Readiness is a TCP connect to the local port, not a sleep: the forward is
    usable the moment ssh binds, and a fixed sleep either wastes billed seconds
    or races.
    """
    argv = render_ssh_command(template, key_path, local_port)
    proc = subprocess.Popen(argv, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True)
    try:
        deadline = time.time() + ready_timeout
        while time.time() < deadline:
            if proc.poll() is not None:
                _, err = proc.communicate()
                raise SessionError(
                    f"ssh tunnel exited immediately (rc {proc.returncode}): "
                    f"{(err or '').strip()[:400]}"
                )
            with socket.socket() as s:
                s.settimeout(2)
                if s.connect_ex(("127.0.0.1", local_port)) == 0:
                    break
            time.sleep(1)
        else:
            raise SessionError(
                f"ssh tunnel did not begin listening on 127.0.0.1:{local_port} "
                f"within {ready_timeout}s"
            )
        yield local_port
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except Exception:  # noqa: BLE001 -- a stuck ssh must not block release
            proc.kill()


def free_port() -> int:
    """An unused local TCP port, chosen by the OS rather than guessed."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def adb(port: int, *args: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run a LOCAL `adb` against the REMOTE adb server on `port`.

    `-P` (server port), not `-s` (serial): the tunnel points at an adb server,
    so this process speaks to that server and inherits whatever device it has.
    """
    return subprocess.run(["adb", "-P", str(port), *args],
                          capture_output=True, text=True, timeout=timeout)


def adb_shell(port: int, cmd: str, timeout: int = 300) -> subprocess.CompletedProcess:
    """Run `cmd` ON THE DEVICE through the tunnelled adb server.

    Same runner/phone distinction `utils.sh()` records for the batch path: the
    command runs on the phone, everything around it does not."""
    return adb(port, "shell", cmd, timeout=timeout)


def public_key_from_pem(pem_path: str) -> str:
    """The public half of a QDC-issued private key, via `ssh-keygen -y`.

    QDC GENERATES THE KEYPAIR AND REGISTERS THE PUBLIC HALF ITSELF, then hands
    back only the private `.pem`. `create()` still has to send a public key,
    and sending one QDC has never seen fails with a 400 whose message --
    "The provided SSH public key could not be found or has not been created
    for user" -- reads like a lookup miss rather than a format complaint,
    because it is one. Deriving the public half from the issued private key is
    what makes the two agree.
    """
    if not os.path.isfile(pem_path):
        raise SessionError(f"no private key at {pem_path}")
    r = subprocess.run(["ssh-keygen", "-y", "-f", pem_path],
                       capture_output=True, text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise SessionError(
            f"ssh-keygen could not read a public key out of {pem_path}: "
            f"{(r.stderr or '').strip()}"
        )
    return r.stdout.strip()


def protected_key_copy(pem_path: str, dest_dir: str) -> str:
    """A copy of `pem_path` that OpenSSH will actually use.

    A key readable by anyone is REFUSED by ssh ("UNPROTECTED PRIVATE KEY
    FILE"), and the QDC download lands as -rw-r--r--. On Windows OpenSSH
    honours ACLs rather than POSIX bits, so `chmod` alone is not enough:
    `icacls` re-grants the current user only and strips inheritance. Both are
    attempted; failure to lock down is raised rather than ignored, because the
    alternative is an ssh that refuses the key inside a billing session.

    THIS FUNCTION MUST CLEAN UP AFTER ITSELF, AND THAT IS A FIX FOR A REAL
    BUG. It previously opened `dest` for writing directly. That works exactly
    once: the `icacls /grant:r <user>:R` below leaves the copy READ-ONLY, so the
    NEXT run raises `PermissionError` on its own previous output -- which is
    what happened, and which failed a run before it had reserved anything. The
    stale copy is therefore made writable and removed first. Read-only is the
    correct end state (ssh only needs to read it), so the fix is to delete
    rather than to loosen the grant.
    """
    os.makedirs(dest_dir, exist_ok=True)
    dest = os.path.join(dest_dir, os.path.basename(pem_path))

    if os.path.exists(dest):
        if os.name == "nt":
            user = os.environ.get("USERNAME") or ""
            if user:
                # Full control, briefly, so the delete below is permitted --
                # a file granted only :R cannot be replaced.
                subprocess.run(["icacls", dest, "/grant", f"{user}:F"],
                               capture_output=True, text=True)
        try:
            os.chmod(dest, 0o600)
        except OSError:
            pass
        os.remove(dest)

    with open(pem_path, "rb") as src, open(dest, "wb") as dst:
        dst.write(src.read())
    os.chmod(dest, 0o600)
    if os.name == "nt":
        user = os.environ.get("USERNAME") or ""
        subprocess.run(["icacls", dest, "/inheritance:r"],
                       capture_output=True, text=True)
        if user:
            subprocess.run(["icacls", dest, "/grant:r", f"{user}:R"],
                           capture_output=True, text=True)
    return dest
