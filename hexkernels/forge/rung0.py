"""Rung 0 of the ladder: the bare, one-shot ask, on the 128-task eval core.

    # 1. generate (network only, no toolchain)
    python -m hexkernels.forge.rung0 generate --model gpt-5.6-luna --seeds 5
    # 2. pre-trace what grading needs, so grading never spends its start-up on it
    python -m hexkernels.forge.rung0 facts
    # 3. grade -- ONE simulator, serial. T0,T1 first: a T2 attempt costs ~4 min
    python -m hexkernels.forge.rung0 grade --model gpt-5.6-luna --tier T0,T1
    python -m hexkernels.forge.rung0 grade --model gpt-5.6-luna
    # 4. report
    python -m hexkernels.forge.rung0 report --model gpt-5.6-luna

WHY GENERATE AND GRADE ARE SEPARATE COMMANDS, not one pipeline. They are bound by
different resources and fail in different ways: generation is 640 HTTPS calls that
cost money and cannot be repeated for free, grading is 640 simulator runs that cost
only wall-clock and can be repeated forever. Fusing them would mean a simulator
timeout in task 3 blocks the API call for task 4, and a crash halfway through would
leave the expensive half half-done. Split, the money is spent once and the
CPU-bound half is restartable and safe to run in a plain terminal where nothing
reaps it at 70 seconds.

GRADING IS SERIAL: ONE `hexagon-sim` AT A TIME. Standing instruction from the user,
2026-08-17 -- never parallelise the simulator on Windows. `--jobs` above 1 is
refused, not clamped. The consequence to plan around is that a T2 attempt costs ~4
minutes of wall-clock against a few seconds for a T0, so `--tier T0,T1` exists to
get a reportable baseline out before the big kernels are queued.

RESUMABILITY IS PER ATTEMPT, for the same reason `run_batch` persists per batch.
Every attempt writes its own `attempt.json` the moment it completes, so a killed
run loses only what was in flight; re-running skips anything already recorded
unless `--redo` is passed. `grade` likewise skips an attempt that already carries a
verdict.

THE ATTEMPT RECORD IS THE DURABLE INTERFACE -- rungs 1-3 and the aggregator all
read it, so it is versioned (`schema_version`) and its fields are documented in
`_attempt_skeleton`. Two of its choices are worth stating here because they are
easy to get wrong and expensive to fix afterwards:

  * A task is keyed by `(batch, index)`, never by kernel name. Names collide across
    tiers by construction -- see `hexkernels.forge.evalset`.
  * `eval_set` is stamped on every record. PLAN.md section 3 budgets silicon on 64
    tasks while `eval_core.json` holds 128, so two values of n are in play and the
    aggregator must refuse to mix them rather than average across them silently.

WHAT COUNTS AS A RESULT, and what is merely recorded. `correct` is a correctness
verdict from the simulator against PyTorch goldens. `genuine` is the number the
thesis rests on: correct AND at least one mechanism the task's SIZE entitles it to
actually fires in the disassembly. A correct kernel that never touches the
accelerator is a task FAILURE at rung 0 and the record says so explicitly rather
than leaving a reader to compute it.
"""
import argparse
import concurrent.futures
import datetime
import difflib
import hashlib
import json
import os
import pathlib
import random
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request

from hexkernels.core.toolchain import DEFAULT_SDK_ROOT
from hexkernels.forge import evalset, prompt as _prompt

REPO = pathlib.Path(__file__).resolve().parents[1]
RUNS = REPO / "runs"
SCHEMA_VERSION = 1
EVAL_SET = "core128"

#: The flag in a `verify()` verdict that proves each entitled mechanism was USED.
#:
#: `hvx` maps to `hvx_compute` and not to `hvx`, and that is the whole anti-cheat
#: thesis in one line: a kernel that merely vector-LOADS and vector-STORES trips
#: `hvx` while doing every arithmetic operation in scalar registers. Crediting that
#: as "used the accelerator" is exactly the false positive the instrument exists to
#: remove (see `hexkernels.anticheat.anticheat._disasm_has_hvx_compute`).
MECHANISM_FLAG = {"hvx": "hvx_compute", "hmx": "hmx", "dma": "dma",
                  "vtcm": "vtcm", "l2fetch": "l2fetch"}


#: Serialises the rename half of `_write_atomic`; see that function for why.
_REPLACE_LOCK = threading.Lock()
_REPLACE_TRIES = 5


def _now() -> str:
    return datetime.datetime.now(datetime.timezone.utc).isoformat(
        timespec="seconds")


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _write_atomic(path, text: str) -> None:
    """Write via a temp file and replace, so a kill cannot leave a half record.

    An attempt record truncated mid-write reads as valid JSON exactly never, but a
    truncated `candidate.cpp` reads as valid C++ that fails to compile -- which
    would be recorded against the model.

    CONCURRENT WRITERS TO ONE PATH NEED BOTH HALVES OF THIS, and Windows taught
    each half separately:

      1. A UNIQUE TEMP NAME. Five seeds of one task run concurrently and all five
         write that task's prompt file. A shared `<path>.tmp` put two threads in
         the same file and `os.replace` died with `WinError 32: being used by
         another process`, which aborted the whole pool -- it killed a 640-attempt
         run after 6 attempts.
      2. A SERIALISED REPLACE. Unique temps alone were not enough: two threads
         replacing the SAME target at once then failed with `WinError 5: access is
         denied`, because Windows will not swap a file another handle is mid-swap
         on. POSIX `rename` has no such problem, so this is invisible on Linux and
         fatal here.

    The lock is global rather than per path: these writes are a few KB and the
    pool's real work is a network call or a simulator run, so there is nothing to
    gain from finer granularity and a keyed lock table would be one more thing to
    get wrong. The retry covers contention this process cannot see -- a second
    grading pass, or an antivirus scanner holding the file for a moment.
    """
    path = str(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.{os.getpid()}.{threading.get_ident()}.tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)
    with _REPLACE_LOCK:
        for i in range(_REPLACE_TRIES):
            try:
                os.replace(tmp, path)
                return
            except PermissionError:
                if i + 1 == _REPLACE_TRIES:
                    raise
                time.sleep(0.05 * (i + 1))


def run_dir(model: str, root=RUNS, rung=0) -> pathlib.Path:
    """Per-model run root. The model id is slugified only for the PATH; the
    record always carries the id verbatim.

    `rung` selects the tree, defaulting to 0 so every rung-0 path is byte-identical
    to what it was before rung 1 existed. GRADING, SWEEPING AND REPORTING ARE
    RUNG-AGNOSTIC by construction -- they read `candidate.cpp` and write a verdict,
    and nothing about that depends on which prompt produced the candidate -- so they
    are shared rather than copied. Only GENERATION differs per rung, which is why
    `rung1.py` implements a generator and reuses everything else here.
    """
    slug = model.replace("/", "_").replace(":", "_")
    return pathlib.Path(root) / f"rung{int(rung)}" / slug


def attempt_dir(model, ref, seed_index, root=RUNS, rung=0) -> pathlib.Path:
    return run_dir(model, root, rung) / "attempts" / ref.key / f"seed{seed_index}"


# ---------------------------------------------------------------- the API call

def load_key(key_file=None) -> str:
    """The OpenAI key, from a file if given, else the environment.

    A FILE IS TRIED FIRST AND THIS IS DELIBERATE. Measured on this machine:
    `OPENAI_API_KEY` in the environment is a dead key that returns 401 while
    `.openai_token` in the repo root works. Preferring the env var would have made
    every one of 640 calls fail with an authentication error that reads nothing
    like "your shell has a stale export".
    """
    if key_file:
        return pathlib.Path(key_file).read_text(encoding="utf-8").strip()
    default = REPO / ".openai_token"
    if default.exists():
        tok = default.read_text(encoding="utf-8").strip()
        if tok:
            return tok
    return os.environ.get("OPENAI_API_KEY", "")


def _post(url, body, headers, timeout):
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode())


def complete(model, text, key, *, max_tokens=16384, seed=None, timeout=900,
             attempts=4, temperature=None) -> dict:
    """One chat completion, with the usage accounting an experiment needs.

    Returns `{"text", "usage", "finish_reason", "seed_honored", "error",
    "latency_s", "http_attempts"}`. Never raises for an API-side failure: a
    refused or empty completion is DATA about the model and must land in the
    record rather than abort a 640-call run.

    RETRIES ARE FOR TRANSPORT ONLY -- 429 and 5xx, with jittered backoff. A 400 is
    not retried because it is a statement about the request, with one exception:
    a model that rejects `seed` (or `temperature`) is retried once without it and
    `seed_honored` records that, because dropping the parameter changes what the
    seed index means and a reader must be able to see it.
    """
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {key}"}
    body = {"model": model,
            "messages": [{"role": "user", "content": text}],
            "max_completion_tokens": max_tokens}
    if seed is not None:
        body["seed"] = seed
    if temperature is not None:
        body["temperature"] = temperature
    out = {"text": "", "usage": {}, "finish_reason": None,
           "seed_honored": seed is not None, "error": None, "latency_s": None,
           "http_attempts": 0}
    started = time.time()
    for i in range(attempts):
        out["http_attempts"] = i + 1
        try:
            data = _post(url, body, headers, timeout)
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode(errors="replace")[:600]
            if exc.code == 400 and ("seed" in detail or "temperature" in detail):
                # Drop the unsupported knob and say so, rather than failing 640
                # calls over a parameter the model simply does not take.
                dropped = False
                for knob in ("seed", "temperature"):
                    if knob in detail and knob in body:
                        body.pop(knob)
                        dropped = True
                        if knob == "seed":
                            out["seed_honored"] = False
                if dropped:
                    continue
            if exc.code in (429, 500, 502, 503, 504) and i + 1 < attempts:
                time.sleep(min(60, 2 ** i * 5) * (0.5 + random.random()))
                continue
            out["error"] = f"HTTP {exc.code}: {detail}"
            break
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            if i + 1 < attempts:
                time.sleep(min(60, 2 ** i * 5) * (0.5 + random.random()))
                continue
            out["error"] = f"{type(exc).__name__}: {exc}"
            break
        choice = (data.get("choices") or [{}])[0]
        out["text"] = (choice.get("message") or {}).get("content") or ""
        out["finish_reason"] = choice.get("finish_reason")
        out["usage"] = data.get("usage") or {}
        out["response_model"] = data.get("model")
        if not out["text"]:
            # A reasoning model that spends its whole budget thinking returns an
            # empty string with finish_reason "length". That is a real rung-0
            # failure mode and is recorded, not retried into a different one.
            out["error"] = f"empty completion (finish_reason={out['finish_reason']})"
        break
    out["latency_s"] = round(time.time() - started, 2)
    return out


def strip_fence(text: str) -> str:
    """Delegates to `model_client.strip_fence` -- one fence rule for the project."""
    from hexkernels.forge.model_client import strip_fence as _sf
    return _sf(text)


def cost_usd(usage, price_in, price_out, price_cached=None):
    """Dollar cost from token counts, or None when no price was supplied.

    NOT a built-in price table. Prices change and a stale constant compiled into
    the record would be a fabricated number that looks measured; the rate used is
    recorded beside the cost so any total can be recomputed or corrected.

    THREE RATES, not two, and the third was a real defect. Cached input is billed
    at a fraction of the uncached rate, and rung 1 charged 385,150 cache hits
    (49.6% of its input) at full price. Rung 0 was unaffected -- every rung-0
    prompt sat under the ~1024-token floor that triggers caching at all -- which is
    exactly why the bug survived a whole rung.

    An UNSUPPLIED `price_cached` bills cached tokens at the full rate rather than
    discounting them silently: an unknown rate is unknown, not free. The share is
    recorded either way (`prompt_tokens_details.cached_tokens` comes straight from
    the provider), so a total can be recomputed once the rate is known.
    """
    if price_in is None or price_out is None or not usage:
        return None
    pin = usage.get("prompt_tokens") or 0
    pout = usage.get("completion_tokens") or 0
    cached = ((usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0)
    cached = max(0, min(cached, pin))
    uncached = pin - cached
    rate_cached = price_in if price_cached is None else price_cached
    return round(uncached / 1e6 * price_in
                 + cached / 1e6 * rate_cached
                 + pout / 1e6 * price_out, 6)


def reprice(usage: dict) -> dict:
    """Recompute `usage["cost_usd"]` from the CURRENT token counts, in place.

    A cost computed once is stale as soon as the counts grow. `generate_one` prices
    the first call; a retry wave then accumulates that attempt's later turns into the
    same `usage` dict, and without this the record reports N turns of tokens against
    one turn of dollars -- while `aggregate` sums this very field as the headline.

    The rates come from `usage["price_per_1m"]`, which `generate_one` recorded
    alongside the cost precisely so any total can be recomputed later. No rates
    recorded means the attempt was never priced, and it stays unpriced: inventing a
    rate here would be the fabricated-number failure `cost_usd`'s own docstring
    refuses.
    """
    rates = usage.get("price_per_1m") or {}
    price_in = rates.get("in")
    price_out = rates.get("out")
    price_cached = rates.get("cached_in")
    if price_in is None or price_out is None:
        return usage
    usage["cost_usd"] = cost_usd(usage, price_in, price_out, price_cached)
    return usage


# ------------------------------------------------------------------- the record

def _attempt_skeleton(ref, model, seed_index, rung=0):
    """One attempt's record. Field order is documentation; keys are the interface.

    `grade` is present but empty from the start, so a consumer never has to
    distinguish "not graded" from "field absent" -- the same discipline
    `verify()` follows for its verdict.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "eval_set": EVAL_SET,
        "rung": int(rung),
        "seed_index": seed_index,
        "task": {"key": ref.key, "task_id": ref.task_id, "name": ref.name,
                 "batch": ref.batch, "index": ref.index, "tier": ref.tier,
                 "dtype": ref.dtype, "shape": list(ref.shape),
                 "entitled_mechanisms": list(ref.mechanisms)},
        "model": {"provider": "openai", "requested": model},
        "prompt": {},
        "response": {},
        "candidate": {},
        "usage": {},
        "turns_used": 1,
        "generation_error": None,
        "generated_at": None,
        "grade": {},
    }


#: What GRADING actually needs from a traced task, and nothing more.
#:
#: WHY THIS CACHE EXISTS. Grading calls `lint` and `lint` wants five facts the
#: source cannot supply: the signature to compare against, the tier, the entitled
#: mechanisms, the element count and the dtype. Obtaining them meant tracing all 65
#: gradable tasks with torch.fx before the first simulator run -- and the harness
#: reaped a grading run DURING that phase, so it graded nothing at all and threw
#: away ~10 minutes. All five are small and JSON-serialisable, so they are cached
#: once and grading thereafter needs no torch, which also removes the segfault
#: hazard class described in `grade_one` from the grading path entirely.
LINT_FACT_KEYS = ("signature", "tier", "mechanisms", "nelem", "dtype")


def _facts_from_art(art) -> dict:
    return {k: art.get(k) for k in LINT_FACT_KEYS}


def lint_facts_path(root=RUNS) -> pathlib.Path:
    return pathlib.Path(root) / "lint_facts.json"


def load_lint_facts(root=RUNS) -> dict:
    path = lint_facts_path(root)
    if not path.exists():
        return {}
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return {}


def save_lint_facts(facts, root=RUNS) -> None:
    _write_atomic(lint_facts_path(root),
                  json.dumps(facts, indent=1, sort_keys=True) + "\n")


def _task_artifacts(ref):
    """Build stages (a)-(f) for one task: signature, graph, harness, scalar.

    Pure Python and no toolchain, per `run_batch.build`'s contract. Called once per
    TASK and shared across that task's seeds, since the prompt is identical for
    every seed -- what varies is the sample, not the ask.
    """
    from hexkernels.forge import mined, run_batch
    spec = mined.MINED_BATCHES[ref.batch][ref.index]
    art = run_batch.build(spec)
    return art


def first_prompt(rung: int, ref, art) -> str:
    """The FIRST ask for one task at `rung`. One switch, in one place.

    The mislabelling this prevents is the quiet kind: a rung-1 run that generated
    rung-0 prompts would complete, record `rung: 1` on every attempt, and report a
    second rung-0 measurement under a rung-1 heading. Nothing downstream could detect
    it -- the prompt sha is recorded but there is nothing to compare it against.

    Rung 3 is NOT reachable from this driver and raises rather than fall back:
    it adds HexagonGym (profile -> prescription -> repair) and needs silicon, not
    a prompt template. Falling back to a lower rung for an unknown rung would be
    the same silent mislabelling in a different direction.
    """
    if rung == 0:
        return _prompt.build_rung0(ref.name, art["signature"], art["graph"],
                                   ref.entry)
    if rung == 1:
        return _prompt.build_rung1(ref.name, art["signature"], art["graph"],
                                   ref.entry, art["kernel_c"])
    if rung == 2:
        # The whole apparatus: reference, hardware facts, Linalg, schedule,
        # mechanism budget, full contract. `art` already carries every piece --
        # `run_batch.build` computes the plan at stage (c) and the Linalg at stage
        # (b2) -- so nothing here re-derives anything.
        return _prompt.build_rung2(ref.name, art["signature"], art["graph"],
                                   ref.entry, art["kernel_c"], art["plan"],
                                   linalg_ir=art["linalg_ir"])
    raise ValueError(
        f"no first-ask template for rung {rung}: this driver implements rungs 0, "
        "1 and 2. Rung 3 adds HexagonGym (profile -> prescription -> repair) and "
        "needs silicon, not a prompt template.")


def generate_one(ref, art, model, seed_index, key, *, max_tokens, price_in,
                 rung=0,
                 price_out, seed_base, root=RUNS, temperature=None,
                 price_cached=None):
    """Ask the model once for one task and record the attempt. Returns the record."""
    adir = attempt_dir(model, ref, seed_index, root, rung)
    adir.mkdir(parents=True, exist_ok=True)
    rec = _attempt_skeleton(ref, model, seed_index, rung)

    if art["failed_stage"] is not None:
        # The TASK could not be built, which is not the model's failure. Recorded
        # as such so it cannot be counted as an incorrect kernel.
        rec["generation_error"] = (f"task build failed at stage "
                                   f"{art['failed_stage']}: "
                                   f"{art['error'].splitlines()[0]}")
        rec["generated_at"] = _now()
        _write_atomic(adir / "attempt.json", json.dumps(rec, indent=1) + "\n")
        return rec

    text = first_prompt(rung, ref, art)
    # One prompt per TASK, shared by its seeds -- what varies across seeds is the
    # sample, not the ask. Written once: all five seeds run concurrently and would
    # otherwise each rewrite the identical file.
    ppath = run_dir(model, root, rung) / "prompts" / f"{ref.key}.md"
    if not (ppath.exists() and ppath.read_text(encoding="utf-8") == text):
        _write_atomic(ppath, text)
    rec["prompt"] = {"sha256": _sha(text), "bytes": len(text)}

    seed = None if seed_base is None else seed_base + seed_index
    res = complete(model, text, key, max_tokens=max_tokens, seed=seed,
                   temperature=temperature)
    rec["model"].update({"resolved": res.get("response_model"),
                         "max_completion_tokens": max_tokens,
                         "seed": seed, "seed_honored": res["seed_honored"],
                         "temperature": temperature})
    rec["response"] = {"sha256": _sha(res["text"]) if res["text"] else None,
                       "bytes": len(res["text"]),
                       "finish_reason": res["finish_reason"],
                       "http_attempts": res["http_attempts"],
                       "latency_s": res["latency_s"]}
    rec["usage"] = dict(res["usage"])
    rec["usage"]["cost_usd"] = cost_usd(res["usage"], price_in, price_out,
                                       price_cached)
    rec["usage"]["price_per_1m"] = {"in": price_in, "out": price_out,
                                    "cached_in": price_cached}
    rec["generation_error"] = res["error"]
    rec["generated_at"] = _now()

    if res["text"]:
        _write_atomic(adir / "response.txt", res["text"])
        src = strip_fence(res["text"])
        _write_atomic(adir / "candidate.cpp", src)
        cpath = adir / "candidate.cpp"
        try:
            cpath = cpath.relative_to(REPO)
        except ValueError:
            pass
        # POSIX separators in the record: it is read on whatever machine analyses
        # the run, and a Windows backslash path in JSON is both an escaping hazard
        # and unusable there.
        rec["candidate"] = {"sha256": _sha(src), "bytes": len(src),
                            "path": cpath.as_posix(),
                            "fenced": src != res["text"].strip() + "\n"}
    _write_atomic(adir / "attempt.json", json.dumps(rec, indent=1) + "\n")
    return rec


def _existing(adir):
    path = adir / "attempt.json"
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def cmd_generate(args) -> int:
    key = load_key(args.key_file)
    if not key:
        print("no API key: pass --key-file or set OPENAI_API_KEY", file=sys.stderr)
        return 2
    tasks, report = evalset.core_tasks()
    if report["missing"]:
        print(f"REFUSING: {len(report['missing'])} eval-core ids do not resolve",
              file=sys.stderr)
        return 2
    tasks = select_tasks(tasks, args)
    if args.limit:
        tasks = tasks[:args.limit]
    seeds = list(range(1, args.seeds + 1))

    todo = []
    for ref in tasks:
        for s in seeds:
            if not args.redo:
                have = _existing(attempt_dir(args.model, ref, s, args.root,
                                             args.rung))
                # A generation ERROR is worth retrying; a completed attempt is not.
                if have and not have.get("generation_error"):
                    continue
            todo.append((ref, s))
    print(f"{len(tasks)} tasks x {len(seeds)} seeds; {len(todo)} attempts to make")
    if not todo:
        return 0

    # Artifacts are built once per task, in the main thread: `build()` traces with
    # torch, which is not something to call concurrently for no benefit -- the API
    # call is the only slow part and it is what gets the pool.
    arts, t0 = {}, time.time()
    for ref in {r.key: r for r, _s in todo}.values():
        arts[ref.key] = _task_artifacts(ref)
        bad = arts[ref.key]["failed_stage"]
        if bad:
            print(f"  {ref.key:38s} TASK BUILD FAILED at {bad}")
    print(f"built {len(arts)} task artifacts in {time.time() - t0:.0f}s")

    done = {"ok": 0, "err": 0, "cost": 0.0}

    def work(item):
        ref, s = item
        # ONE ATTEMPT MAY NOT TAKE THE RUN DOWN WITH IT. `pool.map` propagates the
        # first exception and abandons everything queued behind it, so an
        # unhandled error in attempt 7 discards the other 633 -- which is exactly
        # what a `WinError 32` on a shared temp file did. A crashed attempt is
        # recorded as a failed attempt and the run continues.
        try:
            return ref, s, generate_one(
                ref, arts[ref.key], args.model, s, key,
                max_tokens=args.max_tokens, price_in=args.price_in,
                price_out=args.price_out, price_cached=args.price_cached,
                seed_base=args.seed_base,
                root=args.root, rung=args.rung, temperature=args.temperature)
        except Exception as exc:                  # noqa: BLE001 - recorded, not raised
            rec = _attempt_skeleton(ref, args.model, s, args.rung)
            rec["generation_error"] = (f"driver error: {type(exc).__name__}: "
                                       f"{exc}")
            rec["generated_at"] = _now()
            adir = attempt_dir(args.model, ref, s, args.root, args.rung)
            adir.mkdir(parents=True, exist_ok=True)
            _write_atomic(adir / "attempt.json", json.dumps(rec, indent=1) + "\n")
            _write_atomic(adir / "driver_error.txt", traceback.format_exc())
            return ref, s, rec

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for ref, s, rec in pool.map(work, todo):
            err = rec.get("generation_error")
            done["err" if err else "ok"] += 1
            done["cost"] += (rec.get("usage", {}).get("cost_usd") or 0)
            n = done["ok"] + done["err"]
            print(f"  [{n}/{len(todo)}] {ref.key:38s} seed{s} "
                  f"{'ERR ' + err[:60] if err else str(rec['candidate'].get('bytes')) + ' B'}",
                  flush=True)
    print(f"\ngenerated {done['ok']} ok, {done['err']} failed"
          + (f", ${done['cost']:.2f}" if done["cost"] else ""))
    return 0


# -------------------------------------------------------------------- grading

def _timeout_skip_reason(error_text) -> str:
    """The one wording for "the budget ran out", shared by the grader and by
    `invalidate --timed-out`.

    Shared deliberately: `aggregate.summarise` buckets ungraded attempts BY THIS
    STRING, so two sites phrasing it differently would split one cause into two
    lines of the summary and make the gap look like two unrelated problems.
    """
    return (f"{error_text or 'timed out'}: the run never reached its comparison, "
            "so this is not a verdict -- raise --timeout and re-grade.")


def grade_one(ref, model, seed_index, *, root=RUNS, rung=0, timeout=900,
              sdk_root=DEFAULT_SDK_ROOT, timing=False, facts=None):
    """Lint, simulate and score one already-generated candidate. Updates its record.

    `facts` MUST be supplied when this is called from a worker thread. Deriving it
    means tracing the module with torch.fx, and doing THAT concurrently SEGFAULTS
    the interpreter: measured, `grade --jobs 3` died with SIGSEGV and, because
    stdout was buffered into a pipe, took its own progress output with it, so the
    run looked like it had done nothing at all. `cmd_grade` resolves every task's
    facts up front (from a cache, or once on the main thread) and the pool then only
    runs the compiler and the simulator, which are subprocesses and genuinely
    parallel.
    """
    from hexkernels.forge import lint as _lint
    from hexkernels.forge import verify as _v
    from hexkernels.gym import intrinsic_repair as _ir
    from hexkernels.gym import metrics as _metrics

    adir = attempt_dir(model, ref, seed_index, root, rung)
    rec = _existing(adir)
    if rec is None:
        return None
    src_path = adir / "candidate.cpp"
    grade = {"graded": False, "reference_verified": False, "skip_reason": "",
             "graded_at": _now()}

    if not src_path.exists():
        grade["skip_reason"] = "no candidate (generation failed)"
        rec["grade"] = grade
        _write_atomic(adir / "attempt.json", json.dumps(rec, indent=1) + "\n")
        return rec

    status = evalset.reference_status(ref)
    grade["reference_verified"] = status["verified"]
    if not status["verified"]:
        # NOT graded rather than graded-as-failed. Without a reference that passed
        # its own harness, a verdict is about the harness (run_batch stage g).
        grade["skip_reason"] = f"reference not usable: {status['reason']}"
        rec["grade"] = grade
        _write_atomic(adir / "attempt.json", json.dumps(rec, indent=1) + "\n")
        return rec

    src = src_path.read_text(encoding="utf-8")
    harness = pathlib.Path(status["harness"]).read_text(encoding="utf-8")
    if facts is None:
        facts = _facts_from_art(_task_artifacts(ref))

    findings = _lint.lint(src, signature=f'extern "C" {facts["signature"]}',
                          tier=facts.get("tier"),
                          mechanisms=facts.get("mechanisms") or (),
                          nelem=facts.get("nelem"), dtype=facts.get("dtype"))
    blocking = _lint.errors(findings)
    grade["lint"] = {"errors": [str(f) for f in blocking],
                     "warnings": [str(f) for f in findings
                                  if f.severity == "warn"]}

    if blocking:
        # The static gate is BEFORE the compiler here exactly as in
        # `model_client`: a kernel refused by lint never reaches the simulator, and
        # spending a simulator run on it teaches nothing.
        verdict = {"compiled": False, "ran": False, "correct": False,
                   "mechanisms": {}, "rd_leak": [], "error_text":
                   _lint.format_findings(blocking), "stdout": ""}
    else:
        verdict = _v.verify(src, harness, name=f"{ref.key}_s{seed_index}",
                            timing=timing, sdk_root=sdk_root, timeout=timeout)

    # AN EXHAUSTED BUDGET IS NOT A WRONG ANSWER.
    #
    # `verify` sets `timed_out` when the simulator (or the compiler) was still
    # running when its budget expired. Nothing was compared against the golden
    # vectors, so there is no verdict to record -- but the verdict dict says
    # `correct: false`, and writing that straight through books a slow kernel as an
    # incorrect one.
    #
    # MEASURED, and it biased the headline. 19 of the 27 rung-0 failures were the
    # 900 s budget running out, all of them in T2/T3, whose median attempt runs ~12
    # min -- so the tasks it hit were the slow tail, and correctness came out 95.3%
    # where the evidence supports 98.5%. The error ran in the direction that
    # flatters the thesis, which is the direction that must never be silent.
    #
    # Same rule as the guard below, and for the same reason: an infrastructure limit
    # must not be able to masquerade as a measurement. Raise `--timeout` and re-grade.
    if verdict.get("timed_out"):
        grade["skip_reason"] = _timeout_skip_reason(verdict.get("error_text"))
        rec["grade"] = grade
        _write_atomic(adir / "attempt.json", json.dumps(rec, indent=1) + "\n")
        return rec

    # A FAILURE WITH NO EVIDENCE IS NOT A VERDICT.
    #
    # `verify` always records something on a real compile failure: the compiler's
    # stderr/stdout, or the string "compile timed out". A verdict that says
    # `compiled: false` with BOTH streams empty therefore did not come from the
    # compiler disagreeing with the kernel -- it came from the compiler never
    # running, or being killed.
    #
    # MEASURED, and it silently corrupted a tier. Killing the grading process (and
    # the simulator under it) mid-pass made every remaining attempt fail instantly
    # with empty output, and 126 of 145 T2 attempts were written as "did not
    # compile" inside one second -- indistinguishable, in the record, from 126
    # models writing bad C++. It even looked like a finding: a compile rate that
    # fell from 100% at T1 to 13% at T2.
    #
    # So it is recorded as UNGRADED with a reason, which puts the attempt back in
    # the queue for the next pass. Refusing to record a verdict here is the whole
    # point: an infrastructure failure must never be able to masquerade as a
    # measurement.
    if (not verdict.get("compiled") and not (verdict.get("error_text") or "")
            and not (verdict.get("stdout") or "") and not blocking):
        grade["skip_reason"] = ("no compiler output: the toolchain did not run "
                                "(killed process, or a broken environment). Not a "
                                "verdict -- will retry.")
        rec["grade"] = grade
        _write_atomic(adir / "attempt.json", json.dumps(rec, indent=1) + "\n")
        return rec

    mech = verdict.get("mechanisms") or {}
    entitled = list(ref.mechanisms)
    fired = [m for m in entitled if mech.get(MECHANISM_FLAG.get(m, m))]
    valid = _ir.valid_intrinsics(sdk_root)
    halluc = sorted(_metrics.hallucinated_intrinsics(src, valid))

    scalar = pathlib.Path(status["dir"]) / "kernel.cpp"
    similarity = None
    if scalar.exists():
        # DIAGNOSTIC, not a verdict. Rung 0 hands over no scalar code, so this
        # measures whether the model independently produced something close to the
        # naive reference -- which is the evidence behind "it wrote scalar code",
        # and the number that answers the objection rung 0 exists to answer.
        similarity = round(difflib.SequenceMatcher(
            None, scalar.read_text(encoding="utf-8"), src).ratio(), 4)

    grade.update({
        "graded": True,
        "compiled": bool(verdict.get("compiled")),
        "ran": bool(verdict.get("ran")),
        "correct": bool(verdict.get("correct")),
        "mechanisms": mech,
        "entitled_mechanisms": entitled,
        "mechanisms_fired": fired,
        "genuine": bool(verdict.get("correct")) and bool(fired),
        "rd_leak": list(verdict.get("rd_leak") or []),
        "insns": verdict.get("insns"),
        "pcycles": verdict.get("pcycles"),
        "sim_timing": bool(timing),
        "hallucinated_intrinsics": halluc,
        "scalar_similarity": similarity,
        "error_text": (verdict.get("error_text") or "")[:4000],
        "stdout": (verdict.get("stdout") or "")[:2000],
        "graded_at": _now(),
    })
    rec["grade"] = grade
    _write_atomic(adir / "attempt.json", json.dumps(rec, indent=1) + "\n")
    return rec


def select_tasks(tasks, args):
    """Apply `--only` and `--tier` uniformly, so every command scopes the same way."""
    only = getattr(args, "only", None)
    if only:
        wanted = set(only.split(","))
        tasks = [t for t in tasks if t.key in wanted or t.task_id in wanted]
    tiers = getattr(args, "tier", None)
    if tiers:
        keep = {t.strip().upper() for t in tiers.split(",")}
        tasks = [t for t in tasks if t.tier in keep]
    return tasks


def sweep_one(ref, model, seed_index, *, root=RUNS, rung=0,
               sdk_root=DEFAULT_SDK_ROOT):
    """Record which mechanisms one candidate USES. No simulator, no correctness.

    Written to `mechanism_scan`, NEVER to `grade`. The two are different claims
    and the record keeps them apart on purpose: `grade` means the kernel was
    executed against golden vectors, and nothing here executes anything.
    """
    from hexkernels.forge import verify as _v
    from hexkernels.gym import intrinsic_repair as _ir
    from hexkernels.gym import metrics as _metrics

    adir = attempt_dir(model, ref, seed_index, root, rung)
    rec = _existing(adir)
    if rec is None:
        return None
    src_path = adir / "candidate.cpp"
    if not src_path.exists():
        rec["mechanism_scan"] = {"scanned": False,
                                 "reason": "no candidate (generation failed)",
                                 "scanned_at": _now()}
        _write_atomic(adir / "attempt.json", json.dumps(rec, indent=1) + "\n")
        return rec

    src = src_path.read_text(encoding="utf-8")
    res = _v.mechanisms_of(src, name=f"{ref.key}_s{seed_index}",
                           sdk_root=sdk_root)
    entitled = list(ref.mechanisms)
    mech = res.get("mechanisms") or {}
    fired = [m for m in entitled if mech.get(MECHANISM_FLAG.get(m, m))]
    valid = _ir.valid_intrinsics(sdk_root)
    rec["mechanism_scan"] = {
        "scanned": bool(res.get("scanned")),
        "compiled": bool(res.get("compiled")),
        "mechanisms": mech,
        "entitled_mechanisms": entitled,
        "mechanisms_fired": fired,
        # THE ONE CONCLUSION A SCAN CAN REACH ON ITS OWN. `genuine` is
        # `correct AND fired`; with no mechanism firing the conjunction is false
        # whatever correctness turns out to be, so `genuine` is settled as False
        # without a simulator. The converse is NOT settled: firing tells you
        # nothing about correctness, and `genuine_possible` says only that this
        # attempt is still in the running.
        "genuine_ruled_out": bool(res.get("scanned")) and not fired,
        "genuine_possible": bool(fired),
        "hallucinated_intrinsics": sorted(
            _metrics.hallucinated_intrinsics(src, valid)),
        "uses_any_q6_intrinsic": "Q6_" in src,
        "error_text": (res.get("error_text") or "")[:2000],
        "scanned_at": _now(),
    }
    _write_atomic(adir / "attempt.json", json.dumps(rec, indent=1) + "\n")
    return rec


def sweep_turns_one(ref, model, seed_index, *, root=RUNS, rung=0,
                    sdk_root=DEFAULT_SDK_ROOT):
    """Mechanism flags for EVERY turn of one attempt, from the saved candidates.

    Written to `mechanism_scan_turns`, never to `mechanism_scan`: the latter is the
    rung's headline field and the aggregator reads it, so this has to be additive or
    a second run of the command would quietly redefine the result.

    WHY THIS EXISTS. Rung 1's open item -- count mechanism engagement on the FINAL
    candidate or on ANY turn -- had a 0-versus-1 answer there, and reporting only
    the final count would have erased the one attempt in 1,280 that reached for the
    hardware. At rung 2 waves rewrite candidates in volume, so both numbers are
    needed, and turn 1 is additionally the only measurement taken before
    `prompt.retry_facts` names an intrinsic.

    POST-HOC BY DESIGN. It reads `candidate_turn<N>.cpp` off disk rather than being
    threaded into the retry wave, so it carries no ordering constraint and cannot be
    invalidated by when it is run -- unlike the final sweep, which must follow the
    waves. It touches no simulator, so it also never contends for the one allowed
    `hexagon-sim`.
    """
    from hexkernels.forge import verify as _v
    from hexkernels.gym import intrinsic_repair as _ir
    from hexkernels.gym import metrics as _metrics

    adir = attempt_dir(model, ref, seed_index, root, rung)
    rec = _existing(adir)
    if rec is None:
        return None
    final = adir / "candidate.cpp"
    if not final.exists():
        rec["mechanism_scan_turns"] = []
        _write_atomic(adir / "attempt.json", json.dumps(rec, indent=1) + "\n")
        return rec

    # Superseded candidates are `candidate_turn<N>.cpp` for N = 1..turns-1, and the
    # live `candidate.cpp` is the last turn. Sorted numerically, not lexically: a
    # ten-turn attempt would otherwise put turn 10 before turn 2.
    prior = sorted(adir.glob("candidate_turn*.cpp"),
                   key=lambda p: int(p.stem.rsplit("candidate_turn", 1)[1]))
    valid = _ir.valid_intrinsics(sdk_root)
    scans = []
    for i, path in enumerate(list(prior) + [final], start=1):
        src = path.read_text(encoding="utf-8")
        res = _v.mechanisms_of(src, name=f"{ref.key}_s{seed_index}_t{i}",
                               sdk_root=sdk_root)
        mech = res.get("mechanisms") or {}
        entitled = list(ref.mechanisms)
        scans.append({
            "turn": i,
            "is_final": path == final,
            "candidate": path.name,
            "scanned": bool(res.get("scanned")),
            "compiled": bool(res.get("compiled")),
            "mechanisms": mech,
            "mechanisms_fired": [m for m in entitled
                                 if mech.get(MECHANISM_FLAG.get(m, m))],
            "uses_any_q6_intrinsic": "Q6_" in src,
            "hallucinated_intrinsics": sorted(
                _metrics.hallucinated_intrinsics(src, valid)),
            "error_text": (res.get("error_text") or "")[:2000],
        })
    rec["mechanism_scan_turns"] = scans
    rec["mechanism_scan_turns_at"] = _now()
    _write_atomic(adir / "attempt.json", json.dumps(rec, indent=1) + "\n")
    return rec


def cmd_sweep(args) -> int:
    """Mechanism flags for every attempt, with no simulator involved.

    ~0.4 s per attempt (a `-c` compile plus an objdump) against ~12 minutes for a
    T2 correctness run, so this answers the mechanism half of the thesis across a
    whole eval set in minutes while the simulator is still working through the
    first tier. It touches no simulator, so it is safe to run beside a grading
    pass without breaking the one-`hexagon-sim` rule.
    """
    tasks = select_tasks(evalset.core_tasks()[0], args)
    all_turns = getattr(args, "all_turns", False)
    # WHICH FIELD MEANS "ALREADY DONE" DEPENDS ON WHICH SCAN THIS IS. The two write
    # different fields, so a pre-filter keyed on the ordinary sweep's field would make
    # `--all-turns` skip every attempt the ordinary sweep had already scanned -- which,
    # in the run script's ordering (ordinary sweep, then grade, then `--all-turns`
    # last), is all of them, making the flag a silent no-op in the one sequence it
    # exists to serve.
    scan_key = "mechanism_scan_turns" if all_turns else "mechanism_scan"
    todo = []
    for ref in tasks:
        for s in range(1, args.seeds + 1):
            adir = attempt_dir(args.model, ref, s, args.root, args.rung)
            rec = _existing(adir)
            if rec is None:
                continue
            if rec.get(scan_key) and not args.redo:
                continue
            todo.append((ref, s))
    print(f"{len(todo)} attempts to scan (no simulator)", flush=True)
    if not todo:
        return 0
    n = fired = ruled_out = failed = 0
    t0 = time.time()
    fn = sweep_turns_one if all_turns else sweep_one
    for ref, s in todo:
        rec = fn(ref, args.model, s, root=args.root, rung=args.rung)
        n += 1
        # `--all-turns` writes a LIST, one entry per candidate on disk, and no entry
        # is the "final" reading `mechanism_scan` would give -- an attempt fired if
        # ANY turn fired, which is the number this scan exists to produce, and it is
        # unscannable if it has no turns at all or any turn could not be compiled.
        if scan_key == "mechanism_scan_turns":
            scans = (rec or {}).get("mechanism_scan_turns") or []
            if not scans or not all(x.get("scanned") for x in scans):
                failed += 1
            elif any(x.get("mechanisms_fired") for x in scans):
                fired += 1
            else:
                ruled_out += 1
        else:
            ms = (rec or {}).get("mechanism_scan") or {}
            if not ms.get("scanned"):
                failed += 1
            elif ms.get("mechanisms_fired"):
                fired += 1
            else:
                ruled_out += 1
        if n % 25 == 0 or n == len(todo):
            print(f"  [{n}/{len(todo)}] fired={fired} none={ruled_out} "
                  f"unscannable={failed} ({time.time() - t0:.0f}s)", flush=True)
    # The phrase names WHICH candidate(s) the count is about, because under
    # `--all-turns` "used an entitled mechanism" would read as a claim about the
    # final candidate when it is actually a claim about the attempt's turn history.
    fired_desc = "used an entitled mechanism on some turn" if all_turns else \
        "used an entitled mechanism"
    print(f"\nscanned {n}: {fired} {fired_desc}, {ruled_out} used "
          f"none (genuine ruled out), {failed} could not be scanned")
    return 0


def cmd_invalidate(args) -> int:
    """Drop verdicts that were produced under conditions we no longer trust.

    The GENERATION half of a record is never touched -- that is the expensive,
    unrepeatable part. Only `grade` is cleared, which puts the attempt back in the
    queue for the next `grade` pass.

    Three reasons this exists rather than a one-off script, all of which happened:

      * VERDICTS FROM PARALLEL SIMULATION. 342 attempts were graded with four
        simulators running. Concurrent `hexagon-sim` is not used on this platform,
        and under contention a run can exceed its timeout and be recorded as a
        wrong answer -- so those verdicts are suspect in a direction that matters
        (it understates correctness) and are re-taken serially.
      * VERDICTS FROM A BROKEN GATE. The entry-point lint rule rejected 19
        attempts whose declaration differed from the prompt only in whitespace.
        They link fine; the rule has been fixed and their verdicts are void.
      * VERDICTS FROM AN EXHAUSTED BUDGET (`--timed-out`). 19 attempts hit the
        900 s simulator cap and were written as `correct: false` -- on tasks whose
        median attempt runs ~12 min, so the cap caught the slow tail and understated
        correctness by 3 points. `grade_one` now refuses to record these at all;
        this clears the ones already on disk. Unlike the two above they are
        RECLASSIFIED rather than blanked, because the timeout is a known cause and
        the record should keep saying so.

    Being able to say WHICH verdicts were discarded and why is the difference
    between a corrected experiment and an unexplained one.
    """
    tasks = select_tasks(evalset.core_tasks()[0], args)
    hit, kept = [], 0
    for ref in tasks:
        for s in range(1, args.seeds + 1):
            adir = attempt_dir(args.model, ref, s, args.root, args.rung)
            rec = _existing(adir)
            if rec is None or not rec.get("grade"):
                continue
            g = rec["grade"]
            match = True
            if args.graded_before:
                match = match and (g.get("graded_at") or "") < args.graded_before
            if args.lint_blocked_only:
                match = match and bool(g.get("lint", {}).get("errors"))
            if args.no_evidence:
                # `compiled: false` with no compiler output at all -- the toolchain
                # never ran. See the guard in `grade_one`.
                match = match and (g.get("graded") and not g.get("compiled")
                                   and not (g.get("error_text") or "")
                                   and not (g.get("stdout") or "")
                                   and not g.get("lint", {}).get("errors"))
            if args.timed_out:
                # The string, not just the `timed_out` flag `verify` now sets: the
                # 19 records this exists to correct were written BEFORE that flag,
                # so a flag-only test would match none of them. `ran` false is what
                # keeps a completed run that merely disagreed out of the selection.
                match = match and (g.get("graded") and not g.get("ran")
                                   and (g.get("timed_out")
                                        or "timed out" in (g.get("error_text") or "")))
            if not match:
                kept += 1
                continue
            hit.append((ref, s, g.get("graded_at")))
            if not args.dry_run:
                # RECLASSIFIED, NOT BLANKED, for this selector alone. The other two
                # reasons leave nothing worth keeping -- a verdict taken under
                # contention or behind a broken gate is simply void. A timeout, by
                # contrast, is itself the finding: these attempts are ungraded
                # BECAUSE the task outran its budget, and blanking the record would
                # turn a known cause into an unexplained hole in the summary.
                rec["grade"] = ({"graded": False, "skip_reason":
                                 _timeout_skip_reason(g.get("error_text")),
                                 "reference_verified": g.get("reference_verified"),
                                 "graded_at": g.get("graded_at"),
                                 "voided_at": _now()}
                                if args.timed_out else {})
                _write_atomic(adir / "attempt.json",
                              json.dumps(rec, indent=1) + "\n")
    verb = "would clear" if args.dry_run else "cleared"
    print(f"{verb} {len(hit)} verdicts, kept {kept}")
    if hit[:3]:
        for ref, s, when in hit[:3]:
            print(f"  e.g. {ref.key} seed{s} (graded_at {when})")
    return 0


def cmd_facts(args) -> int:
    """Populate the lint-facts cache, so grading never has to trace anything.

    ITS OWN COMMAND because tracing is the phase the reaper kept killing: twice a
    grading run died partway through it and therefore graded nothing, and each death
    threw the phase away. Run once, it is minutes; after it, `grade` starts
    instantly and a kill costs only the attempts in flight.

    Traces every eval-core task by default, not just the currently gradable ones,
    so a later reference build does not send anyone back here.
    """
    tasks = select_tasks(evalset.core_tasks()[0], args)
    if args.gradable_only:
        tasks = [t for t in tasks if evalset.reference_status(t)["verified"]]
    cache = load_lint_facts(args.root)
    missing = [t for t in tasks if t.key not in cache or args.redo]
    print(f"{len(tasks)} tasks, {len(cache)} already cached, "
          f"{len(missing)} to trace", flush=True)
    t0 = time.time()
    failed = []
    for i, ref in enumerate(missing, 1):
        art = _task_artifacts(ref)
        if art["failed_stage"] is not None:
            failed.append((ref.key, art["failed_stage"]))
        else:
            cache[ref.key] = _facts_from_art(art)
        if i % 5 == 0 or i == len(missing):
            save_lint_facts(cache, args.root)
            print(f"  {i}/{len(missing)} traced ({time.time() - t0:.0f}s)",
                  flush=True)
    save_lint_facts(cache, args.root)
    print(f"cached {len(cache)} tasks in {time.time() - t0:.0f}s")
    for key, stage in failed:
        print(f"  TASK BUILD FAILED {key} at {stage}")
    return 0


def cmd_grade(args) -> int:
    # Refused rather than clamped, so a caller who asked for 6 finds out they did
    # not get 6 instead of quietly believing they did.
    if args.jobs != 1:
        print(f"--jobs {args.jobs} refused: simulation runs single-threaded on "
              f"this platform (one hexagon-sim at a time). Re-run with --jobs 1.",
              file=sys.stderr)
        return 2
    tasks, _ = evalset.core_tasks()
    tasks = select_tasks(tasks, args)
    todo = []
    for ref in tasks:
        for s in range(1, args.seeds + 1):
            adir = attempt_dir(args.model, ref, s, args.root, args.rung)
            rec = _existing(adir)
            if rec is None:
                continue
            if rec.get("grade") and not args.redo:
                # Re-grade an attempt that was skipped only because its reference
                # was unbuilt at the time: that is the state the parallel reference
                # build is there to change.
                g = rec["grade"]
                if g.get("graded") or "no candidate" in (g.get("skip_reason") or ""):
                    continue
            todo.append((ref, s))
    print(f"{len(todo)} attempts to grade with {args.jobs} jobs", flush=True)
    if not todo:
        return 0

    # ON THE MAIN THREAD, before the pool starts: tracing concurrently segfaults
    # (see `grade_one`). Cached across runs, because the reaper killed a grading run
    # in exactly this phase and it therefore graded nothing. Only tasks that will
    # actually reach the simulator are resolved -- an attempt whose reference is
    # unusable needs no facts.
    cache = load_lint_facts(args.root)
    needed = {r.key: r for r, _s in todo
              if evalset.reference_status(r)["verified"]}
    missing = [r for k, r in needed.items() if k not in cache]
    if missing:
        t0 = time.time()
        for i, ref in enumerate(missing, 1):
            cache[ref.key] = _facts_from_art(_task_artifacts(ref))
            if i % 10 == 0 or i == len(missing):
                # Saved as we go: a kill mid-phase then costs only the task in
                # flight instead of the whole phase.
                save_lint_facts(cache, args.root)
                print(f"  traced {i}/{len(missing)} new tasks "
                      f"({time.time() - t0:.0f}s)", flush=True)
        save_lint_facts(cache, args.root)
    print(f"lint facts ready for {len(needed)} tasks "
          f"({len(missing)} newly traced, {len(needed) - len(missing)} cached)",
          flush=True)
    facts = {k: cache[k] for k in needed if k in cache}

    counts = {"correct": 0, "compiled": 0, "genuine": 0, "skipped": 0, "n": 0}

    def work(item):
        ref, s = item
        # As in `cmd_generate`: a crash in one attempt must not discard the queue.
        # Here it matters even more, because a grading pass is hours long.
        try:
            return ref, s, grade_one(ref, args.model, s, root=args.root,
                                     rung=args.rung,
                                     timeout=args.timeout, timing=args.timing,
                                     facts=facts.get(ref.key))
        except Exception as exc:                  # noqa: BLE001 - recorded, not raised
            adir = attempt_dir(args.model, ref, s, args.root, args.rung)
            rec = _existing(adir) or _attempt_skeleton(ref, args.model, s,
                                                       args.rung)
            rec["grade"] = {"graded": False, "reference_verified": None,
                            "skip_reason": f"grader error: {type(exc).__name__}: "
                                           f"{exc}", "graded_at": _now()}
            _write_atomic(adir / "attempt.json", json.dumps(rec, indent=1) + "\n")
            _write_atomic(adir / "grader_error.txt", traceback.format_exc())
            return ref, s, rec

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.jobs) as pool:
        for ref, s, rec in pool.map(work, todo):
            g = (rec or {}).get("grade") or {}
            counts["n"] += 1
            if not g.get("graded"):
                counts["skipped"] += 1
                print(f"  [{counts['n']}/{len(todo)}] {ref.key:38s} seed{s} "
                      f"SKIP {g.get('skip_reason', '')[:50]}", flush=True)
                continue
            for k in ("compiled", "correct", "genuine"):
                counts[k] += 1 if g.get(k) else 0
            print(f"  [{counts['n']}/{len(todo)}] {ref.key:38s} seed{s} "
                  f"compiled={g['compiled']} correct={g['correct']} "
                  f"genuine={g['genuine']}", flush=True)
    print(f"\ngraded {counts['n'] - counts['skipped']}, skipped "
          f"{counts['skipped']}; compiled {counts['compiled']}, correct "
          f"{counts['correct']}, genuine {counts['genuine']}")
    return 0


# --------------------------------------------------------------------- report

def collect(model, seeds, root=RUNS, rung=0):
    """Every attempt record for one model, in task then seed order."""
    tasks, _ = evalset.core_tasks()
    out = []
    for ref in tasks:
        for s in range(1, seeds + 1):
            rec = _existing(attempt_dir(model, ref, s, root, rung))
            if rec is not None:
                out.append(rec)
    return out


def cmd_report(args) -> int:
    from hexkernels.forge import aggregate
    recs = collect(args.model, args.seeds, args.root, args.rung)
    if not recs:
        print("no attempts found", file=sys.stderr)
        return 2
    out = run_dir(args.model, args.root, args.rung)
    jsonl = out / "attempts.jsonl"
    _write_atomic(jsonl, "".join(json.dumps(r, sort_keys=True) + "\n"
                                 for r in recs))
    summary = aggregate.summarise(recs)
    _write_atomic(out / "summary.json", json.dumps(summary, indent=1) + "\n")
    text = aggregate.render(summary, model=args.model)
    _write_atomic(out / "SUMMARY.md", text)
    print(text)
    print(f"\nwrote {jsonl} ({len(recs)} attempts) and {out / 'SUMMARY.md'}")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--model", default="gpt-5.6-terra")
        p.add_argument("--seeds", type=int, default=5,
                       help="how many samples per task (PLAN.md section 2 "
                            "requires 5 minimum)")
        p.add_argument("--root", default=str(RUNS))
        # Which LADDER RUNG's tree to act on. Grading, sweeping and reporting are
        # rung-agnostic (see `run_dir`), so they are shared across rungs rather than
        # copied per rung -- which is also what makes a rung-0/rung-1 delta a
        # comparison of prompts and not of two grading implementations.
        p.add_argument("--rung", type=int, default=0,
                       help="ladder rung whose attempts to act on (default 0)")
        p.add_argument("--only", default=None,
                       help="comma-separated task keys or ids")
        # TIER IS A SCHEDULING TOOL, not a scoping one. Serial simulation makes a
        # T2 attempt cost ~4 minutes against a few seconds for T0, so grading in
        # arrival order puts hours of big kernels in front of a result that is
        # otherwise minutes away. Grading T0,T1 first buys a reportable baseline
        # early; the tiers left out stay queued, and every record is stamped with
        # its tier so a partial run can never be mistaken for a whole one.
        p.add_argument("--tier", default=None,
                       help="comma-separated tiers to include, e.g. T0,T1")
        p.add_argument("--redo", action="store_true")

    g = sub.add_parser("generate", help="call the model; no toolchain needed")
    common(g)
    g.add_argument("--jobs", type=int, default=4)
    g.add_argument("--limit", type=int, default=None,
                   help="first N tasks only, for a pilot")
    g.add_argument("--max-tokens", type=int, default=16384)
    g.add_argument("--temperature", type=float, default=None)
    g.add_argument("--seed-base", type=int, default=1000,
                   help="request seed = base + seed index; None to omit")
    g.add_argument("--key-file", default=None)
    g.add_argument("--price-in", type=float, default=None,
                   help="USD per 1M prompt tokens, for cost accounting")
    g.add_argument("--price-out", type=float, default=None)
    g.add_argument("--price-cached", type=float, default=None,
                   help="USD per 1M CACHED prompt tokens. Omitted means unknown, "
                        "and cached tokens are then billed at the full input rate "
                        "rather than silently discounted")
    g.set_defaults(fn=cmd_generate)

    w = sub.add_parser("sweep",
                       help="mechanism flags for every attempt; no simulator")
    common(w)
    w.add_argument("--all-turns", action="store_true",
                   help="scan every saved turn's candidate into "
                        "`mechanism_scan_turns` instead of only the final one. "
                        "Additive: it never writes `mechanism_scan`, and it needs "
                        "no simulator, so it is safe to run at any time")
    w.set_defaults(fn=cmd_sweep)

    i = sub.add_parser("invalidate",
                       help="clear verdicts (never generations) so they re-grade")
    common(i)
    i.add_argument("--graded-before", default=None,
                   help="ISO timestamp; clear verdicts older than this")
    i.add_argument("--lint-blocked-only", action="store_true",
                   help="clear only verdicts blocked by the static lint gate")
    i.add_argument("--no-evidence", action="store_true",
                   help="clear 'did not compile' verdicts that carry no compiler "
                        "output -- the toolchain never ran, so they are not results")
    i.add_argument("--timed-out", action="store_true",
                   help="reclassify verdicts whose simulator ran out of budget as "
                        "UNGRADED with that reason -- a timeout is an exhausted "
                        "budget, never a wrong answer")
    i.add_argument("--dry-run", action="store_true")
    i.set_defaults(fn=cmd_invalidate)

    f = sub.add_parser("facts", help="pre-trace the lint facts grading needs")
    common(f)
    f.add_argument("--gradable-only", action="store_true",
                   help="only tasks whose reference already passed stage (g)")
    f.set_defaults(fn=cmd_facts)

    r = sub.add_parser("grade", help="simulate and score; no network needed")
    common(r)
    # ONE SIMULATOR AT A TIME. NOT A TUNING PARAMETER.
    #
    # Standing instruction from the user, 2026-08-17: "never parallelize sims in
    # windows ... use only 1 sim thread." Raising this is not a speed/robustness
    # trade the caller gets to make -- concurrent `hexagon-sim` on this platform is
    # off the table, so the default is 1 and `--jobs` above 1 is refused below.
    #
    # What was observed before the instruction, for whoever reads this later:
    # running several simulators alongside the reference build starved the box, and
    # orphaned `hexagon-sim.exe` processes survived every kill and kept relaunching
    # from a shell loop that also survived. Single-threaded grading has neither
    # failure mode.
    r.add_argument("--jobs", type=int, default=1,
                   help="MUST be 1: concurrent simulation is not used on Windows")
    r.add_argument("--timeout", type=int, default=900)
    # OFF, and settled rather than merely defaulted. The user, 2026-08-17: "We will
    # measure the timings and other metrics on the silicon." The simulator is a
    # correctness and mechanism gate; no cycle number is taken from it. Kept as a
    # flag only for one-off diagnosis -- a record made with it carries
    # `sim_timing: true` so it can never be pooled with the rest unnoticed.
    r.add_argument("--timing", action="store_true",
                   help="diagnosis only: cycle-accurate simulation, ~5x slower. "
                        "Timings for the paper are measured on silicon, so no "
                        "result should need this")
    r.set_defaults(fn=cmd_grade)

    p = sub.add_parser("report", help="aggregate to SUMMARY.md + attempts.jsonl")
    common(p)
    p.set_defaults(fn=cmd_report)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
