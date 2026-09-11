"""Generate candidates with an external model. Provider-agnostic, HTTP only.

    python -m hexkernels.forge.model_client --batch 12 \
        --out run_artifacts/forge2/batch12 \
        --provider openai --model gpt-5 \
        --candidates run_artifacts/forge2/batch12/candidates_gpt5

then judge them exactly as a hand-written candidate is judged:

    python -m hexkernels.forge.run_batch --batch 12 \
        --out run_artifacts/forge2/batch12 \
        --candidates run_artifacts/forge2/batch12/candidates_gpt5

WHY THIS FILE IS SO SMALL, AND WHY IT IS SEPARATE
-------------------------------------------------
`forge2` has no model in it. The pipeline writes `<out>/<kernel>/prompt.md` and
reads `<candidates>/<kernel>.cpp`; what happens in between is not its business.
Every kernel in this corpus so far was written by hand against those prompts, so
"swap in an API" is not a port -- it is a different producer for the same
directory. This module exists to make that concrete and to keep the HTTP
dependency out of the pipeline: nothing under `forge2` imports it, and the gate
does not need a network.

The prompts are already API-shaped. They are self-contained plain text, 5-15 KB
apiece, and they end with "Return only the C++ translation unit, no prose and no
markdown fence."

WHAT THE PIPELINE STILL ENFORCES, WHOEVER PRODUCED THE FILE
----------------------------------------------------------
Nothing here is trusted. `run_batch` runs the same stages on an API-written
candidate as on a hand-written one, and two of them are the reason this is safe:

  * `provenance.rd_leaks` scans the source for the R&D header names AND the
    exported symbol names, and rejects BEFORE compiling. A model that regurgitates
    `HVX_VTCM_BASE` or `hmx_tile_matmul_*` from its training data is refused, not
    silently accepted -- which is what makes a third-party model's output
    admissible on the same terms as anything else here.
  * correctness is decided by executing the kernel on the simulator against
    golden vectors from PyTorch. There is no path by which a plausible-looking
    answer passes.

ONE-SHOT IS THE WRONG EXPECTATION, and the honest note to whoever runs this.
Several kernels in this corpus needed simulator MEASUREMENTS to get right -- the
`vlalign` shift direction, the deal-order of HVX widening, the DMA priming rule --
and one whole line of enquiry (HMX int8) ended in a documented dead end after four
probe rounds. A single completion gets one attempt at that. `--rounds N` feeds
each failure back (compiler diagnostics, the harness's error count and first bad
index, any provenance leak), which is the setting this pipeline is actually shaped
for: `verify()` already returns all of it structured.
"""
import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.request

from hexkernels.forge import prompt as _prompt

# provider -> (env var, url, request builder, response extractor)
#
# Kept as a table rather than an SDK dependency: the whole interaction is one
# POST with a text prompt and one text answer, so an SDK would be a heavier
# dependency than the thing it wraps, and it would pin versions the gate has no
# reason to care about.
PROVIDERS = {
    "openai": {
        "env": "OPENAI_API_KEY",
        "url": "https://api.openai.com/v1/chat/completions",
        "auth": lambda key: {"Authorization": f"Bearer {key}"},
        "body": lambda model, prompt, max_tok: {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_completion_tokens": max_tok,
        },
        "text": lambda r: r["choices"][0]["message"]["content"],
    },
    "anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "url": "https://api.anthropic.com/v1/messages",
        "auth": lambda key: {"x-api-key": key, "anthropic-version": "2023-06-01"},
        "body": lambda model, prompt, max_tok: {
            "model": model,
            "max_tokens": max_tok,
            "messages": [{"role": "user", "content": prompt}],
        },
        "text": lambda r: "".join(b.get("text", "") for b in r["content"]),
    },
    # Anything speaking the OpenAI chat schema: vLLM, llama.cpp, Together,
    # Groq, OpenRouter, a local server. Point --base-url at it.
    "openai-compatible": {
        "env": "MODEL_API_KEY",
        "url": None,
        "auth": lambda key: {"Authorization": f"Bearer {key}"} if key else {},
        "body": lambda model, prompt, max_tok: {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tok,
        },
        "text": lambda r: r["choices"][0]["message"]["content"],
    },
}

_FENCE = re.compile(r"```(?:cpp|c\+\+|c)?\s*\n(.*?)```", re.S)


def strip_fence(text: str) -> str:
    """The prompt asks for a bare translation unit; take one anyway if fenced.

    Defensive rather than trusting: models fence code even when told not to, and
    a stray fence is a COMPILE failure that would be recorded against the model's
    kernel-writing ability instead of against its instruction following. Those are
    different measurements and conflating them would flatter or damn the wrong
    thing.
    """
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip() + "\n"
    return text.strip() + "\n"


def complete(provider: str, model: str, prompt: str, *, base_url=None,
             max_tokens=16384, timeout=600) -> str:
    spec = PROVIDERS[provider]
    key = os.environ.get(spec["env"], "")
    if spec["env"] and not key and provider != "openai-compatible":
        raise SystemExit(f"{spec['env']} is not set")
    url = base_url or spec["url"]
    if not url:
        raise SystemExit("--base-url is required for openai-compatible")
    body = json.dumps(spec["body"](model, prompt, max_tokens)).encode()
    headers = {"Content-Type": "application/json"}
    headers.update(spec["auth"](key))
    req = urllib.request.Request(url, data=body, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return spec["text"](json.loads(resp.read().decode()))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode(errors="replace")[:400]
        raise SystemExit(f"{provider} HTTP {exc.code}: {detail}") from None


def feedback(verdict: dict) -> str:
    """Turn one verify() verdict into the text a retry needs.

    Ordered by what blocks first, because that is the order the pipeline applies
    them: a provenance leak is judged before compiling, and compiling before
    running. Telling a model its numbers are wrong when it never compiled would
    send it to fix the wrong thing.
    """
    if verdict.get("rd_leak"):
        return ("Your kernel was REJECTED WITHOUT COMPILING: it references "
                "R&D material from the benchmark repository, which the "
                "provenance rules forbid. Offending names: "
                + ", ".join(verdict["rd_leak"])
                + ". Derive those facts yourself under your own names.")
    if verdict.get("lint_errors"):
        # Before the compiler, because that is where the static gate sits. Telling
        # a model "you did not compile" when it was never compiled would send it
        # looking for a syntax error that is not there.
        return ("Your kernel was REJECTED WITHOUT COMPILING by the static gate. "
                "Each of these is a mistake that has produced a crash or silently "
                "wrong data in this corpus before:\n"
                + "\n".join(f"  - {e}" for e in verdict["lint_errors"])
                + "\n\n" + _prompt.retry_facts(verdict))
    if not verdict.get("compiled"):
        return ("Your kernel did not compile. Compiler output:\n"
                + (verdict.get("error_text") or "")[:3000])
    if not verdict.get("ran"):
        return ("Your kernel compiled but did not run to completion on the "
                "simulator (a fault or a timeout). Output:\n"
                + (verdict.get("stdout") or "")[:2000])
    if not verdict.get("correct"):
        # THE SHAPE OF THE BAD SET, then the measured ISA note. Both are
        # retry-only on purpose. `verify.diagnose` reasons from WHICH elements are
        # wrong; `prompt.retry_facts` names intrinsics the first prompt
        # deliberately withholds, and that function documents why the line sits
        # between the first ask and a retry rather than in one place or the other.
        # An author working through an API cannot inspect the disassembly the way
        # a local session can, so without these a lane-order bug is three rounds
        # of guessing.
        parts = ["Your kernel compiled and ran but produced WRONG VALUES. The "
                 "harness reports:\n" + (verdict.get("stdout") or "")[:2000]]
        diag = verdict.get("diagnosis") or []
        if diag:
            parts.append("The SHAPE of the wrong elements says where to look. "
                         "These are hypotheses, not conclusions:\n"
                         + "\n".join(f"  - {d}" for d in diag))
        parts.append(_prompt.retry_facts(verdict))
        parts.append("The reference in the prompt is the specification; match it "
                     "exactly within the stated tolerance.")
        return "\n\n".join(parts)
    return ""


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--batch", type=int, required=True)
    ap.add_argument("--out", required=True,
                    help="the run_batch output dir holding <kernel>/prompt.md")
    ap.add_argument("--candidates", required=True,
                    help="directory to write <kernel>.cpp into")
    ap.add_argument("--provider", default="openai", choices=sorted(PROVIDERS))
    ap.add_argument("--model", required=True)
    ap.add_argument("--base-url", default=None)
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--rounds", type=int, default=1,
                    help="attempts per kernel; >1 verifies and feeds the failure "
                         "back, which needs the toolchain")
    ap.add_argument("--only", default=None, help="one kernel name")
    args = ap.parse_args(argv)

    from hexkernels.forge.kernels import batch as _batch
    names = [s.name for s in _batch(args.batch)]
    if args.only:
        names = [n for n in names if n == args.only] or names
    os.makedirs(args.candidates, exist_ok=True)

    summary = []
    for name in names:
        prompt_path = os.path.join(args.out, name, "prompt.md")
        if not os.path.exists(prompt_path):
            print(f"  {name:24s} SKIP (no prompt -- was the reference built?)")
            continue
        with open(prompt_path, encoding="utf-8") as f:
            prompt = f.read()
        dest = os.path.join(args.candidates, f"{name}.cpp")

        convo, verdict = prompt, None
        for rnd in range(1, args.rounds + 1):
            src = strip_fence(complete(args.provider, args.model, convo,
                                       base_url=args.base_url,
                                       max_tokens=args.max_tokens))
            with open(dest, "w", encoding="utf-8") as f:
                f.write(src)
            if args.rounds == 1:
                print(f"  {name:24s} wrote {len(src):6d} B  (round {rnd})")
                break
            # Verify and retry. Imported here so a --rounds 1 run needs no
            # toolchain at all.
            from hexkernels.forge import lint as _lint
            from hexkernels.forge import run_batch as _rb
            from hexkernels.forge import verify as _v
            art = _rb.build(next(s for s in _batch(args.batch) if s.name == name))
            # THE STATIC GATE FIRST, on this path most of all. A round spent
            # compiling and simulating a kernel whose DMA descriptor is on the
            # stack is a round spent learning nothing -- and here a round costs an
            # API call as well as a simulator run. Nine rules, milliseconds,
            # `feedback` turns the findings into the retry text.
            #
            # This was the THIRD consumer of candidate source found to be
            # bypassing the gate (after `check.py`), and all three were found by
            # asking which entry points reach it rather than by reading it.
            findings = _lint.lint(src, signature=f'extern "C" {art["signature"]}',
                                  tier=art.get("tier"),
                                  mechanisms=art.get("mechanisms") or (),
                                  nelem=art.get("nelem"), dtype=art.get("dtype"))
            blocking = _lint.errors(findings)
            if blocking:
                verdict = {"present": True, "compiled": False, "ran": False,
                           "correct": False, "mechanisms": {}, "rd_leak": [],
                           "lint_errors": [str(f) for f in blocking],
                           "error_text": _lint.format_findings(blocking)}
            else:
                verdict = _v.verify(src, art["harness_c"], name=f"{name}_api")
                verdict["lint_errors"] = []
                verdict["lint_warnings"] = [str(f) for f in findings
                                           if f.severity == "warn"]
            fb = feedback(verdict)
            ok = "OK" if not fb else "retry"
            print(f"  {name:24s} round {rnd}: {ok}"
                  f" compiled={verdict.get('compiled')}"
                  f" correct={verdict.get('correct')}")
            if not fb:
                break
            convo = prompt + "\n\n---\nYour previous attempt failed.\n" + fb
        summary.append((name, verdict))

    if any(v for _n, v in summary):
        ok = sum(1 for _n, v in summary if v and v.get("correct"))
        print(f"\n{ok}/{len(summary)} correct after {args.rounds} round(s)")
    print(f"\nwrote candidates to {args.candidates}\nnow run:\n"
          f"  python -m hexkernels.forge.run_batch --batch {args.batch} "
          f"--out {args.out} --candidates {args.candidates}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
