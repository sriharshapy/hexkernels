"""Frontier-model generator behind the SAME interface the local models use.

The whole point of the frontier baseline is that it is not a different experiment. This
implements only the two methods `gym_holdout_batched.py` calls -- `generate_batch` and
`chat_batch` -- so the frontier arms run through the UNCHANGED driver and inherit the
identical harness, reward tiers, anti-cheat and scoreboard. A parallel eval path would
make the comparison worthless no matter how carefully it was written.

BUILD THE STRONGEST HONEST BASELINE. A handicapped baseline is worth nothing: a reviewer
will assume it was rigged, and rightly. Hence `system_preamble`, which carries the
escalating context conditions (valid intrinsic list, few-shot exemplars) as a constant
prefix -- constant so the provider's prompt cache absorbs it, which is what makes the
full 1392-name list affordable.

BUDGET MATCHING is the caller's job: pass the same k, max_tokens and turn count the local
arm gets. An unmatched comparison is not evidence.

The key is read from a file or the environment and never logged.
"""
import os
import threading
import time

_UNSET = object()


def _load_key(key_file=None):
    if key_file and os.path.exists(key_file):
        with open(key_file, encoding="utf-8") as f:
            return f.read().strip()
    for env in ("OPENAI_API_KEY", "HEXBENCH_OPENAI_KEY"):
        if os.environ.get(env):
            return os.environ[env].strip()
    raise RuntimeError(
        "no API key: pass key_file or set OPENAI_API_KEY (never hardcode one)")


class OpenAIGenerator:
    """CandidateGenerator-compatible wrapper over an OpenAI chat model."""

    def __init__(self, model_id="gpt-5", key_file=None, temperature=0.8,
                 max_new_tokens=4096, seed=None, system_preamble=None,
                 concurrency=8, max_retries=6):
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.system_preamble = system_preamble
        self.concurrency = concurrency
        self.max_retries = max_retries
        self._key = _load_key(key_file)
        self._lock = threading.Lock()
        self.n_calls = 0
        self.n_prompt_tokens = 0
        self.n_completion_tokens = 0
        self._client = None

    def _ensure(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI(api_key=self._key)
        return self._client

    def _messages(self, user_content, prior=None):
        msgs = []
        if self.system_preamble:
            msgs.append({"role": "system", "content": self.system_preamble})
        if prior:
            msgs.extend(prior)
        else:
            msgs.append({"role": "user", "content": user_content})
        return msgs

    def _one(self, messages):
        """One chat completion, with backoff. Returns '' rather than raising: a single
        failed request must not void a 64-task run, and an empty completion is scored
        exactly as a non-compiling kernel -- which is the honest outcome."""
        client = self._ensure()
        delay = 2.0
        for attempt in range(self.max_retries):
            try:
                kw = dict(model=self.model_id, messages=messages,
                          max_completion_tokens=self.max_new_tokens)
                # Some frontier models reject non-default sampling knobs; omit rather
                # than fail, and record that the arm ran at provider defaults.
                if self.temperature is not None:
                    kw["temperature"] = self.temperature
                if self.seed is not None:
                    kw["seed"] = self.seed
                r = client.chat.completions.create(**kw)
                with self._lock:
                    self.n_calls += 1
                    u = getattr(r, "usage", None)
                    if u:
                        self.n_prompt_tokens += getattr(u, "prompt_tokens", 0) or 0
                        self.n_completion_tokens += getattr(u, "completion_tokens", 0) or 0
                return r.choices[0].message.content or ""
            except Exception as e:
                msg = str(e)
                if "temperature" in msg and "temperature" in kw:
                    self.temperature = None       # retry at provider default
                    continue
                if attempt == self.max_retries - 1:
                    return ""
                time.sleep(delay)
                delay = min(delay * 2, 60)
        return ""

    def _map(self, jobs):
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.concurrency) as ex:
            return list(ex.map(self._one, jobs))

    # --- the two methods the eval driver calls -----------------------------
    def generate_batch(self, prompts, k, enable_thinking=_UNSET, max_new_tokens=None):
        """List[prompt] -> List[List[str]] with k completions each."""
        if max_new_tokens:
            self.max_new_tokens = max_new_tokens
        jobs = [self._messages(p) for p in prompts for _ in range(k)]
        outs = self._map(jobs)
        return [outs[i * k:(i + 1) * k] for i in range(len(prompts))]

    def chat_batch(self, conversations, max_new_tokens=None):
        """List[messages] -> List[str]. Used by the multi-turn repair loop."""
        if max_new_tokens:
            self.max_new_tokens = max_new_tokens
        return self._map([self._messages(None, prior=c) for c in conversations])

    def generate(self, prompt, k):
        return self.generate_batch([prompt], k)[0]

    def chat(self, messages):
        return self.chat_batch([messages])[0]

    def usage(self):
        return {"model": self.model_id, "calls": self.n_calls,
                "prompt_tokens": self.n_prompt_tokens,
                "completion_tokens": self.n_completion_tokens,
                "temperature": self.temperature, "seed": self.seed,
                "max_new_tokens": self.max_new_tokens}
