"""Policies over HexagonKernelEnv. Pure orchestration: LLM via injected call_fn,
evaluation via the env. No vllm/torch import here."""
from hexkernels.core.code_extract import extract_c_source
from hexkernels.gym import reward as R

_FIX_TAIL = "\n\nReturn the full corrected file in one ```c code block."


def _rec(src, fb, spec):
    return {"src": src, "feedback": fb, "reward": R.reward_tier(fb, spec)}


def single_shot(env, task_id, call_fn):
    prompt = env.reset(task_id)
    src = extract_c_source(call_fn([{"role": "user", "content": prompt}]) or "")
    obs, rew, done, fb = env.step(src)
    return {"src": src, "feedback": fb, "reward": rew}


def _fix_messages(prompt, anchor_src, obs):
    return [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": "```c\n" + anchor_src + "\n```"},
        {"role": "user", "content": obs + _FIX_TAIL},
    ]


def multiturn_repair(env, task_id, call_fn):
    prompt = env.reset(task_id)
    recs = []
    src = extract_c_source(call_fn([{"role": "user", "content": prompt}]) or "")
    obs, rew, done, fb = env.step(src)
    recs.append({"src": src, "feedback": fb, "reward": rew})
    anchor = src
    while not done:
        src = extract_c_source(call_fn(_fix_messages(prompt, anchor, obs)) or "")
        obs, rew, done, fb = env.step(src)
        recs.append({"src": src, "feedback": fb, "reward": rew})
        anchor = src or anchor
    return recs


def _key(rec):
    kc = rec["feedback"].get("kernel_cycles")
    return (rec["reward"], -(kc if kc is not None else float("inf")))


def best_of_n(env, task_id, call_fn, n, spec):
    prompt = env.reset(task_id)
    recs = []
    for _ in range(n):
        src = extract_c_source(call_fn([{"role": "user", "content": prompt}]) or "")
        fb = env.evaluate_once(task_id, src)
        recs.append(_rec(src, fb, spec))
    return max(recs, key=_key)


def pick_best(records):
    """Best record in a trajectory: highest reward tier, then fewest kernel_cycles.
    Never credits a later regression below a better earlier attempt (multi-turn can
    otherwise push a correct-but-scalar kernel into a broken genuine attempt)."""
    return max(records, key=_key)
