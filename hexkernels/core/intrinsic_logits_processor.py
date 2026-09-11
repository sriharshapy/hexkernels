"""vLLM V1 per-request logits processor: forbid fabricated Q6_ intrinsics, but ONLY while the
decoder is inside a Q6_ identifier (char-level masking). ~99% of tokens are outside any
intrinsic name -> no mask computed -> ~full speed. This is the fast, targeted alternative to the
whole-output xgrammar grammar, whose per-token CPU mask over the full vocab is the throughput
bottleneck (GPU sits idle). See docs/superpowers/specs/2026-07-30-constrained-intrinsic-decoding.

Registered at engine init: ``LLM(logits_processors=[IntrinsicMaskLogitsProcessor])``.
Activated per request: ``SamplingParams(extra_args={"constrain_intrinsics": True})`` (absent ->
this processor returns None for that request -> no effect, so the unconstrained arm is untouched).

Valid-name source (in priority order): env ``INTRINSIC_NAMES_JSON`` (a JSON list) else
``HEXAGON_SDK_ROOT`` header parse. Fail-closed: empty name set -> the trie allows nothing inside
a fragment (but new_req_logits_processor still only engages when explicitly requested)."""
import json
import os
import re

from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from hexkernels.gym.intrinsic_grammar import (
    IntrinsicTrie, masked_token_ids_tokenlevel, valid_intrinsics)

_WORD = re.compile(r"[A-Za-z0-9_]")
_TAIL = 64  # tokens of tail to decode; inside a fragment tokens are 1 char, names <~25 chars


def _load_names():
    nf = os.environ.get("INTRINSIC_NAMES_JSON")
    if nf and os.path.exists(nf):
        return set(json.load(open(nf)))
    sdk = os.environ.get("HEXAGON_SDK_ROOT")
    return valid_intrinsics(sdk) if sdk else set()


class IntrinsicMaskLogitsProcessor(AdapterLogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        super().__init__(vllm_config, device, is_pin_memory)
        from transformers import AutoTokenizer
        mc = vllm_config.model_config
        tok = AutoTokenizer.from_pretrained(mc.tokenizer or mc.model)
        self._tok = tok
        self._trie = IntrinsicTrie(_load_names())
        # token-level index: id -> decoded string, first-char -> [ids], boundary ids (start with
        # a non-word char -> end an identifier). Enables keeping the model's NATURAL multi-char
        # tokens (no char-by-char forcing -> no name drift).
        n = len(tok)
        id2str = [""] * n
        first_char_index = {}
        boundary = []
        for tid in range(n):
            try:
                s = tok.decode([tid])
            except Exception:
                s = ""
            id2str[tid] = s
            if not s:
                continue
            first_char_index.setdefault(s[0], []).append(tid)
            if not _WORD.match(s[0]):
                boundary.append(tid)
        self._id2str = id2str
        self._first_char_index = first_char_index
        self._boundary = boundary

    def is_argmax_invariant(self) -> bool:
        return False  # masking can change the argmax token

    def new_req_logits_processor(self, params):
        ea = getattr(params, "extra_args", None) or {}
        if not ea.get("constrain_intrinsics"):
            return None  # request opted out -> no masking (unconstrained arm)
        trie, id2str, fci, boundary, tok = (
            self._trie, self._id2str, self._first_char_index, self._boundary, self._tok)

        def _req_lp(output_ids, logits):
            if not output_ids:
                return logits
            import torch
            tail = tok.decode(output_ids[-_TAIL:])
            ids = masked_token_ids_tokenlevel(trie, tail, id2str, fci, boundary)
            if ids is None or not ids:
                return logits  # outside a Q6_ fragment (the common case) -> no-op
            mask = torch.full_like(logits, float("-inf"))
            mask[torch.as_tensor(ids, device=logits.device, dtype=torch.long)] = 0.0
            return logits + mask

        return _req_lp
