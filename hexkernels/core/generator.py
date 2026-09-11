"""Candidate kernel generators for the eval (mockable; the real ones lazy-load
torch/transformers/vLLM so ``import hexkernels.core.generator`` works without
CUDA/torch/vLLM installed).

Port of ``m_rl/eval/generator.py`` (OLD HVX repo). No dataset-version paths in
this module (it only talks to HF/vLLM model loading), so nothing to repoint;
the heavy imports (torch, transformers, peft, liger_kernel, vllm) were already
lazy (inside ``_ensure``/``generate``) in the OLD module and remain so here.
"""


# Sentinel distinguishing "caller did not pass enable_thinking" (fall back to the
# instance default) from an explicit ``None`` (let the chat template decide). A plain
# ``None`` default can't express that, so we use a private object.
_UNSET = object()


class CandidateGenerator:
    def generate(self, prompt, k):
        """Return a list of k raw completion strings for one prompt."""
        raise NotImplementedError


class LocalModelGenerator(CandidateGenerator):
    def __init__(self, model_id="Qwen/Qwen2.5-Coder-1.5B-Instruct", temperature=0.8,
                 max_new_tokens=512, adapter=None, enable_thinking=None,
                 precision="bf16", attn="sdpa", use_liger=False, seed=None):
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.adapter = adapter          # optional LoRA/PEFT adapter dir (e.g. SFT warm-start)
        # None = let the chat template decide; False = explicitly disable the Qwen3
        # <think> block (the pipeline can't forward chat-template kwargs, so when set we
        # render the template to a string ourselves — see _render).
        self.enable_thinking = enable_thinking
        # T4 (sm75) has no native bf16 tensor-core path → fp16 is much faster there.
        # attn="sdpa" is the flash-attention equivalent that works on T4 (flash-attn2
        # is unsupported on sm75). use_liger applies the fused Liger kernels.
        self.precision = precision
        self.attn = attn
        self.use_liger = use_liger
        # seed=None -> today's behaviour: no seeding, fully random do_sample draws each
        # call. When set, torch is reseeded right before sampling so a run is
        # reproducible/distinct across seeds (see run.py --seed).
        self.seed = seed
        self._pipe = None
        self._tok = None

    def _ensure(self):
        if self._pipe is not None:
            return
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.precision, torch.bfloat16)
        tok = AutoTokenizer.from_pretrained(self.model_id)
        self._tok = tok
        model_kwargs = dict(dtype=dtype, device_map="cuda")
        if self.attn:
            model_kwargs["attn_implementation"] = self.attn
        loader = AutoModelForCausalLM
        if self.use_liger:
            # AutoLigerKernelForCausalLM auto-detects the arch and monkeypatches the
            # fused Liger kernels (RMSNorm/RoPE/SwiGLU) before constructing the model.
            from liger_kernel.transformers import AutoLigerKernelForCausalLM
            loader = AutoLigerKernelForCausalLM
        model = loader.from_pretrained(self.model_id, **model_kwargs)
        if self.adapter:
            # Apply a trained LoRA adapter (SFT warm-start) on top of the base model.
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter)
        self._pipe = pipeline("text-generation", model=model, tokenizer=tok)

    def _render(self, prompt):
        """Model input for one prompt. Default: hand the pipeline the message list and
        let it apply the chat template. If enable_thinking is set we pre-render the
        template to a string (with that flag) because the text-generation pipeline has
        no way to forward chat-template kwargs like enable_thinking."""
        msgs = [{"role": "user", "content": prompt}]
        if self.enable_thinking is None:
            return msgs
        return self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True,
            enable_thinking=self.enable_thinking)

    def generate(self, prompt, k):
        self._ensure()
        if self.seed is not None:
            import torch
            torch.manual_seed(self.seed)
        inp = self._render(prompt)
        # Batch all k samples in ONE call (num_return_sequences) so the GPU generates
        # them together — ~k x fewer forward loops than a Python for-loop, and
        # statistically identical (k independent do_sample draws). Big T4 speedup.
        try:
            r = self._pipe(inp, max_new_tokens=self.max_new_tokens, do_sample=True,
                           temperature=self.temperature, return_full_text=False,
                           num_return_sequences=k)
            return [x["generated_text"] for x in r]
        except Exception:
            # Fallback to the sequential path if the pipeline can't batch this model.
            outs = []
            for _ in range(k):
                r = self._pipe(inp, max_new_tokens=self.max_new_tokens, do_sample=True,
                               temperature=self.temperature, return_full_text=False)
                outs.append(r[0]["generated_text"])
            return outs

    def chat(self, messages):
        """ONE (n=1) completion from a full chat-messages list (role/content
        dicts, e.g. from `core.multiturn`'s fix_messages or
        `core`'s plan/implement/fix_messages).

        This is the real call_fn/agentic-engine glue for the three-mode eval's
        multi_turn/agentic modes (see `core.run.run_mode_eval`):
        `call_fn(messages) -> str`, same shape as
        `core.generate._llm_call()`'s return. The pipeline
        accepts the raw message list directly (it applies the chat template
        itself), same as `generate()`'s enable_thinking=None default path.
        """
        self._ensure()
        if self.seed is not None:
            import torch
            torch.manual_seed(self.seed)
        # Same enable_thinking handling as _render(): None -> hand the pipeline the
        # raw message list (it applies the chat template itself); explicit
        # True/False -> pre-render to a string with that flag forced (the pipeline
        # can't forward chat-template kwargs).
        if self.enable_thinking is None:
            inp = messages
        else:
            inp = self._tok.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
                enable_thinking=self.enable_thinking)
        r = self._pipe(inp, max_new_tokens=self.max_new_tokens, do_sample=True,
                       temperature=self.temperature, return_full_text=False,
                       num_return_sequences=1)
        return r[0]["generated_text"]


class UnslothGenerator(CandidateGenerator):
    """Unsloth/HF in-memory generator with the SAME batch API as VLLMGenerator
    (``generate_batch`` / ``chat_batch`` / ``chat``), for models vLLM can't serve.

    Reason it exists: a LoRA adapter on a fused-MoE base (Qwen3-Coder-30B-A3B, expert
    ``gate_up_proj``) hits ``vllm/lora/layers/fused_moe.py::set_lora`` -> ``AssertionError``
    (vLLM's fused-MoE-LoRA path can't consume peft's expert LoRA). This loads the base +
    adapter with Unsloth's ``FastModel`` (recreates the SAME fused representation the adapter
    was TRAINED under -> no shape mismatch, no merge, no extra disk) and generates with plain
    batched HF ``model.generate`` (left-padded, chunked to bound KV-cache memory). Slower than
    vLLM but correct; sampling is the same do_sample/top_k/top_p family so pass@k is comparable.

    ``model_id`` may be a base repo (no adapter) OR an adapter dir (Unsloth reads
    ``adapter_config``'s base_model and loads base+LoRA); pass ``adapter`` to load a base repo
    then attach an adapter dir. ``precision`` MUST be bf16 for Qwen3-MoE (fp16 is unstable)."""

    def __init__(self, model_id, adapter=None, precision="bf16", max_seq_len=6144,
                 max_new_tokens=4096, temperature=0.8, top_k=50, top_p=1.0,
                 batch_size=8, seed=None, enable_thinking=False):
        self.model_id = model_id
        self.adapter = adapter
        self.precision = precision
        self.max_seq_len = max_seq_len
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.batch_size = batch_size          # chunk size for batched generate (KV-cache bound)
        self.seed = seed
        self.enable_thinking = enable_thinking
        self._model = None
        self._tok = None

    def _ensure(self):
        if self._model is not None:
            return
        import torch
        from unsloth import FastModel
        dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}.get(self.precision, torch.bfloat16)
        # adapter dir alone loads base+LoRA; else load base then attach adapter.
        load_from = self.adapter if (self.adapter and not self.model_id) else self.model_id
        model, tok = FastModel.from_pretrained(
            load_from, max_seq_length=self.max_seq_len, dtype=dtype,
            load_in_4bit=False, load_in_8bit=False)
        if self.adapter and self.model_id and self.adapter != self.model_id:
            from peft import PeftModel
            model = PeftModel.from_pretrained(model, self.adapter)
        FastModel.for_inference(model)        # Unsloth 2x-faster inference path
        # left-pad so the newly generated tokens are a clean suffix per row in the batch
        tok.padding_side = "left"
        if tok.pad_token_id is None:
            tok.pad_token = tok.eos_token
        self._model, self._tok = model, tok

    def _render(self, msgs):
        kw = {} if self.enable_thinking is None else {"enable_thinking": self.enable_thinking}
        return self._tok.apply_chat_template(msgs, tokenize=False,
                                             add_generation_prompt=True, **kw)

    def _gen(self, rendered, max_new_tokens=None):
        """List[str rendered prompt] -> List[str completion], chunked batched generate."""
        import torch
        max_tok = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        if self.seed is not None:
            torch.manual_seed(self.seed)
        outs = []
        for i in range(0, len(rendered), self.batch_size):
            chunk = rendered[i:i + self.batch_size]
            enc = self._tok(chunk, return_tensors="pt", padding=True, truncation=True,
                            max_length=self.max_seq_len).to("cuda")
            with torch.no_grad():
                g = self._model.generate(
                    **enc, max_new_tokens=max_tok, do_sample=True,
                    temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
                    pad_token_id=self._tok.pad_token_id)
            plen = enc["input_ids"].shape[1]
            for j in range(len(chunk)):
                outs.append(self._tok.decode(g[j][plen:], skip_special_tokens=True))
        return outs

    def generate_batch(self, prompts, k, enable_thinking=_UNSET, max_new_tokens=None):
        self._ensure()
        rendered = [self._render([{"role": "user", "content": p}]) for p in prompts]
        result = [[] for _ in prompts]
        for _ in range(k):                    # k independent do_sample draws per prompt
            for idx, o in enumerate(self._gen(rendered, max_new_tokens)):
                result[idx].append(o)
        return result

    def generate(self, prompt, k):
        return self.generate_batch([prompt], k)[0]

    def chat_batch(self, conversations, max_new_tokens=None):
        self._ensure()
        rendered = [self._render(c) for c in conversations]
        return self._gen(rendered, max_new_tokens)

    def chat(self, messages):
        self._ensure()
        return self._gen([self._render(messages)])[0]


class VLLMGenerator(CandidateGenerator):
    """vLLM-backed generator that batches EVERY prompt (all tasks) x k samples through
    vLLM continuous batching in one call — the eval analog of a batched RL rollout.
    ~5-10x faster than the HF-pipeline LocalModelGenerator, which processes tasks
    sequentially.

    NOTE: this is a DIFFERENT generation implementation than LocalModelGenerator, so its
    pass@k etc. are not byte-identical to HF-eval'd base/SFT numbers. Re-baseline base/SFT
    under this engine before comparing across the ladder.
    """
    def __init__(self, model_id, temperature=0.8, max_new_tokens=4096, adapter=None,
                 precision="fp16", max_lora_rank=32, gpu_mem_util=0.55, max_seq_len=6144,
                 enable_thinking=False, seed=None, top_k=50, top_p=1.0,
                 intrinsic_grammar=None, intrinsic_mask=False):
        self.model_id = model_id
        self.temperature = temperature
        self.max_new_tokens = max_new_tokens
        self.adapter = adapter
        self.precision = precision
        self.max_lora_rank = max_lora_rank
        self.gpu_mem_util = gpu_mem_util
        self.max_seq_len = max_seq_len
        self.enable_thinking = enable_thinking
        # Optional xgrammar GBNF (from hexkernels.gym.intrinsic_grammar.build_intrinsic_grammar):
        # when set, structured decoding forbids hallucinated Q6_ intrinsic names in EVERY
        # generation (single-shot + repair turns). None -> unchanged default path.
        self.intrinsic_grammar = intrinsic_grammar
        # Faster alternative to the grammar: register the targeted logits processor (masks
        # ONLY inside Q6_ fragments -> ~full speed). Toggle per-call via `constrain_active`
        # (so ONE resident engine serves both the unconstrained and constrained arms).
        self.intrinsic_mask = intrinsic_mask
        self.constrain_active = False  # validated by the property below
        # top_k=50, top_p=1.0 MATCH the HF pipeline's sampling defaults (LocalModelGenerator) so
        # vLLM pass@k is comparable to the HF-eval'd base/SFT numbers (vLLM's default top_k=-1 =
        # full-vocab sampling gave ~4-5pt lower pass@8 by admitting more degenerate tokens).
        self.top_k = top_k
        self.top_p = top_p
        # seed=None -> vLLM sampling is diverse (matches HF's random do_sample). A fixed seed does
        # NOT make vLLM deterministic anyway (async continuous batching), so it bought false
        # reproducibility while cutting per-task diversity -> lower pass@8. Report >=2 eval seeds.
        self.seed = seed
        self._llm = None
        self._tok = None
        self._lora = None

    def _ensure(self):
        if self._llm is not None:
            return
        from vllm import LLM
        from transformers import AutoTokenizer
        dtype = {"fp16": "float16", "bf16": "bfloat16"}.get(self.precision, "float16")
        # vLLM ENGINE seed must be an int (None -> pydantic ValidationError). Use a fixed engine
        # seed; per-request DIVERSITY comes from SamplingParams(seed=None) below, so seed=None
        # (diverse sampling) is preserved without breaking engine init.
        kw = dict(model=self.model_id, dtype=dtype,
                  gpu_memory_utilization=self.gpu_mem_util,
                  max_model_len=self.max_seq_len,
                  seed=(self.seed if self.seed is not None else 0))
        if self.adapter:
            kw["enable_lora"] = True
            kw["max_lora_rank"] = self.max_lora_rank
        if self.intrinsic_mask:
            from hexkernels.core.intrinsic_logits_processor import IntrinsicMaskLogitsProcessor
            kw["logits_processors"] = [IntrinsicMaskLogitsProcessor]
        # Escape hatch for hosts where vLLM's auto-selected MoE kernel will not build.
        # On Blackwell (sm_120) the default picks FlashInfer CUTLASS, which JIT-compiles
        # `fused_moe_120` through ninja at engine init and fails -> the engine never starts.
        # HEXBENCH_MOE_BACKEND=triton routes around it. Unset => untouched auto behaviour,
        # so this cannot change results on hosts that already worked.
        import os
        moe_backend = os.environ.get("HEXBENCH_MOE_BACKEND")
        if moe_backend:
            kw["kernel_config"] = {"moe_backend": moe_backend}
        self._llm = LLM(**kw)
        self._tok = AutoTokenizer.from_pretrained(self.model_id)
        if self.adapter:
            from vllm.lora.request import LoRARequest
            self._lora = LoRARequest("eval_adapter", 1, self.adapter)

    def _render(self, prompt, enable_thinking=_UNSET):
        # enable_thinking: _UNSET -> use the instance default; None -> omit the kwarg
        # (let the chat template decide); True/False -> force. Per-prompt so ONE engine
        # load can render both the thinking and no-thinking arms (dual-thinking eval).
        et = self.enable_thinking if enable_thinking is _UNSET else enable_thinking
        msgs = [{"role": "user", "content": prompt}]
        kw = {}
        if et is not None:
            kw["enable_thinking"] = et
        return self._tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **kw)

    def _structured_field(self, grammar):
        """The SamplingParams kwarg for vLLM structured (grammar-constrained) decoding. The
        exact API name differs across vLLM versions -- confirmed on-box against vLLM 0.26.0,
        which uses ``structured_outputs=StructuredOutputsParams(grammar=...)``; older builds
        used ``guided_decoding=GuidedDecodingParams(grammar=...)``. This is the one switch."""
        try:
            from vllm.sampling_params import StructuredOutputsParams
            return {"structured_outputs": StructuredOutputsParams(grammar=grammar)}
        except Exception:
            pass
        try:
            from vllm.sampling_params import GuidedDecodingParams
        except Exception:
            from vllm import GuidedDecodingParams
        return {"guided_decoding": GuidedDecodingParams(grammar=grammar)}

    def _sampling_params(self, n, max_tok, **extra):
        """SamplingParams with the shared sampling config, attaching the intrinsic grammar
        when set. Single funnel for ALL generation (single-shot + every multiturn repair
        turn), so the constraint -- when enabled -- applies everywhere uniformly."""
        from vllm import SamplingParams
        kw = dict(n=n, temperature=self.temperature, top_k=self.top_k, top_p=self.top_p,
                  max_tokens=max_tok, seed=self.seed)
        kw.update(extra)
        if self.intrinsic_grammar:
            kw.update(self._structured_field(self.intrinsic_grammar))
        if self.intrinsic_mask and self.constrain_active:
            kw["extra_args"] = {"constrain_intrinsics": True}
        return SamplingParams(**kw)

    # --- constrained decoding: registration vs arming -----------------------
    # Two flags are deliberate, not redundant. `intrinsic_mask` REGISTERS the
    # logits processor with the vLLM engine and can only be set at construction
    # (engine init). `constrain_active` ARMS it per call, so ONE resident engine
    # serves both the constrained and unconstrained arms of an eval.
    #
    # The failure mode that motivated this guard: setting `constrain_active =
    # True` on a generator built with `intrinsic_mask=False` is a SILENT no-op.
    # No processor was ever registered, `extra_args` is never attached, and the
    # run generates completely unconstrained while every log line and results
    # field says constrained decoding was ON. There is no downstream signal --
    # both configurations produce 0 fabricated intrinsics on most prompts, so
    # the difference is invisible in the output.
    #
    # Arming what was never registered now raises instead.
    @property
    def constrain_active(self):
        """True iff the intrinsic mask is armed for subsequent generations."""
        return self._constrain_active

    @constrain_active.setter
    def constrain_active(self, value):
        if value and not self.intrinsic_mask:
            raise ValueError(
                "constrain_active=True requires intrinsic_mask=True at construction: "
                "the logits processor is registered at vLLM engine init and cannot be "
                "added later. Arming it here would silently generate UNCONSTRAINED "
                "while reporting constrained decoding as ON. Rebuild the generator "
                "with VLLMGenerator(..., intrinsic_mask=True)."
            )
        self._constrain_active = bool(value)

    @property
    def constraining(self) -> bool:
        """Effective state -- registered AND armed. Log this, never the flags
        individually, or a run can report a constraint it is not applying."""
        return bool(self.intrinsic_mask and self._constrain_active)

    def generate_batch(self, prompts, k, enable_thinking=_UNSET, max_new_tokens=None):
        """List[prompt] -> List[List[str]] (k completions per prompt). ALL prompts x k
        go through vLLM in one continuous-batched call.

        enable_thinking / max_new_tokens override the instance defaults for THIS call
        only (the engine is already resident, so the two thinking arms of a dual eval
        share one weight load -- issue one generate_batch per arm with its own budget)."""
        self._ensure()
        max_tok = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        sp = self._sampling_params(k, max_tok)
        rendered = [self._render(p, enable_thinking) for p in prompts]
        kw = {"lora_request": self._lora} if self._lora else {}
        outs = self._llm.generate(rendered, sp, **kw)   # order-preserving
        return [[o.text for o in r.outputs] for r in outs]

    def generate_batch_budget_think(self, prompts, k, think_budget, answer_budget):
        """Budget-forced thinking. Two phases on the SAME resident engine:

          1. THINK: render with enable_thinking=True, generate with stop="</think>" and a
             hard `think_budget` cap.
          2. ANSWER: if the model closed </think> on its own, continue from there; if it hit
             the cap without closing, INJECT "</think>" first -- then generate the code with
             its own `answer_budget`.

        Bounds each completion to ~think_budget+answer_budget tokens and GUARANTEES a code
        answer. Without this, non-reasoning models (base/SFT, not RL-tuned reasoners) forced
        to think ramble to the cap and emit NO code (measured: 99% of base samples never
        closed </think> at an 8192 cap). Returns List[List[str]] (k completions per prompt),
        each = "<think>...</think>\n\n<answer>", so downstream code-extraction/think-strip work.
        """
        self._ensure()
        lora_kw = {"lora_request": self._lora} if self._lora else {}
        rendered = [self._render(p, enable_thinking=True) for p in prompts]
        # phase 1 -- think, capped, stop at the close tag (kept in the output)
        sp1 = self._sampling_params(k, think_budget, stop=["</think>"],
                                    include_stop_str_in_output=True)
        outs1 = self._llm.generate(rendered, sp1, **lora_kw)
        # build phase-2 continuations (force-close any think block that hit the cap)
        p2_prompts, think_parts, index = [], [], []
        for i, r in enumerate(outs1):
            for j, o in enumerate(r.outputs):
                th = o.text
                if "</think>" not in th:
                    th = th + "\n</think>\n\n"
                think_parts.append(th)
                p2_prompts.append(rendered[i] + th)
                index.append((i, j))
        # phase 2 -- answer (code), one continuation per (prompt, sample)
        sp2 = self._sampling_params(1, answer_budget)
        outs2 = self._llm.generate(p2_prompts, sp2, **lora_kw)
        result = [[None] * k for _ in prompts]
        for (i, j), th, o2 in zip(index, think_parts, outs2):
            result[i][j] = th + o2.outputs[0].text
        return result

    def generate(self, prompt, k):
        return self.generate_batch([prompt], k)[0]

    def chat(self, messages):
        """ONE (n=1) completion from a full chat-messages list, on the resident
        vLLM engine. The real call_fn/agentic-engine glue for the three-mode
        eval's multi_turn/agentic modes (see `core.run.run_mode_eval`);
        mirrors `core.generate._llm_call()`'s glue (reaches
        into `_llm`/`_tok`/`_lora` directly -- there is no public multi-turn
        chat API on this class) as a reusable instance method so `run.py` can
        share ONE resident engine across single_shot's batched generation and
        the box-only multi_turn/agentic per-task calls.
        """
        self._ensure()
        kw = {}
        if self.enable_thinking is not None:
            kw["enable_thinking"] = self.enable_thinking
        rendered = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, **kw)
        sp = self._sampling_params(1, self.max_new_tokens)
        lora_kw = {"lora_request": self._lora} if self._lora else {}
        outs = self._llm.generate([rendered], sp, **lora_kw)
        return outs[0].outputs[0].text

    def chat_batch(self, conversations, max_new_tokens=None):
        """List[messages] -> List[str]: ONE completion per conversation, ALL batched
        through vLLM continuous batching in a SINGLE generate call (order-preserving).

        The batched analog of `chat()`, for the breadth-first multi_turn/agentic mode
        drivers: advance every active rollout's current turn in one call instead of a
        per-rollout Python loop (which serializes generation to batch-size-1). Same
        per-request SamplingParams and LoRA/enable_thinking handling as `chat()`."""
        self._ensure()
        max_tok = self.max_new_tokens if max_new_tokens is None else max_new_tokens
        kw = {}
        if self.enable_thinking is not None:
            kw["enable_thinking"] = self.enable_thinking
        rendered = [self._tok.apply_chat_template(
            c, tokenize=False, add_generation_prompt=True, **kw) for c in conversations]
        sp = self._sampling_params(1, max_tok)
        lora_kw = {"lora_request": self._lora} if self._lora else {}
        outs = self._llm.generate(rendered, sp, **lora_kw)   # order-preserving
        return [o.outputs[0].text for o in outs]
