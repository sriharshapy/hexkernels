"""Constrained-intrinsic decoding: forbid hallucinated Q6_ names at generation time.

100% of the model's holdout no-compiles are fabricated HVX intrinsics -- ``call to
undeclared function 'Q6_...'`` for a name not in the SDK. This module makes such names
IMPOSSIBLE to emit: it turns the valid-name set (shared with :mod:`intrinsic_repair`, one
dictionary) into a vLLM structured-decoding grammar so every ``Q6_`` identifier the model
writes must be a real intrinsic. Purely inference-time; nothing is trained.

Pieces:
- :func:`output_has_only_valid_intrinsics` -- the accept/reject ORACLE = the language def.
- :func:`build_intrinsic_grammar` -- xgrammar GBNF for vLLM structured decoding (primary).
- :class:`IntrinsicTrie` -- prefix-trie masking decision for a logits-processor FALLBACK.
- :func:`training_subset` -- the ~204-name "seen" set (ablation / feasibility fallback).

Spec: docs/superpowers/specs/2026-07-30-constrained-intrinsic-decoding-design.md
"""
import glob
import re

# Re-export so callers get the valid-name set from ONE place (fail-closed, header-derived).
from hexkernels.gym.intrinsic_repair import valid_intrinsics  # noqa: F401

# A maximal C identifier that starts with Q6_ (greedy: a real name + trailing identifier
# chars, e.g. Q6_Vh_vadd_VhVhX, is a DIFFERENT, undeclared name -> flagged).
_Q6 = re.compile(r"\bQ6_[A-Za-z0-9_]+\b")
# Any maximal C identifier (independent tokenizer for the grammar reference acceptor).
_IDENT = re.compile(r"[A-Za-z0-9_]+")

#: sentinel returned by :meth:`IntrinsicTrie.allowed_after` meaning "a complete valid name
#: ends here, so a word boundary (ending the intrinsic) is permitted".
END = "<END>"


def output_has_only_valid_intrinsics(text, valid):
    """True iff every maximal ``Q6_`` identifier in ``text`` is in ``valid``.

    This IS the language the grammar / trie must realize; the on-box A/B re-checks every
    generated kernel with this oracle to prove the constraint bit."""
    return all(m in valid for m in _Q6.findall(text or ""))


def training_subset(paths):
    """Set of every ``Q6_`` name appearing in the given corpus files (globs of jsonl/c).

    The ~204-name "seen" set -- an opt-in ablation / feasibility fallback, NEVER the default
    allowed set (using it at inference would leak train-corpus vocabulary into holdout)."""
    names = set()
    for p in paths:
        for f in glob.glob(p):
            try:
                with open(f, encoding="utf-8", errors="ignore") as fh:
                    names.update(_Q6.findall(fh.read()))
            except OSError:
                continue
    return names


# GBNF for xgrammar (vLLM structured decoding). The output is whitespace/punctuation
# ("non-word") interleaved with maximal word tokens, and every two word tokens are separated
# by >=1 non-word char (so a valid name cannot be silently extended -- Q6_..VhX has no
# parse). A word token is EITHER a real intrinsic (Q6_ + a valid suffix) OR an identifier
# that does NOT start with "Q6_" (idnq). Anything starting "Q6_" therefore has a parse ONLY
# as a listed intrinsic -> fabricated names are rejected by construction.
_GRAMMAR_TEMPLATE = """root ::= ws ( word-tok ( sep word-tok )* )? ws
word-tok ::= intrinsic | idnq
sep ::= non-word+
ws ::= non-word*
non-word ::= [^A-Za-z0-9_]
wc ::= [A-Za-z0-9_]
nq ::= [0-9A-PR-Za-z_]
n6 ::= [0-57-9A-Za-z_]
nu ::= [0-9A-Za-z]
idnq ::= nq wc* | "Q" | "Q" n6 wc* | "Q6" | "Q6" nu wc*
intrinsic ::= "Q6_" ( %(alt)s )
"""


def build_intrinsic_grammar(valid):
    """GBNF (xgrammar) string constraining every ``Q6_`` identifier to a name in ``valid``.

    Fail-closed: an empty set raises rather than yielding an unconstrained grammar."""
    if not valid:
        raise ValueError("empty valid-intrinsic set; refusing to build an unconstrained grammar")
    suffixes = sorted(n[3:] for n in valid if n.startswith("Q6_"))
    if not suffixes:
        raise ValueError("valid set has no Q6_ names")
    alt = " | ".join('"%s"' % s for s in suffixes)
    return _GRAMMAR_TEMPLATE % {"alt": alt}


_TRIE_HEAD = """root ::= ws ( word-tok ( sep word-tok )* )? ws
word-tok ::= intrinsic | idnq
sep ::= non-word+
ws ::= non-word*
non-word ::= [^A-Za-z0-9_]
wc ::= [A-Za-z0-9_]
nq ::= [0-9A-PR-Za-z_]
n6 ::= [0-57-9A-Za-z_]
nu ::= [0-9A-Za-z]
idnq ::= nq wc* | "Q" | "Q" n6 wc* | "Q6" | "Q6" nu wc*
intrinsic ::= "Q6_" %(root)s
"""


def build_intrinsic_grammar_trie(valid):
    """Like :func:`build_intrinsic_grammar` (SAME language) but emits the intrinsic name set
    as a shared-prefix CHAR TRIE of GBNF rules instead of a flat 1392-way string alternation.

    Rationale (measured on-box): xgrammar's per-token cost blows up with a large flat
    alternation (1392 names -> ~175x slowdown) but not with a small one (201 -> 2.15x). HVX
    names share massive prefixes, so a trie collapses the branching -> far smaller automaton,
    usable speed at the full 1392. Fail-closed on empty."""
    if not valid:
        raise ValueError("empty valid-intrinsic set; refusing to build an unconstrained grammar")
    suffixes = [n[3:] for n in valid if n.startswith("Q6_")]
    if not suffixes:
        raise ValueError("valid set has no Q6_ names")
    root = {}
    for s in suffixes:
        node = root
        for ch in s:
            node = node.setdefault(ch, {})
        node[END] = True

    rules = []          # (name, body) in emission order
    counter = [0]

    def emit(node):
        name = "t%d" % counter[0]
        counter[0] += 1
        opts = []
        for ch in sorted(k for k in node if k != END):
            child = node[ch]
            lit = '"%s"' % ch
            child_kids = any(k != END for k in child)
            if child.get(END) and not child_kids:
                opts.append(lit)                       # leaf: consume char, stop
            else:
                opts.append("%s %s" % (lit, emit(child)))
        if node.get(END) and node is not root:
            opts.append('""')                          # terminal interior node may stop here
        rules.append((name, " | ".join(opts)))
        return name

    root_name = emit(root)
    body = _TRIE_HEAD % {"root": root_name}
    body += "".join("%s ::= %s\n" % (n, b) for n, b in rules)
    return body


def _grammar_accepts(text, valid):
    """Pure-Python reference for the grammar's language, INDEPENDENT of the oracle's regex
    (tokenizes into maximal identifiers instead of matching Q6_ runs). Used to test the
    builder offline without xgrammar; the on-box prototype confirms xgrammar reproduces it."""
    return all(not t.startswith("Q6_") or t in valid for t in _IDENT.findall(text or ""))


def masked_token_ids(trie, tail_text, single_char_ids, boundary_ids):
    """Allowed next-token ids when the tail is INSIDE a ``Q6_`` identifier, else ``None``.

    Char-level constraint for a logits-processor: inside a fragment, only single-character
    word tokens that continue a valid name are allowed (plus, at a complete name, any boundary
    token that ends the identifier). ~99% of tokens are outside any fragment -> returns None ->
    no masking -> full speed (this is why it beats the whole-output grammar). Pure/testable.

    ``single_char_ids``: {char -> token_id} for 1-char word tokens.
    ``boundary_ids``: iterable of token ids that begin with a non-word char (end an identifier).
    """
    allowed = trie.allowed_after(tail_text)
    if allowed is None:
        return None
    ids = [single_char_ids[c] for c in allowed if c != END and c in single_char_ids]
    if END in allowed:
        ids = ids + list(boundary_ids)
    return ids


_WORDCH = re.compile(r"[A-Za-z0-9_]")


def _token_consistent(node, s):
    """True if appending token string ``s`` at trie ``node`` stays on a valid-name path:
    consume ``s`` char by char; if we leave the trie, that's OK ONLY when the current node is a
    completed name (END) and the leaving char is a non-word boundary (the name ended, the rest of
    the token is free text)."""
    cur = node
    for ch in s:
        nxt = cur.get(ch)
        if nxt is None:
            return bool(cur.get(END)) and not _WORDCH.match(ch)
        cur = nxt
    return True


def masked_token_ids_tokenlevel(trie, tail_text, id2str, first_char_index, boundary_ids):
    """Allowed next-token ids inside a ``Q6_`` fragment, TOKEN-level (keeps the model's natural
    multi-char tokens -> no char-by-char re-sampling -> no name drift), or ``None`` if outside a
    fragment.

    ``id2str``: list token_id -> decoded string. ``first_char_index``: {first_char -> [ids]}.
    ``boundary_ids``: ids whose first char is a non-word char (end an identifier)."""
    m = re.search(r"[A-Za-z0-9_]*$", tail_text or "")
    frag = m.group() if m else ""
    if not frag.startswith("Q6_"):
        return None
    node = trie.node_at(frag)
    if node is None:
        return None  # defensive: not a valid prefix -> don't mask (avoid deadlock)
    cand = set()
    for ch in node:
        if ch == END:
            continue
        cand.update(first_char_index.get(ch, ()))
    if node.get(END):
        cand.update(boundary_ids)
    return [t for t in cand if _token_consistent(node, id2str[t])]


class IntrinsicTrie:
    """Prefix trie over the valid names for a logits-processor fallback (used only if the
    grammar path fails the on-box feasibility check). Its decision is pure and GPU-free."""

    def __init__(self, valid):
        self.root = {}
        for name in valid:
            node = self.root
            for ch in name:
                node = node.setdefault(ch, {})
            node[END] = True

    def node_at(self, frag):
        """The trie node reached by consuming string ``frag`` from the root, or None if ``frag``
        is not a valid prefix of any name."""
        node = self.root
        for ch in frag:
            node = node.get(ch)
            if node is None:
                return None
        return node

    def allowed_after(self, prefix):
        """Given text already emitted, return the set of legal next characters when the tail
        is inside an in-progress ``Q6_`` identifier (``END`` in the set = the fragment is a
        complete valid name, so a boundary may follow), or ``None`` when not constrained."""
        m = re.search(r"[A-Za-z0-9_]*$", prefix or "")
        frag = m.group() if m else ""
        if not frag.startswith("Q6_"):
            return None
        node = self.root
        for ch in frag:
            node = node.get(ch)
            if node is None:
                return set()          # dead end: fabricated prefix -> nothing legal
        out = {c for c in node if c != END}
        if node.get(END):
            out.add(END)
        return out
