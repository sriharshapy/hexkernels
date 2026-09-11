"""Op source: PyTorch's operator schema registry.

The first of several sources behind one interface (see forge v2 design §6). It
enumerates every schema the installed torch exposes and parses each one at the
STRING level -- it never imports an operator, never touches CUDA, and never
executes anything.

PROVENANCE. Every row carries the verbatim schema string plus the source name,
torch version and build. The observed schema COUNT is recorded as data, never
asserted: the same version string yields different counts on different builds
(4,409 vs 4,394 have both been observed on "2.7.1"), because the count depends
on which optional backends are importable. A pinned total is a reproducibility
bug, not a guard.

SCHEMA GRAMMAR, from the strings themselves:

    namespace::op[.overload](arg, ...) -> returns

    aten::std(Tensor self, bool unbiased=True) -> Tensor
    aten::copy_(Tensor(a!) self, Tensor src, bool non_blocking=False) -> Tensor(a!)
    aten::_weight_norm_interface.out(Tensor v, ..., *, Tensor(a!) out0) -> (Tensor(a!), Tensor(b!))
    aten::dsplit.int(Tensor(a -> *) self, int sections) -> Tensor(a)[]
    aten::update.Tensor(Dict(Tensor, t)(a!) self, Dict(Tensor, t)(a!) to_add) -> ()

Notation that matters for parsing:
  * ``Tensor(a!)``  -- mutable alias; the ``!`` marks in-place mutation
  * ``Tensor?``     -- optional
  * ``Tensor[]``    -- list
  * ``*``           -- keyword-only separator, not an argument
  * defaults follow ``=``
  * commas appear INSIDE ``Dict(Tensor, t)`` and inside alias annotations like
    ``(a -> *)``, so argument splitting must be depth-aware. Naive
    ``split(",")`` mis-parses those and silently corrupts the row.
"""
import re

SOURCE_NAME = "torch_registry"


def _split_top_level(text, sep=","):
    """Split on `sep` at bracket depth zero. `Dict(Tensor, t)` stays one field."""
    out, depth, cur = [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        out.append(tail)
    return out


def _match_paren(text, start):
    """Index of the ')' matching the '(' at `start`, or -1."""
    depth = 0
    for i in range(start, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


_TENSORY = re.compile(r"\bTensor\b")


def parse_arg(text):
    """One argument declaration -> dict. `text` has no surrounding whitespace."""
    default = None
    # Split the default off at top level: `int dim=0`, `ScalarType? dtype=None`.
    parts = _split_top_level(text, "=")
    if len(parts) > 1:
        text, default = parts[0].strip(), "=".join(parts[1:]).strip()
    # The name is the last whitespace-separated token; everything before is the type.
    bits = text.rsplit(" ", 1)
    if len(bits) == 2:
        type_s, name = bits[0].strip(), bits[1].strip()
    else:
        type_s, name = text.strip(), ""
    return {
        "name": name,
        "type": type_s,
        "default": default,
        "is_tensor": bool(_TENSORY.search(type_s)),
        "is_mutable": "!" in type_s,
        "is_optional": "?" in type_s,
        "is_list": "[]" in type_s,
    }


def parse_schema(raw):
    """Parse one schema string into a row. Returns None if it does not parse.

    Never raises: an unparseable schema is recorded as such by the caller rather
    than crashing a 4,000-row harvest.
    """
    try:
        open_i = raw.index("(")
    except ValueError:
        return None
    close_i = _match_paren(raw, open_i)
    if close_i < 0:
        return None

    qualified = raw[:open_i].strip()
    if "::" not in qualified:
        return None
    namespace, rest = qualified.split("::", 1)
    op, _, overload = rest.partition(".")

    args = []
    kwonly = False
    for field in _split_top_level(raw[open_i + 1:close_i]):
        if field == "*":
            kwonly = True
            continue
        if not field:
            continue
        a = parse_arg(field)
        a["kwonly"] = kwonly
        args.append(a)

    after = raw[close_i + 1:]
    ret_s = after.split("->", 1)[1].strip() if "->" in after else ""
    inner = ret_s[1:-1] if ret_s.startswith("(") and ret_s.endswith(")") else ret_s
    returns = [{"type": t, "is_tensor": bool(_TENSORY.search(t))}
               for t in _split_top_level(inner) if t]

    return {
        "raw": raw,
        "namespace": namespace,
        "op": op,
        "overload": overload,
        "args": args,
        "returns": returns,
        # In-place is read from the SCHEMA, not from the name. The signature of
        # an in-place op is `Tensor(a!) self, ... -> Tensor(a!)`: it takes a
        # mutable alias as its first argument and hands the same alias back.
        #
        # The previous rule, `op.endswith("_") and not op.startswith("__")`, was
        # a name heuristic with a dunder exemption added to protect `__and__`
        # (which ends in `_` and is not in-place). That exemption also swallowed
        # the `__i*__` family -- `__iand__`, `__ilshift__`, `__ixor__` and
        # friends are Python's in-place operator protocol and ARE mutating, so
        # they were being classified as ordinary forward kernels and would have
        # entered the corpus as such. The schema says so directly; the name only
        # correlates.
        "in_place": bool(
            args and args[0]["is_tensor"] and args[0]["is_mutable"]
            and any(r["is_tensor"] and "!" in r["type"] for r in returns)),
        "out_variant": overload == "out" or any(
            a["is_mutable"] and a["name"].startswith("out") for a in args),
        "has_tensor_in": any(a["is_tensor"] for a in args),
        "has_tensor_out": any(r["is_tensor"] for r in returns),
    }


ACCESSOR = "torch._C._jit_get_all_schemas()"


def provenance():
    """The per-row provenance stamp: WHERE this op came from and from WHICH build.

    Carried on every row, not just in a side file. A catalogue row that has been
    copied, filtered or merged downstream must still answer "which torch, which
    install, which accessor" on its own -- provenance in a separate manifest is
    provenance that gets separated.
    """
    import os
    import torch
    return {
        "source": SOURCE_NAME,
        "torch_version": torch.__version__,
        "torch_path": os.path.dirname(os.path.abspath(torch.__file__)),
        "accessor": ACCESSOR,
    }


def environment():
    """Source-level record. Recorded as data -- never asserted by a test."""
    env = provenance()
    env["citation"] = (f"{ACCESSOR} -- the operator schema registry of the installed "
                       f"torch build at {env['torch_path']}")
    return env


def raw_schemas():
    """Every schema string the installed torch exposes, sorted for determinism."""
    import torch
    return sorted(str(s) for s in torch._C._jit_get_all_schemas())


def rows():
    """Parsed rows, in sorted-by-raw order, each stamped with its provenance.

    Unparseable schemas are yielded with ``parse_error`` set rather than dropped
    -- nothing vanishes silently.
    """
    prov = provenance()
    out = []
    for raw in raw_schemas():
        row = parse_schema(raw) or {"raw": raw, "parse_error": True}
        row.update(prov)
        out.append(row)
    return out
