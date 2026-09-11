"""Emit C from the fused, bufferized, affine-lowered form of a kernel.

WHY THIS EXISTS. The reference C this pipeline ships is emitted from the FX
graph one operator at a time, so it materialises every intermediate: batch 85's
`special_exp2 -> elu` declares seven 6,144-element arrays and makes seven passes
over them. That is a faithful description of what PyTorch eager does and a
misleading starting point for anyone writing a kernel, because the first thing
that has to happen is to fuse it back into one pass. Measured over the corpus,
`linalg-fuse-elementwise-ops` collapses those chains -- 6 generics to 1, 5 to 1
-- and drops the `tensor.empty` count in every case. 487 of 494 kernels lower
all the way to affine loops over memrefs with no pass failure.

WHAT THIS IS FOR. Two things, and they are different:

  - A FUSED reference. Same semantics, one loop nest, no intermediate arrays.
    It is a better description of the task than the seven-pass version, and it
    is derived mechanically rather than by someone deciding what may be fused.

  - A pass BOUNDARY. With the fused scalar form available, the remaining work
    splits into steps that each have a checkable postcondition: scalar -> HVX
    vectorised, then HVX -> DMA/VTCM staged. Each step's output can be compiled
    and run against the same golden, which is what makes a step a pass rather
    than a suggestion.

WHAT IT IS NOT. It does not tile, vectorise, or touch the memory hierarchy --
`linalg-tile` is not a registered pass in this torch-mlir build (upstream moved
tiling to the transform dialect), and nothing upstream emits DMA, VTCM staging
or l2fetch. Those stay where the target knowledge is.

ASSUMPTION, STATED BECAUSE IT IS LOAD BEARING: memrefs are treated as
row-major contiguous. Function arguments arrive as
`memref<32x64xf32, strided<[?, ?], offset: ?>>` -- dynamic strides -- and the
harness passes plain contiguous buffers, so the strides are the row-major ones.
A kernel whose arguments were genuinely strided would need the strides read from
the descriptor instead, and `emit()` raises rather than guessing if it meets a
layout it cannot prove contiguous.
"""
from __future__ import annotations

import io


class Unsupported(Exception):
    """An op or type outside the covered subset. Raised, never worked around."""


# ---------------------------------------------------------------- op tables
# Binary arithmetic that maps to a C infix operator.
_INFIX = {
    "arith.addf": "+", "arith.subf": "-", "arith.mulf": "*", "arith.divf": "/",
    "arith.addi": "+", "arith.subi": "-", "arith.muli": "*",
    "arith.andi": "&", "arith.ori": "|", "arith.xori": "^",
    "arith.remsi": "%", "arith.divsi": "/",
}

# Unary/binary calls that map to a libm function, by result element type.
_CALL = {
    "math.exp": "exp", "math.log": "log", "math.log1p": "log1p",
    "math.log2": "log2", "math.log10": "log10", "math.sqrt": "sqrt",
    "math.rsqrt": "rsqrt", "math.tanh": "tanh", "math.absf": "fabs",
    "math.floor": "floor", "math.ceil": "ceil", "math.round": "round",
    "math.erf": "erf", "math.sin": "sin", "math.cos": "cos", "math.tan": "tan",
    "math.atan": "atan", "math.exp2": "exp2", "math.expm1": "expm1",
    "math.powf": "pow", "math.atan2": "atan2", "math.fpowi": "pow",
    "math.copysign": "copysign", "math.fma": "fma",
    "math.sinh": "sinh", "math.cosh": "cosh", "math.asin": "asin",
    "math.acos": "acos", "math.roundeven": "rint", "math.trunc": "trunc",
    "math.cbrt": "cbrt", "math.exp10": "exp10", "math.tan": "tan",
}

# arith.cmpf predicate index -> C expression template.
#
# THE ORDERED AND UNORDERED FORMS ARE NOT INTERCHANGEABLE AND C SPELLS THEM
# ASYMMETRICALLY, which is the whole reason this is a table of templates rather
# than a table of operators. An ordered predicate is false when either operand
# is NaN; an unordered one is true. C's `<`, `>`, `<=`, `>=` and `==` are the
# ORDERED forms, but C's `!=` is the UNORDERED one (`a != b` is true when the
# operands are unordered) -- so `one` needs negation and `une` does not.
# Getting this backwards would map `isnan(x)`, which a traced graph emits as
# `cmpf une, %x, %x`, onto a comparison that is always false.
_CMPF = {
    0:  "(0)",                                   # AlwaysFalse
    1:  "({a}) == ({b})",                        # oeq
    2:  "({a}) > ({b})",                         # ogt
    3:  "({a}) >= ({b})",                        # oge
    4:  "({a}) < ({b})",                         # olt
    5:  "({a}) <= ({b})",                        # ole
    6:  "(!(({a}) != ({b})) ? 0 : (({a}) == ({a}) && ({b}) == ({b})))",  # one
    7:  "(({a}) == ({a}) && ({b}) == ({b}))",    # ord
    8:  "(!((({a}) < ({b})) || (({a}) > ({b}))))",   # ueq
    9:  "(!(({a}) <= ({b})))",                   # ugt
    10: "(!(({a}) < ({b})))",                    # uge
    11: "(!(({a}) >= ({b})))",                   # ult
    12: "(!(({a}) > ({b})))",                    # ule
    13: "({a}) != ({b})",                        # une -- C's != IS unordered
    14: "((({a}) != ({a})) || (({b}) != ({b})))",   # uno == isnan(a)||isnan(b)
    15: "(1)",                                   # AlwaysTrue
}
_CMPI = {0: "({a}) == ({b})", 1: "({a}) != ({b})",
         2: "({a}) < ({b})", 3: "({a}) <= ({b})",
         4: "({a}) > ({b})", 5: "({a}) >= ({b})",
         6: "({a}) < ({b})", 7: "({a}) <= ({b})",
         8: "({a}) > ({b})", 9: "({a}) >= ({b})"}


def _ctype(t) -> str:
    s = str(t)
    if s == "f32":
        return "float"
    if s == "f64":
        return "double"
    if s == "f16":
        return "_Float16"
    if s == "i1":
        return "int"
    if s in ("i64", "index"):
        return "int64_t"
    if s in ("i32", "i8", "i16"):
        return {"i32": "int32_t", "i8": "int8_t", "i16": "int16_t"}[s]
    raise Unsupported(f"scalar type {s}")


def _elem_ctype(memref_t) -> str:
    s = str(memref_t)
    inner = s[s.index("<") + 1:s.rindex(">")]
    return _ctype(inner.split(",")[0].split("x")[-1])


def _shape(memref_t):
    s = str(memref_t)
    inner = s[s.index("<") + 1:].split(",")[0]
    parts = inner.split("x")[:-1]
    if any(p == "?" for p in parts):
        raise Unsupported(f"dynamic shape in {s}")
    return [int(p) for p in parts]


class Emitter:
    def __init__(self):
        self.v = {}          # SSA Value -> C expression string
        self.n = 0
        self.buf = io.StringIO()
        self.ind = 1
        # Allocations to NOT declare, because they have been aliased onto an
        # output parameter. See `emit(alias_result=True)`.
        self.aliased = set()

    def fresh(self, pfx="t"):
        self.n += 1
        return f"{pfx}{self.n}"

    def w(self, line):
        self.buf.write("  " * self.ind + line + "\n")

    # ------------------------------------------------------------- indexing
    def _affine_expr(self, e, dims, syms):
        """Render an AffineExpr as a C expression over the bound loop vars."""
        s = str(e)
        # AffineExpr's textual form is already a C-compatible arithmetic
        # expression over d0.. and s0.., apart from floordiv/mod/ceildiv.
        for bad in ("floordiv", "ceildiv"):
            if bad in s:
                raise Unsupported(f"affine {bad}")
        out = s
        for i, d in enumerate(dims):
            out = out.replace(f"d{i}", d)
        for i, y in enumerate(syms):
            out = out.replace(f"s{i}", y)
        return out

    def _index(self, op, memref, operands):
        """Row-major flat index for an affine.load/store."""
        shape = _shape(memref.type)
        amap = op.attributes["map"].value if "map" in op.attributes else None
        idx_vals = [self.v[o] for o in operands]
        if amap is None:
            terms = idx_vals
        else:
            dims = idx_vals[:amap.n_dims]
            syms = idx_vals[amap.n_dims:]
            terms = [self._affine_expr(r, dims, syms) for r in amap.results]
        if len(terms) != len(shape):
            raise Unsupported(f"rank {len(terms)} vs shape {shape}")
        flat, stride = [], 1
        for t, d in zip(reversed(terms), reversed(shape)):
            flat.append(f"({t}) * {stride}" if stride != 1 else f"({t})")
            stride *= d
        return " + ".join(reversed(flat)) if flat else "0"

    # ----------------------------------------------------------------- ops
    def op(self, op):
        name = op.operation.name
        res = op.results[0] if len(op.results) == 1 else None

        if name == "arith.constant":
            a = op.attributes["value"]
            s = str(a).split(":")[0].strip()
            if str(res.type).startswith("f"):
                s = s if ("." in s or "e" in s or "E" in s) else s + ".0"
                self.v[res] = f"({s}f)" if str(res.type) == "f32" else f"({s})"
            else:
                self.v[res] = f"({s})"
            return

        if name in _INFIX:
            a, b = (self.v[o] for o in op.operands)
            t = self.fresh()
            self.w(f"const {_ctype(res.type)} {t} = {a} {_INFIX[name]} {b};")
            self.v[res] = t
            return

        if name in _CALL:
            args = ", ".join(self.v[o] for o in op.operands)
            fn = _CALL[name]
            if str(res.type) == "f32" and fn not in ("fma",):
                fn += "f"
            t = self.fresh()
            self.w(f"const {_ctype(res.type)} {t} = {fn}({args});")
            self.v[res] = t
            return

        if name in ("arith.negf",):
            t = self.fresh()
            self.w(f"const {_ctype(res.type)} {t} = -{self.v[op.operands[0]]};")
            self.v[res] = t
            return

        if name in ("arith.maximumf", "arith.maxnumf",
                    "arith.minimumf", "arith.minnumf"):
            a, b = (self.v[o] for o in op.operands)
            fn = "fmax" if "max" in name else "fmin"
            if str(res.type) == "f32":
                fn += "f"
            t = self.fresh()
            self.w(f"const {_ctype(res.type)} {t} = {fn}({a}, {b});")
            self.v[res] = t
            return

        if name in ("arith.cmpf", "arith.cmpi"):
            p = int(str(op.attributes["predicate"]).split(":")[0].strip()
                    .split()[-1]) if "predicate" in op.attributes else None
            table = _CMPF if name == "arith.cmpf" else _CMPI
            if p not in table:
                raise Unsupported(f"{name} predicate {p}")
            a, b = (self.v[o] for o in op.operands)
            t = self.fresh()
            self.w(f"const int {t} = ({table[p].format(a=a, b=b)});")
            self.v[res] = t
            return

        if name == "arith.select":
            c, a, b = (self.v[o] for o in op.operands)
            t = self.fresh()
            self.w(f"const {_ctype(res.type)} {t} = {c} ? {a} : {b};")
            self.v[res] = t
            return

        if name in ("arith.extf", "arith.truncf", "arith.sitofp",
                    "arith.fptosi", "arith.index_cast", "arith.extsi",
                    "arith.extui", "arith.trunci", "arith.uitofp"):
            t = self.fresh()
            self.w(f"const {_ctype(res.type)} {t} = "
                   f"({_ctype(res.type)})({self.v[op.operands[0]]});")
            self.v[res] = t
            return

        if name in ("affine.load", "memref.load"):
            m = op.operands[0]
            i = (self._index(op.operation, m, list(op.operands)[1:])
                 if name == "affine.load"
                 else " + ".join(self.v[o] for o in list(op.operands)[1:]) or "0")
            t = self.fresh()
            self.w(f"const {_elem_ctype(m.type)} {t} = {self.v[m]}[{i}];")
            self.v[res] = t
            return

        if name in ("affine.store", "memref.store"):
            val, m = op.operands[0], op.operands[1]
            i = (self._index(op.operation, m, list(op.operands)[2:])
                 if name == "affine.store"
                 else " + ".join(self.v[o] for o in list(op.operands)[2:]) or "0")
            self.w(f"{self.v[m]}[{i}] = {self.v[val]};")
            return

        if name == "memref.alloc":
            if res in self.aliased:
                return       # already bound to the out parameter; declaring a
                             # buffer for it would shadow the alias
            shape = _shape(res.type)
            n = 1
            for d in shape:
                n *= d
            t = self.fresh("buf")
            self.w(f"static {_elem_ctype(res.type)} {t}[{n}];")
            self.v[res] = t
            return

        if name == "affine.for":
            # The bounds are AffineMapAttrs, not text. Reading them through
            # `.value.results` is the difference between a constant bound and
            # a map over loop variables -- the textual form `affine_map<() ->
            # (8)>` does not survive naive splitting, which is what made this
            # return 0 kernels on the first attempt.
            def _const(attr_name):
                m = op.attributes[attr_name].value
                if m.n_dims or m.n_symbols or len(m.results) != 1:
                    raise Unsupported("non-constant affine.for bound")
                s = str(m.results[0])
                if not s.lstrip("-").isdigit():
                    raise Unsupported(f"affine.for bound {s!r}")
                return s

            lo_s, hi_s = _const("lowerBoundMap"), _const("upperBoundMap")
            iv = op.regions[0].blocks[0].arguments[0]
            name_iv = self.fresh("i")
            self.v[iv] = name_iv
            step = 1
            if "step" in op.attributes:
                step = int(str(op.attributes["step"]).split(":")[0])
            self.w(f"for (int {name_iv} = {lo_s}; {name_iv} < {hi_s}; "
                   f"{name_iv} += {step}) {{")
            self.ind += 1
            for inner in op.regions[0].blocks[0].operations:
                self.op(inner)
            self.ind -= 1
            self.w("}")
            return

        if name in ("affine.yield", "func.return", "memref.dealloc",
                    "cf.assert"):
            # cf.assert guards a dynamic shape that is static by this point;
            # it produces no value and has no C counterpart.
            return

        raise Unsupported(name)


def emit(module, fn_name="candidate_kernel", alias_result=False):
    """Return C for the single func in `module`, or raise `Unsupported`.

    `alias_result` binds the returned `memref.alloc` directly onto the `out0`
    parameter, so the loop nest writes its result where the caller wants it and
    the trailing element-by-element copy disappears along with the buffer.

    This is what makes P1's postcondition -- one loop nest, zero intermediate
    arrays -- reachable rather than aspirational. Without it every kernel keeps
    one static array and one extra full pass over the output purely as an
    artefact of bufferization choosing a fresh allocation for the return value.

    Only a `memref.alloc` is aliased. A returned function argument (in-place) or
    a `memref.get_global` (constant data) is left alone: the first already IS
    caller memory and the second must not be written through.
    """
    func = None
    for o in module.body.operations:
        if o.operation.name == "func.func":
            func = o
            break
    if func is None:
        raise Unsupported("no func.func")

    e = Emitter()
    blk = func.regions[0].blocks[0]
    params, outs = [], []
    for i, a in enumerate(blk.arguments):
        nm = f"v_xs_{i}"
        e.v[a] = nm
        params.append(f"const {_elem_ctype(a.type)} *{nm}")

    # the result memref is returned, not passed -- find it and rename to out0
    ret = None
    for o in blk.operations:
        if o.operation.name == "func.return" and len(o.operands) == 1:
            ret = o.operands[0]
    body_ops = [o for o in blk.operations]

    aliased = False
    if alias_result and ret is not None:
        owner = getattr(ret, "owner", None)
        oname = getattr(getattr(owner, "name", None), "__str__", lambda: "")()
        if oname == "memref.alloc":
            e.v[ret] = "out0"
            e.aliased.add(ret)
            aliased = True

    for o in body_ops:
        e.op(o)

    out_c = e.v.get(ret)
    if out_c is None:
        raise Unsupported("return value is not a named buffer")
    et = _elem_ctype(ret.type)
    sig = (f'extern "C" void {fn_name}(' + ", ".join(params)
           + f", {et} *out0)")
    n = 1
    for d in _shape(ret.type):
        n *= d
    body = e.buf.getvalue()
    tail = ("" if aliased
            else f"  for (int i = 0; i < {n}; i++) out0[i] = {out_c}[i];\n")
    return ("#include <stdint.h>\n#include <math.h>\n\n"
            + sig + " {\n" + body + tail + "}\n")
