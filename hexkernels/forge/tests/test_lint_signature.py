"""The entry-point rule must reject what will not LINK, and nothing else.

It rejected 19 of 342 graded rung-0 attempts for formatting -- declarations that
differ from the prompt's text only in whitespace and link perfectly. Each was
recorded as "did not compile", which understates correctness and attributes a
whitespace preference to the model. These tests pin both halves: formatting is
forgiven, substance is not.
"""
import pytest

from hexkernels.forge import lint

SIG = ('extern "C" void candidate_kernel(const _Float16 *v_args_0, '
       'const _Float16 *v_args_1, _Float16 *out0)')


def _errors(body, signature=SIG):
    return [f for f in lint.lint(body, signature=signature)
            if f.rule == "entry-point-signature"]


@pytest.mark.parametrize("decl", [
    # exactly as the prompt gives it
    'extern "C" void candidate_kernel(const _Float16 *v_args_0, '
    'const _Float16 *v_args_1, _Float16 *out0)',
    # a space after the open paren -- the case that cost 19 attempts
    'extern "C" void candidate_kernel( const _Float16 *v_args_0, '
    'const _Float16 *v_args_1, _Float16 *out0)',
    # star bound to the type instead of the name
    'extern "C" void candidate_kernel(const _Float16* v_args_0, '
    'const _Float16* v_args_1, _Float16* out0)',
    # one parameter per line, as a formatter would leave it
    'extern "C" void candidate_kernel(const _Float16 *v_args_0,\n'
    '                                 const _Float16 *v_args_1,\n'
    '                                 _Float16 *out0)',
    # space before the comma and inside the closing paren
    'extern "C" void candidate_kernel(const _Float16 *v_args_0 , '
    'const _Float16 *v_args_1 , _Float16 *out0 )',
])
def test_formatting_is_forgiven(decl):
    assert _errors(decl + " {}\n") == [], decl


@pytest.mark.parametrize("decl,why", [
    ('extern "C" void candidate_kernel(const _Float16 *renamed, '
     'const _Float16 *v_args_1, _Float16 *out0)', "renamed parameter"),
    ('extern "C" void candidate_kernel(const float *v_args_0, '
     'const _Float16 *v_args_1, _Float16 *out0)', "changed type"),
    ('extern "C" void candidate_kernel(const _Float16 *v_args_1, '
     'const _Float16 *v_args_0, _Float16 *out0)', "reordered parameters"),
    ('extern "C" void candidate_kernel(const _Float16 *v_args_0, '
     '_Float16 *out0)', "dropped a parameter"),
    ('void candidate_kernel(const _Float16 *v_args_0, '
     'const _Float16 *v_args_1, _Float16 *out0)', "missing extern C"),
])
def test_substance_is_still_rejected(decl, why):
    assert _errors(decl + " {}\n"), why


def test_a_missing_entry_point_is_reported_as_such():
    found = _errors("int unrelated(void) { return 0; }\n")
    assert found
    assert "no candidate_kernel declaration" in found[0].evidence


def test_canon_sig_preserves_names_types_and_order():
    """Guard on the canonicaliser itself: it may only remove whitespace."""
    canon = lint._canon_sig(SIG)
    for token in ("candidate_kernel", "const", "_Float16", "v_args_0",
                  "v_args_1", "out0", 'extern "C"'):
        assert token in canon, token
    assert canon.index("v_args_0") < canon.index("v_args_1") < canon.index("out0")
    assert "  " not in canon
