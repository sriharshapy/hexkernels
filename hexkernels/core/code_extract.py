"""Extract C source from an LLM completion (markdown fence or raw).

Port of ``m_rl/code_extract.py`` (OLD HVX repo). No dataset/version paths in
this module, so nothing to repoint -- ported verbatim.

CROSS-DEPENDENCY NOTE (see task-10-brief.md): the eventual home for this is
``core.code_extract``, built in a later ladder task. That
module does not exist yet, so ``core.run`` (this task) imports it
from here instead. When the ladder task lands, this module should become a
thin re-export (or the ladder module should import FROM here) rather than a
second copy -- whichever way, keep exactly one implementation.
"""
import re

# A complete fenced block: opening ```lang, a newline, body, closing ```.
_FENCE = re.compile(r"```(?:c|cpp|C)?[ \t]*\r?\n(.*?)```", re.DOTALL)
# A bare opening fence marker (```lang followed by a newline).
_OPEN = re.compile(r"```(?:c|cpp|C)?[ \t]*\r?\n")
# A line that is nothing but a fence marker.
_FENCE_LINE = re.compile(r"^\s*```")


def extract_c_source(text: str) -> str:
    """Return the C source the model intended.

    Handles three cases so a stray markdown fence never leaks into the
    compiled source (the cause of HVX-tagged NO-COMPILE false-zeros):
    1. A complete ```c ... ``` block -> its contents (first block wins).
    2. An UNCLOSED opening fence (a common LLM truncation) -> everything
       after the opening marker, with any dangling trailing fence removed.
    3. No opening fence -> the text with stray fence-marker lines stripped.
    Never raises.
    """
    if not text:
        return ""
    m = _FENCE.search(text)
    if m:
        return m.group(1).strip()
    m2 = _OPEN.search(text)
    if m2:
        rest = re.sub(r"\r?\n```.*\Z", "", text[m2.end():], flags=re.DOTALL)
        return rest.strip()
    cleaned = "\n".join(ln for ln in text.splitlines() if not _FENCE_LINE.match(ln))
    return cleaned.strip()
