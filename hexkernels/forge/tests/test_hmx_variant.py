"""A contraction selected at fp32 must also be offered at fp16.

The negative half is the point: a rule that returns a narrow dtype for
EVERYTHING is not detecting contractions, and the fp32 negative controls in
the test set would then be mislabelled as HMX tasks.
"""
import torch

from hexkernels.forge import mine
from hexkernels.forge.mechanism import HMX, HMX_MAX_DTYPE_BYTES, plan_for


class Matmul(torch.nn.Module):
    def forward(self, a, b):
        return torch.matmul(a, b)


class Elementwise(torch.nn.Module):
    def forward(self, a, b):
        return a * b + a


def _graph(mod, *shapes, dtype=torch.float32):
    args = tuple(torch.rand(s, dtype=dtype) for s in shapes)
    from hexkernels.forge.frontend.trace import trace
    return trace(mod, args, "hmx_twin_probe")


def test_contraction_gets_a_narrow_twin():
    g = _graph(Matmul(), (128, 128), (128, 128))
    assert mine.hmx_twin_dtype(g) is torch.float16


def test_elementwise_gets_no_twin():
    g = _graph(Elementwise(), (128, 128), (128, 128))
    assert mine.hmx_twin_dtype(g) is None


def test_the_twin_actually_grants_hmx():
    """Guards against returning fp16 for a graph that still would not get HMX."""
    g = _graph(Matmul(), (128, 128), (128, 128))
    dt = mine.hmx_twin_dtype(g)
    assert dt is not None
    assert HMX in plan_for(g, dtype_bytes=2).mechanisms


def test_fp32_contraction_is_refused_hmx():
    """The negative control the test set depends on."""
    g = _graph(Matmul(), (128, 128), (128, 128))
    assert HMX not in plan_for(g, dtype_bytes=4).mechanisms
    assert HMX_MAX_DTYPE_BYTES == 2
