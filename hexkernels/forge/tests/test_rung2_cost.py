"""`cost_usd` billed cached input at the uncached rate.

Rung 1 had 385,150 of 776,613 input tokens (49.6%) served from cache and charged
every one of them at full price. Rung 2's prompt is far larger and every retry
restates it, so the cache share rises and the error rises with it.

`cost_usd`'s own docstring is the reason this matters: it refuses a built-in price
table because "a stale constant compiled into the record would be a fabricated
number that looks measured". A wrong rate is the same defect by a different route.
"""
from hexkernels.forge import rung0


def _usage(prompt, completion, cached=None):
    u = {"prompt_tokens": prompt, "completion_tokens": completion}
    if cached is not None:
        u["prompt_tokens_details"] = {"cached_tokens": cached}
    return u


def test_uncached_cost_is_unchanged():
    """Every rung-0 record was priced by the old formula. It must still agree."""
    got = rung0.cost_usd(_usage(1_000_000, 1_000_000), 1.0, 4.0)
    assert got == 5.0


def test_cached_tokens_are_billed_at_the_cached_rate():
    got = rung0.cost_usd(_usage(1_000_000, 0, cached=500_000), 1.0, 4.0,
                         price_cached=0.1)
    assert got == 0.55, "500k at $1 + 500k at $0.10"


def test_cached_tokens_cost_full_price_when_no_cached_rate_is_given():
    """No silent discount. An unsupplied rate is unknown, not zero."""
    got = rung0.cost_usd(_usage(1_000_000, 0, cached=500_000), 1.0, 4.0)
    assert got == 1.0


def test_no_price_means_no_number():
    assert rung0.cost_usd(_usage(10, 10), None, 4.0) is None
    assert rung0.cost_usd({}, 1.0, 4.0) is None


def test_cached_can_never_exceed_prompt_tokens():
    """Defensive: a provider reporting cached > prompt would otherwise produce a
    negative uncached count and a cost below zero."""
    got = rung0.cost_usd(_usage(100, 0, cached=1000), 1.0, 4.0, price_cached=0.1)
    assert got >= 0


# --------------------------------------------------------------------- reprice

class TestReprice:
    """`reprice` recomputes `usage["cost_usd"]` from whatever counts are CURRENT.

    It exists because `generate_one` prices the first turn only, and a retry wave
    then folds later turns' tokens into that same `usage` dict via `rung2._add_usage`
    without ever touching the dollar figure. `aggregate` sums exactly that figure as
    the rung's headline cost, so a stale `cost_usd` under-reports by whatever the
    retries spent.
    """

    def test_reprices_from_the_current_counts(self):
        usage = {"prompt_tokens": 2_000_000, "completion_tokens": 2_000_000,
                 "cost_usd": 5.0, "price_per_1m": {"in": 1.0, "out": 4.0}}
        out = rung0.reprice(usage)
        assert out is usage, "in place, per the contract"
        assert out["cost_usd"] == 10.0, "double the tokens, double the bill"

    def test_no_recorded_rates_means_it_stays_unpriced(self):
        """Inventing a rate here would be the exact fabricated-number failure
        `cost_usd`'s docstring refuses. No rates recorded, no number produced."""
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 1_000_000}
        out = rung0.reprice(usage)
        assert out.get("cost_usd") is None

    def test_the_cached_rate_is_honoured_through_reprice(self):
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0,
                 "prompt_tokens_details": {"cached_tokens": 500_000},
                 "price_per_1m": {"in": 1.0, "out": 4.0, "cached_in": 0.1}}
        out = rung0.reprice(usage)
        assert out["cost_usd"] == 0.55, "500k at $1 + 500k at $0.10"

    def test_missing_cached_rate_bills_cached_tokens_at_full_price(self):
        usage = {"prompt_tokens": 1_000_000, "completion_tokens": 0,
                 "prompt_tokens_details": {"cached_tokens": 500_000},
                 "price_per_1m": {"in": 1.0, "out": 4.0}}
        out = rung0.reprice(usage)
        assert out["cost_usd"] == 1.0

    def test_partial_rates_also_stay_unpriced(self):
        """`in` recorded but `out` missing: still no fabricated number."""
        usage = {"prompt_tokens": 10, "completion_tokens": 10,
                 "price_per_1m": {"in": 1.0}}
        out = rung0.reprice(usage)
        assert out.get("cost_usd") is None
