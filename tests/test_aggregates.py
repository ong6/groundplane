"""Adversarial fixtures for aggregate_reconciles: arithmetic that reads right and isn't.

Each case names the real-world failure it stands for.
"""

import math

import pytest

from groundplane import FactRegistry
from groundplane.aggregates import aggregate_reconciles, check_aggregate_reconciles
from groundplane.errors import UnsupportedClaim

# Four campaigns. Conversions total 17,910 and the unweighted mean CTR is exactly
# 0.0389 — a number a model will happily report as "the" CTR even though the
# conversion volumes differ by 3x, which is what weight_column exists to catch.
CAMPAIGNS = [
    {"campaign": "north", "conversions": 4200, "ctr": 0.0412, "spend": 88000},
    {"campaign": "harbour", "conversions": 5100, "ctr": 0.0455, "spend": 61200},
    {"campaign": "delta", "conversions": 6210, "ctr": 0.0301, "spend": 40100},
    {"campaign": "vega", "conversions": 2400, "ctr": 0.0388, "spend": 15500},
]

# Five regions, three of them active. A "total for active regions" computed over all
# five is the classic filtered-aggregate error.
REGIONS = [
    {"region": "apac", "status": "active", "revenue": 1200.0},
    {"region": "emea", "status": "active", "revenue": 900.0},
    {"region": "amer", "status": "paused", "revenue": 4400.0},
    {"region": "latam", "status": "active", "revenue": 300.0},
    {"region": "meta", "status": "paused", "revenue": 2500.0},
]

ACTIVE_REVENUE = 1200.0 + 900.0 + 300.0
UNWEIGHTED_CTR = math.fsum(r["ctr"] for r in CAMPAIGNS) / 4
WEIGHTED_CTR = math.fsum(r["ctr"] * r["conversions"] for r in CAMPAIGNS) / 17910


@pytest.fixture
def registry() -> FactRegistry:
    reg = FactRegistry()
    reg.record_table(
        "campaigns",
        CAMPAIGNS,
        key="campaign",
        tool="warehouse.query",
        args={"table": "campaign_daily", "window": "7d"},
    )
    reg.record_table(
        "regions",
        REGIONS,
        key="region",
        tool="warehouse.query",
        args={"table": "region_rollup"},
    )
    return reg


@pytest.fixture
def truncated() -> FactRegistry:
    reg = FactRegistry()
    reg.record_table(
        "campaigns",
        CAMPAIGNS,
        key="campaign",
        tool="warehouse.query",
        args={"table": "campaign_daily", "limit": 4},
        exhaustive=False,
    )
    return reg


# 1. The headline number is simply not the sum of the rows underneath it.
def test_stated_total_does_not_sum_raises(registry):
    with pytest.raises(UnsupportedClaim) as exc:
        check_aggregate_reconciles(
            registry,
            {"total_conversions": 18400},
            fact="campaigns",
            field="total_conversions",
            column="conversions",
            op="sum",
        )
    msg = str(exc.value)
    assert "17910" in msg
    assert "+490" in msg  # the delta is printed, not just "mismatch"
    assert exc.value.field == "total_conversions"
    assert "warehouse.query" in msg


def test_correct_total_passes(registry):
    check_aggregate_reconciles(
        registry,
        {"total_conversions": 17910},
        fact="campaigns",
        field="total_conversions",
        column="conversions",
        op="sum",
    )


# 2. Mean of means: the model averaged the per-campaign CTRs instead of weighting
#    them by conversions. The two figures differ and only one is the real CTR.
def test_unweighted_mean_reported_as_weighted_raises(registry):
    assert not math.isclose(UNWEIGHTED_CTR, WEIGHTED_CTR, abs_tol=1e-6)
    with pytest.raises(UnsupportedClaim) as exc:
        check_aggregate_reconciles(
            registry,
            {"ctr": UNWEIGHTED_CTR},
            fact="campaigns",
            field="ctr",
            column="ctr",
            op="mean",
            weight_column="conversions",
        )
    assert "unweighted mean" in str(exc.value)  # names the likely source of the error


def test_weighted_mean_passes(registry):
    check_aggregate_reconciles(
        registry,
        {"ctr": WEIGHTED_CTR},
        fact="campaigns",
        field="ctr",
        column="ctr",
        op="mean",
        weight_column="conversions",
        rel_tolerance=1e-9,
    )


# 3. "Revenue from active regions" computed over every region, paused ones included.
def test_filtered_aggregate_over_wrong_subset_raises(registry):
    all_revenue = math.fsum(r["revenue"] for r in REGIONS)
    with pytest.raises(UnsupportedClaim) as exc:
        check_aggregate_reconciles(
            registry,
            {"active_revenue": all_revenue},
            fact="regions",
            field="active_revenue",
            column="revenue",
            op="sum",
            where={"status": "active"},
        )
    assert "2400" in str(exc.value)  # the true active total
    check_aggregate_reconciles(
        registry,
        {"active_revenue": ACTIVE_REVENUE},
        fact="regions",
        field="active_revenue",
        column="revenue",
        op="sum",
        where={"status": "active"},
    )


# 4. Arithmetically perfect sum over a result set that was capped. Not verifiable.
def test_aggregate_over_truncated_table_raises(truncated):
    with pytest.raises(UnsupportedClaim) as exc:
        check_aggregate_reconciles(
            truncated,
            {"total_conversions": 17910},
            fact="campaigns",
            field="total_conversions",
            column="conversions",
            op="sum",
        )
    msg = str(exc.value)
    assert "not exhaustive" in msg
    assert "warehouse.query" in msg


def test_extremum_over_truncated_table_also_raises(truncated):
    with pytest.raises(UnsupportedClaim, match="not exhaustive"):
        check_aggregate_reconciles(
            truncated,
            {"best": 0.0455},
            fact="campaigns",
            field="best",
            column="ctr",
            op="max",
        )


# 5. A count arrives as a float. "6.0 campaigns" is a type confusion, not a count.
def test_count_claimed_as_float_raises(registry):
    with pytest.raises(UnsupportedClaim, match="integer claim"):
        check_aggregate_reconciles(
            registry,
            {"n": 5.0},
            fact="regions",
            field="n",
            op="count",
        )
    check_aggregate_reconciles(registry, {"n": 5}, fact="regions", field="n", op="count")


def test_count_claimed_as_bool_raises(registry):
    with pytest.raises(UnsupportedClaim, match="integer claim"):
        check_aggregate_reconciles(
            registry,
            {"n": True},
            fact="regions",
            field="n",
            op="count",
            where={"status": "paused", "region": "amer"},
        )


# 6. Rounding: a presentation-layer round is only acceptable if the caller said so.
def test_rounded_mean_fails_at_zero_tolerance_and_passes_in_tolerance(registry):
    assert math.isclose(UNWEIGHTED_CTR, 0.0389, abs_tol=1e-12)
    with pytest.raises(UnsupportedClaim):
        check_aggregate_reconciles(
            registry, {"ctr": 0.039}, fact="campaigns", field="ctr", column="ctr", op="mean"
        )
    check_aggregate_reconciles(
        registry,
        {"ctr": 0.039},
        fact="campaigns",
        field="ctr",
        column="ctr",
        op="mean",
        tolerance=0.0002,
    )


# 7. A filter that matches nothing must fail loudly, not divide by zero or pass.
def test_empty_filter_result_raises_unsupported_claim(registry):
    with pytest.raises(UnsupportedClaim) as exc:
        check_aggregate_reconciles(
            registry,
            {"total": 0.0},
            fact="regions",
            field="total",
            column="revenue",
            op="sum",
            where={"status": "archived"},
        )
    assert "selected no rows" in str(exc.value)


# 8. Wiring mistakes fail at construction, where a developer sees them.
def test_missing_column_config_raises_at_builder_time():
    with pytest.raises(ValueError, match="needs a column"):
        aggregate_reconciles(fact="t", field="x", op="sum")


def test_weight_column_on_non_mean_raises_at_builder_time():
    with pytest.raises(ValueError, match="only meaningful with op='mean'"):
        aggregate_reconciles(fact="t", field="x", column="c", op="sum", weight_column="w")


# A partial column silently under-totals; a sum over it is a lie with a number on it.
def test_sum_over_partial_column_raises():
    reg = FactRegistry()
    reg.record_table(
        "sparse",
        [{"k": "a", "v": 1.0}, {"k": "b"}, {"k": "c", "v": 2.0}],
        key="k",
        tool="warehouse.query",
    )
    with pytest.raises(UnsupportedClaim, match="partial column"):
        check_aggregate_reconciles(
            reg, {"total": 3.0}, fact="sparse", field="total", column="v", op="sum"
        )
    # ...but an extremum over the rows that do carry the column is well defined.
    check_aggregate_reconciles(
        reg, {"top": 2.0}, fact="sparse", field="top", column="v", op="max"
    )


# The field simply isn't there: an unverifiable claim is a failed claim.
def test_missing_declared_field_raises(registry):
    with pytest.raises(UnsupportedClaim, match="no such field"):
        check_aggregate_reconciles(
            registry,
            {"headline": "Conversions were up."},
            fact="campaigns",
            field="total_conversions",
            column="conversions",
            op="sum",
        )


def test_median_of_even_count_is_the_midpoint(registry):
    expected = (0.0388 + 0.0412) / 2
    check_aggregate_reconciles(
        registry,
        {"median_ctr": expected},
        fact="campaigns",
        field="median_ctr",
        column="ctr",
        op="median",
        rel_tolerance=1e-9,
    )
    with pytest.raises(UnsupportedClaim, match="median"):
        check_aggregate_reconciles(
            registry,
            {"median_ctr": 0.0455},
            fact="campaigns",
            field="median_ctr",
            column="ctr",
            op="median",
        )


def test_builder_runs_as_a_boundary_check(registry):
    check = aggregate_reconciles(
        fact="campaigns", field="total", column="conversions", op="sum"
    )
    check(registry, {"total": 17910})
    with pytest.raises(UnsupportedClaim):
        check(registry, {"total": 17911})
