"""Adversarial fixtures for ``row_integrity``: attribute swap across rows.

Each case names the real-world failure it stands for.
"""

import pytest

from groundplane.errors import UnsupportedClaim
from groundplane.registry import FactRegistry
from groundplane.rows import check_row_integrity, row_integrity

# Two campaigns whose numbers are individually plausible for either of them. "north"
# is the big-spend brand; "harbour" is the efficient one. Swapping a column between
# them produces output that no per-field check can fault.
ROWS = [
    {"campaign": "north", "conversions": 9800, "spend": 88000, "ctr": 0.0412},
    {"campaign": "harbour", "conversions": 12400, "spend": 61200, "ctr": 0.0455},
    {"campaign": "delta", "conversions": 4100, "spend": 30500, "ctr": 0.0301},
]


@pytest.fixture
def registry() -> FactRegistry:
    reg = FactRegistry()
    reg.record_table(
        "campaign_rows",
        ROWS,
        key="campaign",
        tool="warehouse.query",
        args={"table": "campaign_daily", "window": "7d"},
    )
    return reg


def _run(registry, output, **kw):
    check_row_integrity(
        registry,
        output,
        fact="campaign_rows",
        key_field="winner",
        fields={"conversions": "conversions", "spend": "spend"},
        **kw,
    )


# 1. The motivating failure: right winner, right conversions, another row's spend.
def test_attribute_swap_across_rows_raises(registry):
    with pytest.raises(UnsupportedClaim) as exc:
        _run(registry, {"winner": "harbour", "conversions": 12400, "spend": 88000})
    msg = str(exc.value)
    assert exc.value.field == "spend"
    assert "'north'" in msg
    assert "61200" in msg
    assert "warehouse.query" in msg  # provenance is in the error


# 2. A key blended from two real ones. Must be a claim failure, not a lookup crash.
def test_blended_key_not_in_table_raises(registry):
    with pytest.raises(UnsupportedClaim) as exc:
        _run(registry, {"winner": "northwest", "conversions": 9800, "spend": 88000})
    assert exc.value.field == "winner"
    assert "harbour" in str(exc.value) and "north" in str(exc.value)


# 3. Every metric copied wholesale from a different single row.
def test_all_fields_from_one_wrong_row_raises(registry):
    with pytest.raises(UnsupportedClaim) as exc:
        _run(registry, {"winner": "harbour", "conversions": 9800, "spend": 88000})
    assert exc.value.field in {"conversions", "spend"}
    assert "'north'" in str(exc.value)


# 4. Float noise: tolerance is a parameter, not a property of the check.
def test_rounding_drift_under_zero_tolerance_raises(registry):
    output = {"winner": "harbour", "ctr": 0.0455000001}
    with pytest.raises(UnsupportedClaim):
        check_row_integrity(
            registry, output, fact="campaign_rows", key_field="winner", fields={"ctr": "ctr"}
        )
    check_row_integrity(
        registry,
        output,
        fact="campaign_rows",
        key_field="winner",
        fields={"ctr": "ctr"},
        rel_tolerance=1e-6,
    )


# 5. A declared field the model simply omitted: unverifiable is unsupported.
def test_missing_declared_field_raises(registry):
    with pytest.raises(UnsupportedClaim, match="no such field"):
        check_row_integrity(
            registry,
            {"winner": "harbour"},
            fact="campaign_rows",
            key_field="winner",
            fields={"spend": "spend"},
        )


# 6. Classic type confusion: True == 1 in Python, and a bool is not a count.
def test_bool_masquerading_as_number_raises(registry):
    reg = FactRegistry()
    reg.record_table(
        "flags",
        [{"campaign": "harbour", "conversions": 1}],
        key="campaign",
        tool="warehouse.query",
    )
    with pytest.raises(UnsupportedClaim) as exc:
        check_row_integrity(
            reg,
            {"winner": "harbour", "conversions": True},
            fact="flags",
            key_field="winner",
            fields={"conversions": "conversions"},
        )
    assert exc.value.field == "conversions"


# 7. The honest answer passes untouched.
def test_correct_row_passes(registry):
    _run(registry, {"winner": "harbour", "conversions": 12400, "spend": 61200})


# A column that does not exist is a wiring bug, not a model claim.
def test_unknown_column_is_a_developer_error(registry):
    with pytest.raises(KeyError):
        check_row_integrity(
            registry,
            {"winner": "harbour", "roas": 3.1},
            fact="campaign_rows",
            key_field="winner",
            fields={"roas": "roas"},
        )


def test_builder_matches_imperative(registry):
    check = row_integrity(fact="campaign_rows", key_field="winner", fields={"spend": "spend"})
    check(registry, {"winner": "harbour", "spend": 61200})
    with pytest.raises(UnsupportedClaim):
        check(registry, {"winner": "harbour", "spend": 88000})


# A null or NaN cell where the model claimed a number: the table can neither support
# nor refute the claim, so it is a data error naming the cell, not a model failure.
@pytest.mark.parametrize("bad", [None, float("nan"), float("inf")])
def test_null_or_non_finite_recorded_cell_is_a_data_error(bad):
    reg = FactRegistry()
    reg.record_table(
        "campaign",
        [{"campaign": "north", "ctr": bad}],
        key="campaign",
        tool="warehouse.query",
    )
    with pytest.raises(ValueError, match="fact 'campaign' column 'ctr' row 'north'"):
        check_row_integrity(
            reg,
            {"winner": "north", "ctr": 0.04},
            fact="campaign",
            key_field="winner",
            fields={"ctr": "ctr"},
        )
    # A non-numeric claim against the same cell is still judged as a claim.
    with pytest.raises(UnsupportedClaim):
        check_row_integrity(
            reg,
            {"winner": "north", "ctr": "high"},
            fact="campaign",
            key_field="winner",
            fields={"ctr": "ctr"},
        )
