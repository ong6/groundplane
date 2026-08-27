"""Adversarial fixtures for ``entities_recorded``: names that read real and are not.

Each case names the real-world failure it stands for.
"""

import pytest

from groundplane import FactRegistry
from groundplane.entities import check_entities_recorded, entities_recorded
from groundplane.errors import UnsupportedClaim

CAMPAIGNS = ("north", "harbour", "delta", "vega")
REGIONS = ("apac", "emea", "namer")


@pytest.fixture
def domain_registry() -> FactRegistry:
    reg = FactRegistry()
    reg.record_domain(
        "campaigns",
        CAMPAIGNS,
        tool="warehouse.query",
        args={"table": "campaign_daily"},
        label="campaign",
    )
    reg.record_domain("regions", REGIONS, tool="warehouse.query", label="region")
    return reg


def _run(registry: FactRegistry, output: dict, **kw) -> None:
    check_entities_recorded(registry, output, **kw)


# 1. The blend: two real campaign names fused into one that never existed.
def test_blended_name_from_two_real_ones_raises(domain_registry: FactRegistry) -> None:
    with pytest.raises(UnsupportedClaim) as exc:
        _run(domain_registry, {"winner": "northwest"}, fields=["winner"], fact="campaigns")
    msg = str(exc.value)
    assert "'northwest'" in msg
    assert "did you mean 'north'?" in msg
    assert "warehouse.query" in msg
    assert exc.value.field == "winner"


# 2. An id carried over from an earlier tool call: real-looking, wrong fact.
def test_identifier_leaked_from_prior_tool_call_raises(domain_registry: FactRegistry) -> None:
    with pytest.raises(UnsupportedClaim) as exc:
        _run(domain_registry, {"winner": "ACME-4471"}, fields=["winner"], fact="campaigns")
    msg = str(exc.value)
    assert "did you mean" not in msg  # nothing is close; the domain itself is the answer
    assert "harbour" in msg and "vega" in msg


# 3. Majority-correct list: three names, one invented, and the check must not average.
def test_one_invented_member_in_a_valid_list_raises(domain_registry: FactRegistry) -> None:
    with pytest.raises(UnsupportedClaim) as exc:
        _run(
            domain_registry,
            {"regions": ["apac", "emea", "latam"]},
            fields=["regions"],
            fact="regions",
        )
    assert exc.value.claimed == "latam"
    assert exc.value.field == "regions"


# 4. The empty list is a claim ("none missed target"), not an absence of one.
def test_empty_list_smuggles_a_claim(domain_registry: FactRegistry) -> None:
    with pytest.raises(UnsupportedClaim, match="empty"):
        _run(domain_registry, {"regions": []}, fields=["regions"], fact="regions")
    _run(domain_registry, {"regions": []}, fields=["regions"], fact="regions", allow_empty=True)


# 5. Case drift: right entity, wrong spelling, and joins on the string will miss it.
def test_case_drift_raises_unless_permitted(domain_registry: FactRegistry) -> None:
    out = {"winner": "Harbour"}
    with pytest.raises(UnsupportedClaim) as exc:
        _run(domain_registry, out, fields=["winner"], fact="campaigns")
    assert "'harbour'" in str(exc.value)  # error shows the recorded spelling
    _run(domain_registry, out, fields=["winner"], fact="campaigns", case_sensitive=False)


# 6. A number where a name belongs: a claim failure, never a TypeError from the checker.
def test_non_string_member_raises_unsupported_claim_not_typeerror(
    domain_registry: FactRegistry,
) -> None:
    with pytest.raises(UnsupportedClaim, match="not a name"):
        _run(domain_registry, {"regions": ["apac", 7]}, fields=["regions"], fact="regions")


# 7. The same verdict whichever primitive recorded the names.
def test_identical_verdict_across_domain_ranking_and_table() -> None:
    reg = FactRegistry()
    reg.record_domain("d", CAMPAIGNS, tool="warehouse.query")
    reg.record_ranking(
        "r", {name: float(i) for i, name in enumerate(CAMPAIGNS)}, key="ctr", tool="warehouse.query"
    )
    reg.record_table(
        "t",
        [{"campaign": name, "ctr": 0.01} for name in CAMPAIGNS],
        key="campaign",
        tool="warehouse.query",
    )
    for fact in ("d", "r", "t"):
        _run(reg, {"winner": "harbour"}, fields=["winner"], fact=fact)
        with pytest.raises(UnsupportedClaim):
            _run(reg, {"winner": "northwest"}, fields=["winner"], fact=fact)


# 8. A fact that is not a name source at all is a wiring bug, and must say so.
def test_non_domain_fact_raises_typeerror(domain_registry: FactRegistry) -> None:
    domain_registry.record("total", 41, tool="warehouse.query")
    with pytest.raises(TypeError, match="not a Domain, Ranking or Table"):
        _run(domain_registry, {"winner": "north"}, fields=["winner"], fact="total")


# 9. Nested output: a bad path is a claim failure, not a KeyError out of the checker.
def test_missing_nested_path_raises_unsupported_claim(domain_registry: FactRegistry) -> None:
    _run(
        domain_registry,
        {"summary": {"winner": "harbour"}},
        fields=["winner"],
        fact="campaigns",
        path=("summary",),
    )
    with pytest.raises(UnsupportedClaim, match="no object at path"):
        _run(
            domain_registry,
            {"winner": "harbour"},
            fields=["winner"],
            fact="campaigns",
            path=("summary",),
        )


# 10. A declared field the model simply omitted: reported as a claim, not a KeyError.
def test_missing_field_raises_unsupported_claim(domain_registry: FactRegistry) -> None:
    with pytest.raises(UnsupportedClaim, match="declared no such field"):
        _run(domain_registry, {}, fields=["winner"], fact="campaigns")


# 11. Huge domains must stay printable, or the reask prompt becomes the bug.
def test_large_domain_is_truncated_in_the_error() -> None:
    reg = FactRegistry()
    reg.record_domain("many", [f"acct-{i:03d}" for i in range(50)], tool="crm.list")
    with pytest.raises(UnsupportedClaim) as exc:
        _run(reg, {"account": "acct-999"}, fields=["account"], fact="many")
    assert "(+30 more)" in str(exc.value)
    assert len(exc.value.supported) == 21


# 12. Happy path through the builder, exactly as a boundary would call it.
def test_builder_passes_valid_names(domain_registry: FactRegistry) -> None:
    check = entities_recorded(fields=["winner", "regions"], fact="campaigns")
    with pytest.raises(UnsupportedClaim):
        check(domain_registry, {"winner": "harbour", "regions": ["apac"]})
    entities_recorded(fields=["winner"], fact="campaigns")(
        domain_registry, {"winner": "harbour", "regions": ["apac"]}
    )
