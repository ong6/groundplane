import pytest

from groundplane import FactBoundaryError, UnregisteredFact, boundary, superlative


def test_happy_path_context_manager(registry):
    with boundary(registry, facts=["campaign_ctr"], checks=[superlative(fact="campaign_ctr")]) as b:
        assert b.facts()["campaign_ctr"].winner == "harbour"
        out = b.submit({"winner": "harbour", "headline": "Harbour led the week."})
    assert out["winner"] == "harbour"


def test_happy_path_decorator(registry):
    @boundary(registry, facts=["campaign_ctr"], checks=[superlative(fact="campaign_ctr")])
    def summarize():
        return {"winner": "harbour"}

    assert summarize()["winner"] == "harbour"


def test_boundary_rejects_undeclared_fact(registry):
    with pytest.raises(UnregisteredFact), boundary(registry, facts=["nope"]):
        pass


def test_exiting_without_submit_raises(registry):
    with (
        pytest.raises(FactBoundaryError, match="without submit"),
        boundary(registry, facts=["campaign_ctr"]),
    ):
        pass


def test_no_claim_block_may_opt_out(registry):
    with boundary(registry, facts=["campaign_ctr"], require_output=False):
        pass


def test_prose_output_is_rejected(registry):
    with (
        boundary(registry, facts=["campaign_ctr"], require_output=False) as b,
        pytest.raises(TypeError, match="free prose"),
    ):
        b.submit("Harbour was the best performing campaign.")


def test_boundary_declaration_is_reentrant(registry):
    guard = boundary(registry, facts=["campaign_ctr"], checks=[superlative(fact="campaign_ctr")])
    with guard as b:
        b.submit({"winner": b.facts()["campaign_ctr"].winner})
    with guard as b:
        b.submit({"winner": b.facts()["campaign_ctr"].winner})
    with pytest.raises(FactBoundaryError), guard:
        pass


# --- named checks and report mode -------------------------------------------------

import json  # noqa: E402

from groundplane import (  # noqa: E402
    Boundary,
    ClaimsUnsupported,
    NamedCheck,
    UnsupportedClaim,
    aggregate_reconciles,
    comparison,
    entities_recorded,
    field_matches_fact,
    ranking_prefix,
    row_integrity,
)


def _plain(registry, output):
    if output.get("plain") != "ok":
        raise UnsupportedClaim("plain", output.get("plain"), "ok", reason="plain check")


def test_builders_return_named_checks():
    expected = {
        superlative(fact="campaign_ctr"): "superlative(fact='campaign_ctr', field='winner')",
        field_matches_fact(field="n", fact="f"): "field_matches_fact(fact='f', field='n')",
        ranking_prefix(fact="campaign_ctr", field="top"): (
            "ranking_prefix(fact='campaign_ctr', field='top')"
        ),
        aggregate_reconciles(fact="t", field="total", column="v"): (
            "aggregate_reconciles(fact='t', field='total')"
        ),
        entities_recorded(fields=["a", "b"], fact="d"): (
            "entities_recorded(fact='d', fields=['a', 'b'])"
        ),
        row_integrity(fact="t", key_field="id", fields={"v": "v"}): (
            "row_integrity(fact='t', key_field='id')"
        ),
        comparison(fact="r", a_field="a", b_field="b", delta_field="d"): (
            "comparison(fact='r', delta_field='d')"
        ),
    }
    for check, name in expected.items():
        assert isinstance(check, NamedCheck)
        assert check.name == name


def test_plain_callable_is_accepted_and_named(registry):
    with boundary(registry, checks=[_plain]) as b:
        report = b.report({"plain": "ok"})
    assert report.ok
    assert report.passed == ("_plain",)


def test_report_collects_every_violation(registry):
    checks = [superlative(fact="campaign_ctr"), _plain]
    with boundary(registry, facts=["campaign_ctr"], checks=checks) as b:
        report = b.report({"winner": "north", "plain": "nope"})
    assert not report.ok
    assert len(report.violations) == 2
    assert report.failed == (
        "superlative(fact='campaign_ctr', field='winner')",
        "_plain",
    )
    assert report.passed == ()
    assert [v.field for v in report.violations] == ["winner", "plain"]


def test_report_ok_lists_passed_names(registry):
    checks = [superlative(fact="campaign_ctr"), _plain]
    with boundary(registry, facts=["campaign_ctr"], checks=checks) as b:
        report = b.report({"winner": "harbour", "plain": "ok"})
    assert report.ok
    assert report.violations == ()
    assert report.failed == ()
    assert len(report.passed) == 2
    assert report.to_text() == ""


def test_report_counts_as_submitted_even_when_failing(registry):
    with boundary(registry, facts=["campaign_ctr"], checks=[_plain]) as b:
        assert not b.report({"plain": "nope"}).ok
    # no FactBoundaryError on exit: the block was checked


def test_submit_mode_all_raises_claims_unsupported_listing_every_violation(registry):
    checks = [superlative(fact="campaign_ctr"), _plain]
    with (
        pytest.raises(ClaimsUnsupported) as info,
        boundary(registry, facts=["campaign_ctr"], checks=checks) as b,
    ):
        b.submit({"winner": "north", "plain": "nope"}, mode="all")
    text = str(info.value)
    assert text == info.value.report.to_text()
    assert len(text.splitlines()) == 2
    assert "computed argmax" in text
    assert "plain check" in text
    assert isinstance(info.value, FactBoundaryError)


def test_boundary_mode_forwarded_to_submit(registry):
    checks = [superlative(fact="campaign_ctr"), _plain]
    guard = boundary(registry, facts=["campaign_ctr"], checks=checks, mode="all")
    with pytest.raises(ClaimsUnsupported) as info, guard as b:
        b.submit({"winner": "north", "plain": "nope"})
    assert len(info.value.report.violations) == 2
    # and a per-call override wins
    with pytest.raises(UnsupportedClaim), guard as b:
        b.submit({"winner": "north", "plain": "nope"}, mode="first")


def test_submit_mode_first_keeps_first_failure_semantics(registry):
    checks = [superlative(fact="campaign_ctr"), _plain]
    with (
        pytest.raises(UnsupportedClaim, match="computed argmax"),
        boundary(registry, facts=["campaign_ctr"], checks=checks) as b,
    ):
        b.submit({"winner": "north", "plain": "nope"})


def test_config_errors_propagate_from_report(registry):
    # A Ranking fact with column= is a wiring mistake, not a model claim.
    checks = [superlative(fact="campaign_ctr", column="ctr"), _plain]
    with (
        pytest.raises(ValueError, match="already a Ranking"),
        boundary(registry, facts=["campaign_ctr"], checks=checks) as b,
    ):
        b.report({"winner": "harbour", "plain": "ok"})


def test_report_and_claim_to_dict_are_json_serialisable(registry):
    class Opaque:
        def __repr__(self):
            return "<opaque>"

    def weird(registry, output):
        raise UnsupportedClaim("f", Opaque(), {1, 2}, fact="campaign_ctr", reason="odd values")

    with boundary(registry, facts=["campaign_ctr"], checks=[weird, _plain]) as b:
        report = b.report({"plain": "nope"})
    d = report.to_dict()
    json.dumps(d)
    assert d["ok"] is False
    assert d["failed"] == ["weird", "_plain"]
    claim = d["violations"][0]
    assert claim == {
        "field": "f",
        "claimed": "<opaque>",
        "supported": "{1, 2}",
        "fact": "campaign_ctr",
        "provenance": None,
        "reason": "odd values",
    }
    assert set(d["violations"][1]) == {
        "field",
        "claimed",
        "supported",
        "fact",
        "provenance",
        "reason",
    }


def test_boundary_declaration_yields_boundary_and_decorator_preserves_signature(registry):
    guard = boundary(registry, facts=["campaign_ctr"], checks=[superlative(fact="campaign_ctr")])
    with guard as b:
        assert isinstance(b, Boundary)
        b.submit({"winner": "harbour"})

    @guard
    def summarize(winner: str) -> dict:
        return {"winner": winner}

    assert summarize.__name__ == "summarize"
    assert summarize("harbour") == {"winner": "harbour"}
