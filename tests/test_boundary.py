import pytest

from groundplane import FactBoundaryError, UnregisteredFact, boundary, superlative


def test_happy_path_context_manager(registry):
    with boundary(
        registry, facts=["campaign_ctr"], checks=[superlative(fact="campaign_ctr")]
    ) as b:
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
    with pytest.raises(FactBoundaryError, match="without submit"), boundary(
        registry, facts=["campaign_ctr"]
    ):
        pass


def test_no_claim_block_may_opt_out(registry):
    with boundary(registry, facts=["campaign_ctr"], require_output=False):
        pass


def test_prose_output_is_rejected(registry):
    with boundary(registry, facts=["campaign_ctr"], require_output=False) as b, pytest.raises(
        TypeError, match="free prose"
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
