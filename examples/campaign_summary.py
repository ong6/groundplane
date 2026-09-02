"""A model names the wrong campaign as the winner; the boundary catches it.

Run with ``python examples/campaign_summary.py``. Exits 0: the first submission is expected
to raise ``UnsupportedClaim`` and is caught, then the corrected output passes.
"""

from __future__ import annotations

from typing import Any

from groundplane import FactRegistry, UnsupportedClaim, boundary, superlative

# Pretend this came from a warehouse query. The argmax is computed by record_ranking, in code.
CAMPAIGN_CTR = {"north": 0.0412, "harbour": 0.0455, "delta": 0.0301}


def fake_model(facts: dict[str, Any], *, wrong: bool) -> dict[str, Any]:
    """Stand-in for an LLM producing structured output from the permitted facts.

    With ``wrong=True`` it does what a real model sometimes does: names a plausible
    campaign that is not the computed winner.
    """
    ranking = facts["campaign_ctr"]
    winner = "north" if wrong else ranking.winner
    return {
        "winner": winner,
        "ctr": ranking.score_of(winner),
        "summary": f"{winner} had the highest click-through rate this week.",
    }


def main() -> None:
    reg = FactRegistry()
    reg.record_ranking(
        "campaign_ctr",
        CAMPAIGN_CTR,
        key="ctr",
        tool="warehouse.query",
        args={"table": "campaign_daily", "window": "7d"},
    )

    guard = boundary(
        reg,
        facts=["campaign_ctr"],
        checks=[superlative(fact="campaign_ctr", score_field="ctr")],
    )

    print("1. Model names the wrong winner:")
    try:
        with guard as b:
            b.submit(fake_model(b.facts(), wrong=True))
    except UnsupportedClaim as exc:
        print(f"   raised UnsupportedClaim:\n   {exc}\n")
    else:
        raise SystemExit("expected UnsupportedClaim, got a pass")

    print("2. Model phrases the computed winner:")
    with guard as b:
        output = b.submit(fake_model(b.facts(), wrong=False))
    print(f"   passed: {output}")


if __name__ == "__main__":
    main()
