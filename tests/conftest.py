import pytest

from groundplane import FactRegistry

# Campaign CTRs. "north" is the plausible-sounding winner (biggest brand, most spend)
# but "harbour" is the actual argmax. Every adversarial case leans on that gap.
CTR = {"north": 0.0412, "harbour": 0.0455, "delta": 0.0301, "vega": 0.0288}


@pytest.fixture
def registry() -> FactRegistry:
    reg = FactRegistry()
    reg.record_ranking(
        "campaign_ctr",
        CTR,
        key="ctr",
        tool="warehouse.query",
        args={"table": "campaign_daily", "window": "7d"},
    )
    return reg
