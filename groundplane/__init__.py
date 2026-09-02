"""groundplane: a hard line between generated output and computed facts."""

from .aggregates import aggregate_reconciles, check_aggregate_reconciles
from .boundary import Boundary, boundary
from .checks import check_superlative, field_matches_fact, superlative
from .comparison import check_comparison, comparison
from .entities import check_entities_recorded, entities_recorded
from .errors import FactBoundaryError, UnregisteredFact, UnsupportedClaim
from .ordering import check_ranking_prefix, ranking_prefix
from .registry import (
    Domain,
    Fact,
    FactRegistry,
    Provenance,
    Ranking,
    SparseColumnWarning,
    Table,
)
from .rows import check_row_integrity, row_integrity

__version__ = "0.1.0"

__all__ = [
    "Boundary",
    "boundary",
    "superlative",
    "check_superlative",
    "field_matches_fact",
    "ranking_prefix",
    "check_ranking_prefix",
    "comparison",
    "check_comparison",
    "aggregate_reconciles",
    "check_aggregate_reconciles",
    "entities_recorded",
    "check_entities_recorded",
    "row_integrity",
    "check_row_integrity",
    "FactBoundaryError",
    "UnregisteredFact",
    "UnsupportedClaim",
    "Fact",
    "FactRegistry",
    "Provenance",
    "Ranking",
    "Table",
    "Domain",
    "SparseColumnWarning",
    "__version__",
]
