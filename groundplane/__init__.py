"""groundplane: a hard line between generated output and computed facts."""

from .boundary import Boundary, boundary
from .checks import check_superlative, field_matches_fact, superlative
from .errors import FactBoundaryError, UnregisteredFact, UnsupportedClaim
from .registry import Fact, FactRegistry, Provenance, Ranking

__version__ = "0.0.1"

__all__ = [
    "Boundary",
    "boundary",
    "superlative",
    "check_superlative",
    "field_matches_fact",
    "FactBoundaryError",
    "UnregisteredFact",
    "UnsupportedClaim",
    "Fact",
    "FactRegistry",
    "Provenance",
    "Ranking",
    "__version__",
]
