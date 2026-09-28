# environment/configuration.py
"""
Canonical, hashable, serializable representation of a spacecraft configuration.

This is critical for A*: we need to hash configurations and store them in
visited sets. We provide a canonical form that is invariant to module ID
ordering (positions+orientations sorted), plus a full form that preserves IDs.

We separate:
  - geometric configuration (positions + orientations)  -> used for A*/geometry
  - full module states (battery/temp) live elsewhere and are NOT part of the
    geometric hash by default (mission-agnostic).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, List
import numpy as np


@dataclass(frozen=True)
class Configuration:
    """Immutable geometric configuration.

    positions:    tuple of (x,y,z) tuples, indexed by module slot
    orientations: tuple of orientation ints, indexed by module slot
    ids:          tuple of module ids, indexed by module slot
    """
    positions: Tuple[Tuple[int, int, int], ...]
    orientations: Tuple[int, ...]
    ids: Tuple[int, ...]

    @staticmethod
    def from_modules(modules: List) -> "Configuration":
        positions = tuple(tuple(int(v) for v in m.position) for m in modules)
        orientations = tuple(int(m.orientation) for m in modules)
        ids = tuple(int(m.id) for m in modules)
        return Configuration(positions, orientations, ids)

    def canonical_key(self) -> Tuple:
        """A hashable key invariant to module-ID labeling.

        Sort (position, orientation) pairs. Useful when modules are
        interchangeable (identical capabilities). If modules differ, use
        `id_key` instead.
        """
        pairs = sorted(zip(self.positions, self.orientations))
        return tuple(pairs)

    def id_key(self) -> Tuple:
        """Hashable key preserving module identity (id -> pos,orient)."""
        triples = sorted(
            (i, p, o) for i, p, o in zip(self.ids, self.positions, self.orientations)
        )
        return tuple(triples)

    def __hash__(self):
        return hash(self.id_key())

    def positions_set(self):
        return set(self.positions)