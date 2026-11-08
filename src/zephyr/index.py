"""Position-aware segment index."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np


@dataclass
class Segment:
    segment_id: int
    positions: List[int] = field(default_factory=list)
    vectors: List[np.ndarray] = field(default_factory=list)

    def add(self, position: int, vector: np.ndarray) -> None:
        self.positions.append(position)
        self.vectors.append(vector)

    def search(self, query: np.ndarray, k: int = 10) -> List[int]:
        if not self.vectors:
            return []
        matrix = np.stack(self.vectors)
        scores = matrix @ query / (np.linalg.norm(matrix, axis=1) * np.linalg.norm(query) + 1e-9)
        order = np.argsort(-scores)[:k]
        return [self.positions[i] for i in order]


class ZephyrIndex:
    def __init__(self, max_segment_size: int = 4096) -> None:
        self.max_segment_size = max_segment_size
        self.segments: Dict[int, Segment] = {}
        self._next_id = 0

    def insert(self, position: int, vector: np.ndarray) -> None:
        segment = self.segments.get(self._next_id)
        if segment is None or len(segment.positions) >= self.max_segment_size:
            self._next_id = len(self.segments)
            segment = Segment(segment_id=self._next_id)
            self.segments[self._next_id] = segment
        segment.add(position, np.asarray(vector, dtype=np.float32))

    def search(self, query: np.ndarray, k: int = 10) -> List[int]:
        query = np.asarray(query, dtype=np.float32)
        results: List[int] = []
        for segment in self.segments.values():
            results.extend(segment.search(query, k))
        return results[:k]
