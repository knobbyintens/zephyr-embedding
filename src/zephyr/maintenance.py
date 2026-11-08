"""Compaction and maintenance policy for Zephyr-Embedding."""

from __future__ import annotations

from dataclasses import dataclass

from .index import ZephyrIndex


@dataclass
class MaintenancePolicy:
    drift_threshold: float = 0.15
    compaction_ratio: float = 0.6

    def should_compact(self, live_ratio: float, estimated_drift: float) -> bool:
        if estimated_drift >= self.drift_threshold:
            return True
        return live_ratio < self.compaction_ratio

    def compact(self, index: ZephyrIndex) -> dict:
        before = len(index.segments)
        merged: dict = {}
        for segment in index.segments.values():
            for position, vector in zip(segment.positions, segment.vectors):
                merged[position] = vector
        rebuilt = ZephyrIndex(max_segment_size=index.max_segment_size)
        for position in sorted(merged):
            rebuilt.insert(position, merged[position])
        index.segments = rebuilt.segments
        index._next_id = rebuilt._next_id
        return {"segments_before": before, "segments_after": len(index.segments)}
