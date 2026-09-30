"""
Per-request metrics collection.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Metrics:
    """Collect per-request metrics."""
    request_id: str
    start_time: float = field(default_factory=time.monotonic)
    stages: dict[str, float] = field(default_factory=dict)
    retrieval_count: int = 0
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    estimated_cost_usd: Optional[float] = None

    def record_stage(self, name: str) -> None:
        self.stages[name] = time.monotonic() - self.start_time

    @property
    def total_latency_ms(self) -> float:
        return (time.monotonic() - self.start_time) * 1000

    def to_dict(self) -> dict:
        return {
            "request_id": self.request_id,
            "total_latency_ms": round(self.total_latency_ms, 2),
            "stages": {k: round(v * 1000, 2) for k, v in self.stages.items()},
            "retrieval_count": self.retrieval_count,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
        }
