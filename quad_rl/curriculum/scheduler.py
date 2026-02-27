from __future__ import annotations


class LinearScheduler:
    """Simple linear curriculum scheduler utility."""

    def __init__(self, start: float = 0.0, end: float = 1.0, total_steps: int = 1_000_000) -> None:
        self.start = float(start)
        self.end = float(end)
        self.total_steps = max(1, int(total_steps))

    def value(self, step: int) -> float:
        t = min(1.0, max(0.0, float(step) / float(self.total_steps)))
        return self.start + (self.end - self.start) * t
