from __future__ import annotations

from dataclasses import dataclass


@dataclass
class EvalStats:
    episodes: int = 0
    successes: int = 0
    crashes: int = 0
    success_steps_total: int = 0

    def update(self, success: bool, crash: bool, steps: int) -> None:
        self.episodes += 1
        if success:
            self.successes += 1
            self.success_steps_total += int(steps)
        if crash:
            self.crashes += 1

    @property
    def success_rate(self) -> float:
        if self.episodes == 0:
            return 0.0
        return self.successes / self.episodes

    @property
    def crash_rate(self) -> float:
        if self.episodes == 0:
            return 0.0
        return self.crashes / self.episodes

    @property
    def mean_steps_success(self) -> float:
        if self.successes == 0:
            return 0.0
        return self.success_steps_total / self.successes
