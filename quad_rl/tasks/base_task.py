from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Tuple

import numpy as np


def wrap_angle(angle: float) -> float:
    return (angle + np.pi) % (2.0 * np.pi) - np.pi


@dataclass
class TaskStep:
    reward: float
    success: bool = False
    done: bool = False
    info: Dict[str, Any] = field(default_factory=dict)


class BaseTask:
    name = "base"

    def __init__(self, cfg=None) -> None:
        self.cfg = cfg
        self.done = False

    def reset(self, env, rng: np.random.Generator) -> None:
        self.done = False

    def step(self, env) -> TaskStep:
        return TaskStep(reward=0.0)

    def get_target(self, env) -> Tuple[np.ndarray, float]:
        return env.state.pos.copy(), float(env.state.yaw)
