from __future__ import annotations

import numpy as np

from quad_rl.tasks.hover_task import HoverTask
from quad_rl.tasks.landing_task import LandingTask
from quad_rl.tasks.yaw_task import YawTask


def sample_stage_a_task(rng: np.random.Generator, cfg):
    pool = [HoverTask(cfg), YawTask(cfg), LandingTask(cfg)]
    idx = int(rng.integers(0, len(pool)))
    return pool[idx]
