from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class SimArm:
    name: str
    event_times: np.ndarray
    event_observed: np.ndarray  # 1=event, 0=censored


def simulate_survival_arm(
    n: int,
    hazard: float,
    censor_hazard: float,
    rng: np.random.Generator,
) -> SimArm:
    """
    Simulate event times ~ Exp(hazard), censor times ~ Exp(censor_hazard).
    Observed time = min(event, censor).
    """
    t_event = rng.exponential(scale=1.0 / max(hazard, 1e-9), size=n)
    t_cens = rng.exponential(scale=1.0 / max(censor_hazard, 1e-9), size=n)
    t_obs = np.minimum(t_event, t_cens)
    observed = (t_event <= t_cens).astype(np.int32)
    return SimArm(name="arm", event_times=t_obs.astype(np.float32), event_observed=observed.astype(np.int32))


def kaplan_meier_curve(times: np.ndarray, observed: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute KM step curve as arrays (t, S(t)) where t includes 0 at start.
    """
    times = np.asarray(times, dtype=np.float64)
    observed = np.asarray(observed, dtype=np.int32)

    # sort by time
    order = np.argsort(times)
    times = times[order]
    observed = observed[order]

    uniq_times = np.unique(times)
    n = times.size
    at_risk = n

    t_points = [0.0]
    s_points = [1.0]
    s = 1.0

    idx = 0
    for t in uniq_times:
        # count events and censored at this time
        mask_t = (times == t)
        d = int(np.sum(observed[mask_t] == 1))
        c = int(np.sum(observed[mask_t] == 0))

        if at_risk > 0 and d > 0:
            s *= (1.0 - d / at_risk)
            t_points.append(float(t))
            s_points.append(float(s))

        # update at risk after processing this time
        at_risk -= (d + c)
        idx += int(mask_t.sum())

    return np.asarray(t_points, dtype=np.float32), np.asarray(s_points, dtype=np.float32)


def sample_censor_marks(
    times: np.ndarray,
    observed: np.ndarray,
    km_t: np.ndarray,
    km_s: np.ndarray,
    max_marks: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Choose a subset of censored individuals and map them to (t, S(t)) on the KM curve for plotting '+' marks.
    """
    cens_idx = np.where(observed == 0)[0]
    if cens_idx.size == 0:
        return np.zeros((0,), dtype=np.float32), np.zeros((0,), dtype=np.float32)

    k = min(max_marks, cens_idx.size)
    choose = rng.choice(cens_idx, size=k, replace=False)
    t_marks = times[choose].astype(np.float32)

    # survival at each t is last KM step at time <= t
    s_marks = np.zeros_like(t_marks, dtype=np.float32)
    for i, tm in enumerate(t_marks):
        j = int(np.searchsorted(km_t, tm, side="right") - 1)
        j = max(0, min(j, km_s.size - 1))
        s_marks[i] = km_s[j]
    return t_marks, s_marks
