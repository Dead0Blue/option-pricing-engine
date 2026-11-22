"""Cox-Ross-Rubinstein binomial tree for European and American options.

Backward induction is vectorised over the nodes of each time layer, so a
2,000-step tree runs in milliseconds. For American options the value at
each node is max(continuation, immediate exercise).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class TreeResult:
    price: float
    delta: float
    gamma: float
    exercise_boundary: np.ndarray | None = None  # (times, spot levels), American puts


def crr_price(S, K, T, r, sigma, q=0.0, kind="call", american=False, steps=500,
              return_boundary=False) -> TreeResult:
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("risk-neutral probability outside (0,1): reduce dt")
    disc = np.exp(-r * dt)
    sign = 1.0 if kind == "call" else -1.0

    j = np.arange(steps + 1)
    spot = S * u**j * d ** (steps - j)
    value = np.maximum(sign * (spot - K), 0.0)

    boundary_t, boundary_s = [], []
    layers = {}
    for i in range(steps - 1, -1, -1):
        j = np.arange(i + 1)
        spot = S * u**j * d ** (i - j)
        value = disc * (p * value[1:] + (1.0 - p) * value[:-1])
        if american:
            exercise = np.maximum(sign * (spot - K), 0.0)
            early = exercise > value
            value = np.where(early, exercise, value)
            if return_boundary and kind == "put" and early.any():
                boundary_t.append(i * dt)
                boundary_s.append(spot[early].max())
        if i <= 2:
            layers[i] = (spot.copy(), value.copy())

    # Delta and gamma read off the first two layers of the tree.
    (s1, v1), (s2, v2) = layers[1], layers[2]
    delta = (v1[1] - v1[0]) / (s1[1] - s1[0])
    d_up = (v2[2] - v2[1]) / (s2[2] - s2[1])
    d_dn = (v2[1] - v2[0]) / (s2[1] - s2[0])
    gamma = (d_up - d_dn) / (0.5 * (s2[2] - s2[0]))

    boundary = None
    if return_boundary and boundary_t:
        boundary = np.array([boundary_t[::-1], boundary_s[::-1]])
    return TreeResult(float(value[0]), float(delta), float(gamma), boundary)
