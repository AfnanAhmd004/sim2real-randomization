"""System identification: fit simulator parameters to logged data from the "real" robot.

Given observed states and the commands that were sent, find the parameters (and the
integer actuator delay) whose one-step predictions best match the log. Disturbances and
sensor noise are not modelled here, so a robust (soft-L1) loss is used.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

from .envs import CartPole, Params

FREE = {
    "cartpole": ["m_cart", "m_pole", "half_len", "damping", "gain"],
    "quadrotor": ["mass", "inertia", "arm", "gain_l", "gain_r", "tau"],
}


def _params(env, values: dict[str, float], n: int) -> Params:
    return {k: np.full(n, values.get(k, v), dtype=int if k in env.integer else float) for k, v in env.nominal.items()}


def predict(env, obs: np.ndarray, cmds: np.ndarray, values: dict[str, float], delay: int) -> np.ndarray:
    """One-step predictions s[t+1] for t = 0..T-2 from logged obs and commands (single episode)."""
    T = len(obs)
    p = _params(env, {**values, "delay": delay}, 1)
    delayed = np.concatenate([np.zeros((delay, env.n_act)), cmds])[:T]
    act = np.zeros((1, 2)) + (env.nominal.get("mass", 0) * env.g / 2 if env.name == "quadrotor" else 0)
    applied = []
    for t in range(T):
        a, act = env.apply(delayed[t:t + 1], p, act)
        applied.append(a)
    applied = np.concatenate(applied)
    pT = _params(env, {**values, "delay": delay}, T - 1)
    if isinstance(env, CartPole):
        return env.dynamics(obs[:-1], applied[:-1], pT)
    return env.dynamics(obs[:-1], applied[:-1], pT, wind=np.zeros(T - 1))


def identify(env, episodes: list[tuple[np.ndarray, np.ndarray]], max_delay: int = 4) -> tuple[dict[str, float], int, float]:
    """Return (continuous params, delay, residual RMS) from logged episodes [(obs, cmds), ...].

    Only the part of each episode before any failure should be passed in.
    """
    names = FREE[env.name]
    lo = np.log([env.limits[k][0] or 1e-3 for k in names])
    hi = np.log([env.limits[k][1] for k in names])
    x0 = np.log([max(env.nominal[k], 1e-3) for k in names])
    vel = slice(env.n_obs // 2, env.n_obs)  # velocities carry the dynamics information
    scale = np.concatenate([o[:, vel] for o, _ in episodes]).std(axis=0) + 1e-6
    best = None
    for d in range(max_delay + 1):
        def resid(z, d=d):
            vals = dict(zip(names, np.exp(z)))
            return np.concatenate([((predict(env, o, c, vals, d)[:, vel] - o[1:, vel]) / scale).ravel()
                                   for o, c in episodes])
        sol = least_squares(resid, x0, bounds=(lo, hi), loss="soft_l1", max_nfev=200)
        rms = float(np.sqrt(np.mean(sol.fun**2)))
        if best is None or rms < best[2]:
            best = (dict(zip(names, np.exp(sol.x))), d, rms)
    return best
