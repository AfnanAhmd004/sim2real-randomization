"""Policies and training: evolution strategies under nominal, randomised and automatically
widened (ADR) simulators, plus the LQR baseline designed on the nominal model."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import solve_discrete_are

from .envs import CartPole, Params, nominal_params, rollout, sample_params


class LinearPolicy:
    """u = -K · obs, with K of shape (n_act, n_obs)."""

    def __init__(self, env, theta: np.ndarray | None = None):
        self.env = env
        self.shape = (env.n_act, env.n_obs)
        self.theta = np.zeros(np.prod(self.shape)) if theta is None else np.asarray(theta, float)

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        return -obs @ self.theta.reshape(self.shape).T

    def with_theta(self, theta: np.ndarray) -> "LinearPolicy":
        return LinearPolicy(self.env, theta)


def lqr_policy(env) -> LinearPolicy:
    """Discrete LQR from a finite-difference linearisation of the *nominal* model."""
    p = nominal_params(env, 1)
    n, m = env.n_obs, env.n_act
    x0 = np.zeros((1, n))
    eps = 1e-5

    def step(x, u):
        if isinstance(env, CartPole):
            return env.dynamics(x, p["gain"] * u[:, 0], p)
        hover = env.nominal["mass"] * env.g / 2
        thrust = np.stack([hover + u[:, 0] - u[:, 1], hover + u[:, 0] + u[:, 1]], axis=1)
        return env.dynamics(x, thrust, p)

    A = np.zeros((n, n))
    B = np.zeros((n, m))
    for i in range(n):
        d = np.zeros((1, n))
        d[0, i] = eps
        A[:, i] = (step(x0 + d, np.zeros((1, m))) - step(x0 - d, np.zeros((1, m))))[0] / (2 * eps)
    for j in range(m):
        d = np.zeros((1, m))
        d[0, j] = eps
        B[:, j] = (step(x0, d) - step(x0, -d))[0] / (2 * eps)
    Q = np.diag([0.1, 1.0, 0.01, 0.01]) if isinstance(env, CartPole) else np.diag([1, 1, 0.5, 0.1, 0.1, 0.05])
    R = np.eye(m) * (0.001 if isinstance(env, CartPole) else 0.01)
    P = solve_discrete_are(A, B, Q, R)
    K = np.linalg.solve(R + B.T @ P @ B, B.T @ P @ A)
    return LinearPolicy(env, K.ravel())


def evaluate_many(env, thetas: np.ndarray, params: Params, seed: int) -> np.ndarray:
    """Mean return of each linear policy over the same plants, in ONE batched rollout.

    Candidates x plants are flattened into a single batch with a per-row gain matrix.
    """
    c, n = len(thetas), len(next(iter(params.values())))
    K = np.repeat(thetas.reshape(c, env.n_act, env.n_obs), n, axis=0)  # (c*n, a, o)
    tiled = {k: np.tile(v, c) for k, v in params.items()}
    out = rollout(env, lambda obs: -np.einsum("nao,no->na", K, obs), tiled, seed, tile=c)
    return out.returns.reshape(c, n).mean(axis=1)


@dataclass
class ES:
    """OpenAI-style evolution strategies with antithetic sampling, rank shaping and Adam.

    All candidates in a generation see the *same* sampled plants and seeds (common
    random numbers), so differences in fitness come from the policy, not the dice.
    """

    sigma: float = 0.1
    lr: float = 0.05
    pairs: int = 16
    plants_per_eval: int = 32
    seed: int = 0
    history: list[float] = field(default_factory=list)

    def train(self, env, theta0: np.ndarray, sampler, generations: int = 150, on_generation=None) -> np.ndarray:
        rng = np.random.default_rng(self.seed)
        theta = theta0.astype(float).copy()
        m, v = np.zeros_like(theta), np.zeros_like(theta)
        for g in range(generations):
            params = sampler(self.plants_per_eval, rng)
            seed = int(rng.integers(1 << 30))
            eps = rng.normal(0, 1, (self.pairs, len(theta)))
            cands = np.concatenate([theta + self.sigma * eps, theta - self.sigma * eps])
            fit = evaluate_many(env, cands, params, seed)
            ranks = np.empty(len(fit))
            ranks[np.argsort(fit)] = np.arange(len(fit))
            shaped = ranks / (len(fit) - 1) - 0.5
            grad = (shaped[: self.pairs] - shaped[self.pairs:]) @ eps / (2 * self.pairs * self.sigma)
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * grad**2
            theta += self.lr * (m / (1 - 0.9 ** (g + 1))) / (np.sqrt(v / (1 - 0.999 ** (g + 1))) + 1e-8)
            self.history.append(float(fit.mean()))
            if on_generation:
                on_generation(g, theta)
        return theta


class ADR:
    """Automatic domain randomisation (after OpenAI et al., 2019).

    Every parameter starts at its nominal value. Each generation one boundary is probed:
    a batch is evaluated with that parameter pinned at the boundary (others sampled from
    the current ranges). If success at the boundary is high, the range expands; if it is
    low, it shrinks. Training samples from the current ranges, so difficulty grows only as
    fast as the policy can cope.
    """

    def __init__(self, env, step: float = 0.08, expand_at: float = 0.85, shrink_at: float = 0.5, probe_n: int = 32,
                 seed: int = 0):
        self.env = env
        self.ranges = {k: [float(v), float(v)] for k, v in env.nominal.items()}
        self.step, self.expand_at, self.shrink_at, self.probe_n = step, expand_at, shrink_at, probe_n
        self.rng = np.random.default_rng(seed)
        self.log: list[dict] = []

    def sampler(self, n: int, rng) -> Params:
        return sample_params(self.env, {k: tuple(v) for k, v in self.ranges.items()}, n, rng)

    def _delta(self, k: str) -> float:
        lo, hi = self.env.limits[k]
        return 1.0 if k in self.env.integer else self.step * (hi - lo)

    def probe(self, g: int, theta: np.ndarray) -> None:
        k = list(self.ranges)[self.rng.integers(len(self.ranges))]
        side = int(self.rng.integers(2))
        params = self.sampler(self.probe_n, self.rng)
        params[k] = np.full(self.probe_n, self.ranges[k][side], dtype=params[k].dtype)
        succ = rollout(self.env, LinearPolicy(self.env, theta), params, int(self.rng.integers(1 << 30))).success.mean()
        lo_lim, hi_lim = self.env.limits[k]
        d = self._delta(k) * (1 if side else -1)
        if succ >= self.expand_at:
            self.ranges[k][side] = float(np.clip(self.ranges[k][side] + d, lo_lim, hi_lim))
        elif succ <= self.shrink_at:
            nom = self.env.nominal[k]
            new = self.ranges[k][side] - d
            self.ranges[k][side] = float(max(new, nom) if side else min(new, nom))
        self.log.append({"generation": g, "param": k, "side": side, "success": float(succ),
                         **{f"{p}_lo": r[0] for p, r in self.ranges.items()},
                         **{f"{p}_hi": r[1] for p, r in self.ranges.items()}})

    def coverage(self) -> float:
        """Mean share of each parameter's 'reality' range covered by the current ADR range."""
        fr = []
        for k, (rlo, rhi) in self.env.reality.items():
            lo, hi = self.ranges[k]
            fr.append(max(0.0, min(hi, rhi) - max(lo, rlo)) / (rhi - rlo) if rhi > rlo else 1.0)
        return float(np.mean(fr))
