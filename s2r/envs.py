"""Vectorised robot simulators whose physical parameters can be randomised per episode.

Each environment simulates a batch of N independent plants at once (one parameter set
per plant), which makes evolution strategies and large evaluations cheap on a CPU.

Parameters are dicts of arrays of shape (N,). The "reality gap" is modelled with the
usual suspects: masses and geometry, actuator gain, actuator delay and lag, friction,
sensor noise and external disturbances.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Params = dict[str, np.ndarray]


@dataclass
class Rollout:
    returns: np.ndarray  # (N,) mean per-step reward
    success: np.ndarray  # (N,) survived the whole episode within limits
    states: np.ndarray | None = None  # (T, N, n_obs) observed states, if recorded
    commands: np.ndarray | None = None  # (T, N, n_act)
    steps: np.ndarray | None = None  # (N,) steps survived


def _delayed(buf: np.ndarray, delay: np.ndarray) -> np.ndarray:
    """buf: (D+1, N, A) with buf[0] newest; pick buf[delay[i], i]."""
    return buf[delay, np.arange(buf.shape[1])]


class CartPole:
    """Cart-pole balancing with random pushes. Action: horizontal force on the cart."""

    name = "cartpole"
    n_obs, n_act = 4, 1
    dt, horizon, f_max, g = 0.02, 500, 10.0, 9.81
    nominal = {"m_cart": 1.0, "m_pole": 0.1, "half_len": 0.5, "damping": 0.1, "gain": 1.0, "delay": 0,
               "noise": 0.0, "push": 1.0}
    reality = {"m_cart": (0.5, 2.0), "m_pole": (0.05, 0.4), "half_len": (0.25, 1.0), "damping": (0.0, 0.5),
               "gain": (0.6, 1.3), "delay": (0, 3), "noise": (0.0, 0.02), "push": (0.0, 3.0)}
    guess = {"m_cart": (0.7, 1.5), "m_pole": (0.05, 0.3), "half_len": (0.3, 0.8), "damping": (0.0, 0.3),
             "gain": (0.8, 1.2), "delay": (0, 2), "noise": (0.0, 0.01), "push": (0.0, 2.0)}
    limits = {"m_cart": (0.2, 4.0), "m_pole": (0.02, 1.0), "half_len": (0.1, 2.0), "damping": (0.0, 1.0),
              "gain": (0.3, 2.0), "delay": (0, 5), "noise": (0.0, 0.05), "push": (0.0, 6.0)}
    integer = {"delay"}

    def dynamics(self, s: np.ndarray, force: np.ndarray, p: Params) -> np.ndarray:
        x, th, xd, thd = s.T
        mc, mp, l, b = p["m_cart"], p["m_pole"], p["half_len"], p["damping"]
        total = mc + mp
        sin, cos = np.sin(th), np.cos(th)
        tmp = (force - b * xd + mp * l * thd**2 * sin) / total
        thdd = (self.g * sin - cos * tmp) / (l * (4.0 / 3.0 - mp * cos**2 / total))
        xdd = tmp - mp * l * thdd * cos / total
        xd2, thd2 = xd + self.dt * xdd, thd + self.dt * thdd
        return np.stack([x + self.dt * xd2, th + self.dt * thd2, xd2, thd2], axis=1)

    def reset(self, n: int, rng) -> np.ndarray:
        return np.stack([rng.uniform(-0.5, 0.5, n), rng.uniform(-0.15, 0.15, n), np.zeros(n), np.zeros(n)], axis=1)

    def apply(self, cmd: np.ndarray, p: Params, state_act: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return np.clip(p["gain"] * cmd[:, 0], -self.f_max, self.f_max), state_act

    def disturbance(self, n: int, p: Params, rng) -> np.ndarray:
        return (rng.random(n) < 0.02) * rng.normal(0, 1, n) * p["push"] * 10.0

    def failed(self, s: np.ndarray) -> np.ndarray:
        return (np.abs(s[:, 0]) > 2.4) | (np.abs(s[:, 1]) > 0.6)

    def reward(self, s: np.ndarray, force: np.ndarray) -> np.ndarray:
        return -(0.1 * s[:, 0] ** 2 + s[:, 1] ** 2 + 0.001 * force**2)

    def observe(self, s: np.ndarray, p: Params, rng) -> np.ndarray:
        return s + rng.normal(0, 1, s.shape) * p["noise"][:, None]


class PlanarQuadrotor:
    """2-D quadrotor (VTOL) holding position under wind gusts. Actions: two rotor thrusts."""

    name = "quadrotor"
    n_obs, n_act = 6, 2
    dt, horizon, g = 0.02, 500, 9.81
    nominal = {"mass": 1.0, "inertia": 0.01, "arm": 0.15, "gain_l": 1.0, "gain_r": 1.0, "tau": 0.03, "delay": 0,
               "noise": 0.0, "wind": 0.5}
    reality = {"mass": (0.7, 1.6), "inertia": (0.006, 0.02), "arm": (0.1, 0.22), "gain_l": (0.8, 1.1),
               "gain_r": (0.8, 1.1), "tau": (0.01, 0.12), "delay": (0, 3), "noise": (0.0, 0.02), "wind": (0.0, 2.5)}
    guess = {"mass": (0.85, 1.25), "inertia": (0.008, 0.014), "arm": (0.13, 0.18), "gain_l": (0.9, 1.05),
             "gain_r": (0.9, 1.05), "tau": (0.02, 0.06), "delay": (0, 1), "noise": (0.0, 0.01), "wind": (0.0, 1.5)}
    limits = {"mass": (0.4, 3.0), "inertia": (0.003, 0.05), "arm": (0.05, 0.4), "gain_l": (0.5, 1.5),
              "gain_r": (0.5, 1.5), "tau": (0.005, 0.3), "delay": (0, 5), "noise": (0.0, 0.05), "wind": (0.0, 5.0)}
    integer = {"delay"}
    t_max = 15.0  # N per rotor

    def reset(self, n: int, rng) -> np.ndarray:
        s = np.zeros((n, 6))
        s[:, 0] = rng.uniform(-1, 1, n)
        s[:, 1] = rng.uniform(-1, 1, n)
        s[:, 2] = rng.uniform(-0.2, 0.2, n)
        return s

    def apply(self, cmd: np.ndarray, p: Params, motor: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # policy outputs (collective, differential) around the nominal hover thrust
        hover = self.nominal["mass"] * self.g / 2
        t_cmd = np.stack([hover + cmd[:, 0] - cmd[:, 1], hover + cmd[:, 0] + cmd[:, 1]], axis=1)
        t_cmd = np.clip(t_cmd, 0.0, self.t_max)
        motor = motor + (self.dt / p["tau"])[:, None] * (t_cmd - motor)  # first-order rotor lag
        thrust = motor * np.stack([p["gain_l"], p["gain_r"]], axis=1)
        return thrust, motor

    def dynamics(self, s: np.ndarray, thrust: np.ndarray, p: Params, wind: np.ndarray | None = None) -> np.ndarray:
        x, z, phi, vx, vz, om = s.T
        total = thrust.sum(axis=1)
        w = wind if wind is not None else 0.0
        ax = (-total * np.sin(phi) + w) / p["mass"] - 0.1 * vx
        az = total * np.cos(phi) / p["mass"] - self.g - 0.1 * vz
        al = (thrust[:, 1] - thrust[:, 0]) * p["arm"] / p["inertia"]
        vx2, vz2, om2 = vx + self.dt * ax, vz + self.dt * az, om + self.dt * al
        return np.stack([x + self.dt * vx2, z + self.dt * vz2, phi + self.dt * om2, vx2, vz2, om2], axis=1)

    def disturbance(self, n: int, p: Params, rng) -> np.ndarray:
        return rng.normal(0, 1, n) * p["wind"]

    def failed(self, s: np.ndarray) -> np.ndarray:
        return (np.abs(s[:, 0]) > 3) | (np.abs(s[:, 1]) > 3) | (np.abs(s[:, 2]) > 1.2)

    def reward(self, s: np.ndarray, thrust: np.ndarray) -> np.ndarray:
        return -(s[:, 0] ** 2 + s[:, 1] ** 2 + 0.5 * s[:, 2] ** 2 + 0.01 * (thrust[:, 1] - thrust[:, 0]) ** 2)

    def observe(self, s: np.ndarray, p: Params, rng) -> np.ndarray:
        return s + rng.normal(0, 1, s.shape) * p["noise"][:, None]


ENVS = {"cartpole": CartPole, "quadrotor": PlanarQuadrotor}


def sample_params(env, ranges: dict[str, tuple[float, float]], n: int, rng) -> Params:
    out = {}
    for k, nom in env.nominal.items():
        lo, hi = ranges.get(k, (nom, nom))
        if k in env.integer:
            out[k] = rng.integers(int(lo), int(hi) + 1, n)
        else:
            out[k] = rng.uniform(lo, hi, n)
    return out


def nominal_params(env, n: int) -> Params:
    return {k: np.full(n, v, dtype=int if k in env.integer else float) for k, v in env.nominal.items()}


class TiledRNG:
    """Random generator for a batch made of ``tile`` copies of the same plants.

    Every draw is made for one copy and repeated, so all copies (e.g. ES candidates) see
    identical initial states, sensor noise and disturbances: common random numbers.
    """

    def __init__(self, rng, tile: int):
        self.rng, self.tile = rng, tile

    def _draw(self, fn, size, *args):
        if isinstance(size, tuple):
            base = fn(*args, size=(size[0] // self.tile,) + size[1:])
            return np.tile(base, (self.tile,) + (1,) * (len(size) - 1))
        return np.tile(fn(*args, size=size // self.tile), self.tile)

    def uniform(self, lo, hi, size):
        return self._draw(self.rng.uniform, size, lo, hi)

    def normal(self, mu, sd, size):
        return self._draw(self.rng.normal, size, mu, sd)

    def random(self, size):
        return self._draw(self.rng.random, size)


def rollout(env, policy, params: Params, seed: int = 0, record: bool = False, tile: int = 1) -> Rollout:
    """Run ``policy(obs) -> (N, n_act)`` on N plants in parallel.

    Failed episodes stop accumulating reward and are charged the worst step cost for
    the remaining steps, so falling over is always worse than wobbling.
    """
    rng = TiledRNG(np.random.default_rng(seed), tile)
    n = len(next(iter(params.values())))
    s = env.reset(n, rng)
    max_delay = int(params["delay"].max()) if "delay" in params else 0
    buf = np.zeros((max_delay + 1, n, env.n_act))
    act_state = np.zeros((n, 2)) + (env.nominal.get("mass", 0) * env.g / 2 if env.name == "quadrotor" else 0)
    alive = np.ones(n, bool)
    steps = np.zeros(n, int)
    total = np.zeros(n)
    states, cmds = [], []
    worst = -10.0
    for _ in range(env.horizon):
        obs = env.observe(s, params, rng)
        cmd = policy(obs)
        buf = np.roll(buf, 1, axis=0)
        buf[0] = cmd
        applied, act_state = env.apply(_delayed(buf, params["delay"]), params, act_state)
        d = env.disturbance(n, params, rng)
        if env.name == "cartpole":
            s_next = env.dynamics(s, applied + d, params)
        else:
            s_next = env.dynamics(s, applied, params, wind=d)
        with np.errstate(over="ignore", invalid="ignore"):  # diverged plants are about to be marked failed
            r = np.nan_to_num(np.maximum(env.reward(s_next, applied), worst), nan=worst)
        total += np.where(alive, r, worst)
        if record:  # what a real robot can log: noisy observations and the commands it sent
            states.append(obs.copy())
            cmds.append(cmd.copy())
        s = np.where(alive[:, None], s_next, s)
        steps += alive
        alive &= ~env.failed(s_next)
    return Rollout(total / env.horizon, alive, np.array(states) if record else None,
                   np.array(cmds) if record else None, steps)
