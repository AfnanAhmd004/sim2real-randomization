"""The four transfer strategies, run end to end for one environment."""
from __future__ import annotations

import numpy as np

from .envs import nominal_params, rollout, sample_params
from .sysid import identify
from .train import ADR, ES, LinearPolicy, lqr_policy


def train_all(env, generations: int = 150, seed: int = 0) -> dict:
    lqr = lqr_policy(env)
    out = {"LQR (nominal model)": lqr.theta}
    es = ES(seed=seed, sigma=0.3, lr=0.2)
    out["ES, nominal sim"] = es.train(env, lqr.theta, lambda n, r: nominal_params(env, n), generations)
    es = ES(seed=seed, sigma=0.3, lr=0.2)
    out["ES, domain randomisation"] = es.train(env, lqr.theta, lambda n, r: sample_params(env, env.guess, n, r),
                                               generations)
    adr = ADR(env, seed=seed)
    es = ES(seed=seed, sigma=0.3, lr=0.2)
    out["ES, automatic DR"] = es.train(env, lqr.theta, adr.sampler, 2 * generations, on_generation=adr.probe)
    out["_adr"] = adr
    return out


def sysid_finetune(env, real: dict, start_theta: np.ndarray, n_tests: int = 3, generations: int = 40, seed: int = 0):
    """Log a few safe test runs on the real plant, identify it, fine-tune around the estimate."""
    rng = np.random.default_rng(seed)
    explore = 0.5 if env.name == "cartpole" else 0.15
    base = LinearPolicy(env, start_theta)
    eps = []
    for k in range(n_tests):
        r = rollout(env, lambda o: base(o) + rng.normal(0, explore, (len(o), env.n_act)), real, seed=1000 + k, record=True)
        T = int(r.steps[0])
        if T > 20:
            eps.append((r.states[:T, 0], r.commands[:T, 0]))
    if not eps:
        return start_theta, None
    vals, delay, rms = identify(env, eps)
    ranges = {k: (0.9 * v, 1.1 * v) for k, v in vals.items()}
    ranges["delay"] = (delay, delay)
    for k in ("noise", "push", "wind"):
        if k in env.guess:
            ranges[k] = env.guess[k]
    es = ES(seed=seed, sigma=0.2, lr=0.1)
    theta = es.train(env, start_theta, lambda n, r: sample_params(env, ranges, n, r), generations)
    return theta, {"params": vals, "delay": delay, "rms": rms}


def evaluate(env, theta: np.ndarray, plants: dict, seed: int = 7) -> dict:
    r = rollout(env, LinearPolicy(env, theta), plants, seed)
    nom = rollout(env, LinearPolicy(env, theta), nominal_params(env, 200), seed)
    return {"real_success": float(r.success.mean()), "real_return": float(r.returns.mean()),
            "nominal_success": float(nom.success.mean()), "nominal_return": float(nom.returns.mean())}
