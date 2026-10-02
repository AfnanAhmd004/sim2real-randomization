import numpy as np
import pytest

from s2r import (ADR, ES, CartPole, LinearPolicy, PlanarQuadrotor, evaluate_many, identify, lqr_policy, nominal_params,
                 rollout, sample_params)
from s2r.experiment import sysid_finetune


@pytest.mark.parametrize("Env", [CartPole, PlanarQuadrotor])
def test_lqr_stabilises_the_nominal_plant(Env):
    env = Env()
    r = rollout(env, lqr_policy(env), nominal_params(env, 50), seed=0)
    assert r.success.mean() == 1.0


@pytest.mark.parametrize("Env", [CartPole, PlanarQuadrotor])
def test_doing_nothing_fails(Env):
    env = Env()
    r = rollout(env, lambda o: np.zeros((len(o), env.n_act)), nominal_params(env, 20), seed=0)
    assert r.success.mean() == 0.0 and (r.steps < env.horizon).all()


def test_actuator_delay_is_applied():
    env = CartPole()
    p = nominal_params(env, 2)
    p["delay"] = np.array([0, 3])
    p["push"] = np.zeros(2)
    kick = lambda o: np.where(np.arange(len(o))[:, None] >= 0, 5.0, 0.0) * np.ones((len(o), 1))
    r = rollout(env, kick, p, seed=0, record=True)
    v = r.states[:, :, 2]  # cart velocity
    assert v[2, 0] > 0 and abs(v[2, 1]) < abs(v[2, 0])  # the delayed plant has not felt the push yet


def test_rollout_is_deterministic_per_seed():
    env = CartPole()
    p = sample_params(env, env.reality, 30, np.random.default_rng(0))
    a = rollout(env, lqr_policy(env), p, seed=5).returns
    b = rollout(env, lqr_policy(env), p, seed=5).returns
    assert np.array_equal(a, b)


def test_batched_evaluation_matches_individual_rollouts():
    env = CartPole()
    p = nominal_params(env, 8)
    thetas = np.stack([lqr_policy(env).theta, lqr_policy(env).theta * 0.5])
    batched = evaluate_many(env, thetas, p, seed=1)
    single = [rollout(env, LinearPolicy(env, t), p, seed=1).returns.mean() for t in thetas]
    assert np.allclose(batched, single)


def test_domain_randomisation_beats_nominal_training_on_unseen_plants():
    env = CartPole()
    lqr = lqr_policy(env)
    nom = ES(seed=0, sigma=0.3, lr=0.2).train(env, lqr.theta, lambda n, r: nominal_params(env, n), 25)
    dr = ES(seed=0, sigma=0.3, lr=0.2).train(env, lqr.theta, lambda n, r: sample_params(env, env.guess, n, r), 25)
    real = sample_params(env, env.reality, 300, np.random.default_rng(1))
    s_nom = rollout(env, LinearPolicy(env, nom), real, seed=2).success.mean()
    s_dr = rollout(env, LinearPolicy(env, dr), real, seed=2).success.mean()
    assert s_dr > s_nom + 0.05


def test_adr_expands_ranges_when_the_policy_copes():
    env = CartPole()
    adr = ADR(env, seed=0)
    theta = lqr_policy(env).theta
    for g in range(40):
        adr.probe(g, theta)
    widened = [k for k, (lo, hi) in adr.ranges.items() if hi > lo]
    assert len(widened) >= 3 and 0 < adr.coverage() <= 1
    assert all(env.limits[k][0] <= lo <= hi <= env.limits[k][1] for k, (lo, hi) in adr.ranges.items())


def test_system_identification_recovers_delay_and_predicts_well():
    env = CartPole()
    true = sample_params(env, env.guess, 1, np.random.default_rng(3))
    true["delay"] = np.array([2])
    true["push"] = np.array([0.0])
    true["noise"] = np.array([0.0])
    rng = np.random.default_rng(0)
    lqr = lqr_policy(env)
    r = rollout(env, lambda o: lqr(o) + rng.normal(0, 0.5, (len(o), 1)), true, seed=0, record=True)
    T = int(r.steps[0])
    vals, delay, rms = identify(env, [(r.states[:T, 0], r.commands[:T, 0])])
    assert delay == 2 and rms < 0.05


def test_sysid_finetune_runs_end_to_end():
    env = CartPole()
    plant = sample_params(env, env.reality, 1, np.random.default_rng(11))
    theta, info = sysid_finetune(env, plant, lqr_policy(env).theta, n_tests=2, generations=3)
    assert theta.shape == (4,) and (info is None or {"params", "delay", "rms"} <= set(info))
