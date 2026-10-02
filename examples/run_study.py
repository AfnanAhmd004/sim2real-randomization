"""Train every transfer strategy on both robots and test on 1,000 held-out "real" plants.

    python examples/run_study.py           # ~12 min on a laptop CPU
    python examples/run_study.py --quick   # ~2 min, smaller budgets
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

from s2r import CartPole, LinearPolicy, PlanarQuadrotor, rollout, sample_params
from s2r.experiment import evaluate, sysid_finetune, train_all


def per_plant_success(env, theta, plant: dict, episodes: int = 40) -> float:
    rep = {k: np.repeat(v, episodes) for k, v in plant.items()}
    return float(rollout(env, LinearPolicy(env, theta), rep, seed=99).success.mean())


def heatmap(env, policies: dict, x: str, xs, y: str, ys, n: int = 60) -> dict:
    rng = np.random.default_rng(0)
    out = {}
    for name, theta in policies.items():
        grid = np.zeros((len(ys), len(xs)))
        for i, yv in enumerate(ys):
            for j, xv in enumerate(xs):
                p = sample_params(env, {k: (v, v) for k, v in env.nominal.items()}, n, rng)
                p[x] = np.full(n, xv, dtype=p[x].dtype)
                p[y] = np.full(n, yv, dtype=p[y].dtype)
                grid[i, j] = rollout(env, LinearPolicy(env, theta), p, seed=3).success.mean()
        out[name] = grid
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    gens, n_sysid = (40, 6) if args.quick else (150, 30)
    os.makedirs("docs", exist_ok=True)
    report = {}
    for env in (CartPole(), PlanarQuadrotor()):
        print(f"\n=== {env.name}: trained in simulation, tested on 1,000 'real' plants ===")
        trained = train_all(env, generations=gens)
        adr = trained.pop("_adr")
        real = sample_params(env, env.reality, 1000, np.random.default_rng(2026))
        print(f"{'strategy':<28}{'real success':>14}{'real return':>13}{'nominal return':>16}")
        rows = {}
        for name, theta in trained.items():
            e = evaluate(env, theta, real)
            rows[name] = e
            print(f"{name:<28}{e['real_success']:>14.1%}{e['real_return']:>13.3f}{e['nominal_return']:>16.4f}")
        print(f"ADR expanded its ranges to cover {adr.coverage():.0%} of the reality distribution")

        # System identification + fine-tuning, plant by plant, on plants where DR is weakest
        rng = np.random.default_rng(7)
        cands = sample_params(env, env.reality, 200, rng)
        dr = trained["ES, domain randomisation"]
        scores = [per_plant_success(env, dr, {k: v[i:i + 1] for k, v in cands.items()}, 10) for i in range(200)]
        hard = np.argsort(scores)[:n_sysid]
        before, after, adr_s, est = [], [], [], []
        for i in hard:
            plant = {k: v[i:i + 1] for k, v in cands.items()}
            theta, info = sysid_finetune(env, plant, dr, generations=gens // 3 if not args.quick else 15)
            before.append(per_plant_success(env, dr, plant))
            after.append(per_plant_success(env, theta, plant))
            adr_s.append(per_plant_success(env, trained["ES, automatic DR"], plant))
            if info:
                est.append(abs(info["delay"] - int(plant["delay"][0])) == 0)
        print(f"{n_sysid} hardest plants for DR: DR {np.mean(before):.1%} -> DR + SysID fine-tune {np.mean(after):.1%} "
              f"(ADR {np.mean(adr_s):.1%}); delay identified exactly in {np.mean(est):.0%}")
        rows["sysid"] = {"plants": n_sysid, "dr": float(np.mean(before)), "dr_sysid": float(np.mean(after)),
                         "adr": float(np.mean(adr_s)), "delay_exact": float(np.mean(est))}
        rows["adr_ranges"] = adr.ranges
        rows["adr_coverage"] = adr.coverage()
        report[env.name] = rows

        if env.name == "cartpole":
            xs, ys = np.linspace(0.25, 1.2, 8), [0, 1, 2, 3, 4]
            maps = heatmap(env, {k: trained[k] for k in ("LQR (nominal model)", "ES, domain randomisation",
                                                           "ES, automatic DR")}, "half_len", xs, "delay", ys)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), sharey=True)
            for ax, (name, g) in zip(axes, maps.items()):
                im = ax.imshow(g, origin="lower", vmin=0, vmax=1, cmap="RdYlGn", aspect="auto",
                               extent=[xs[0], xs[-1], -0.5, len(ys) - 0.5])
                ax.set_title(name, fontsize=10)
                ax.set_xlabel("pole half-length (m)")
                ax.scatter([env.nominal["half_len"]], [0], marker="*", s=120, color="k")
            axes[0].set_ylabel("actuator delay (steps of 20 ms)")
            fig.colorbar(im, ax=axes, label="success rate", shrink=0.9)
            fig.suptitle("Cart-pole robustness away from the nominal model (★)", y=1.02)
            fig.savefig("docs/robustness.png", dpi=130, bbox_inches="tight")
    with open("docs/results.json", "w") as f:
        json.dump(report, f, indent=1, default=float)
    print("\nsaved docs/robustness.png and docs/results.json")


if __name__ == "__main__":
    main()
