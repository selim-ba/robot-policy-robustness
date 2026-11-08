"""Paired robustness evaluation of a pretrained LeRobot policy on PushT (LeRobot 0.7.0).

Config, env, policy and processors are built exactly as `lerobot-eval` builds them, the env is
wrapped with NoisyAgentPos, and episodes run through LeRobot's own `eval_policy`.
Writes one CSV row per episode. LeRobot CLI args are passed through unchanged.

Modes:
  parity : one eval_policy call over all seeds, global seed set once (= lerobot-eval behaviour).
           With --sigmas 0 it reproduces a plain lerobot-eval run exactly (verified).
  paired : one episode per call, set_seed(env_seed) before each, so a given seed gets the same
           start state, diffusion noise and observation noise in every condition.

Coverage: LeRobot's env does not forward info["coverage"]. gym-pusht defines
reward = clip(coverage / 0.95, 0, 1), so coverage = reward * 0.95 exactly when coverage < 0.95,
and is only known to be >= 0.95 above the success threshold ("capped" columns).
"""
import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np
import torch

import lerobot
from lerobot.configs import parser
from lerobot.configs.eval import EvalPipelineConfig
from lerobot.scripts import lerobot_eval as le  # reuse the exact functions eval_main uses

from perturbations import NoisyAgentPos

SUCCESS_THRESHOLD = 0.95  # gym_pusht/envs/pusht.py: success_threshold, reward = clip(cov / 0.95, 0, 1)

CSV_FIELDS = [
    "condition", "sigma", "mode", "env_seed", "noise_seed", "episode_index", "success",
    "final_coverage_capped", "max_coverage_capped", "sum_reward", "max_reward", "episode_steps",
    "runtime_seconds", "lerobot_version", "checkpoint",
]
OWN_ARGS: argparse.Namespace | None = None


class EvalEnv(NoisyAgentPos):
    """NoisyAgentPos + per-episode record of coverage (derived from reward) and step count. Batch size 1."""

    def __init__(self, env, sigma: float, noise_seed: int, pos_key: str):
        super().__init__(env, sigma=sigma, noise_seed=noise_seed, pos_key=pos_key)
        self.episodes: list[dict] = []

    def reset(self, *, seed=None, options=None):
        out = super().reset(seed=seed, options=options)
        env_seed = seed[0] if isinstance(seed, (list, tuple)) else seed
        if not self.episodes:
            print(f"[check] seed={env_seed} clean {self.pos_key}={self.last_clean_pos[0]} "
                  f"observed={self.last_noisy_pos[0]}")
        self.episodes.append({
            "env_seed": env_seed, "episode_steps": 0,
            "final_coverage_capped": float("nan"), "max_coverage_capped": 0.0,
        })
        return out

    def step(self, actions):
        obs, rewards, terminated, truncated, infos = super().step(actions)
        ep = self.episodes[-1]
        ep["episode_steps"] += 1
        cov = float(np.asarray(rewards)[0]) * SUCCESS_THRESHOLD  # exact below 0.95, capped at 0.95
        ep["final_coverage_capped"] = cov
        ep["max_coverage_capped"] = max(ep["max_coverage_capped"], cov)
        return obs, rewards, terminated, truncated, infos

    def __getattr__(self, name):
        # Forward attributes LeRobot may use (e.g. env.call("render")) to the wrapped vector env.
        if name.startswith("_") or name == "env":
            raise AttributeError(name)
        return getattr(self.env, name)


def build_components(cfg: EvalPipelineConfig):
    """Same order as lerobot_eval.eval_main: seed -> env -> env processors -> policy -> processors."""
    le.set_seed(cfg.seed)
    envs = le.make_env(
        cfg.env, n_envs=cfg.eval.batch_size,
        use_async_envs=cfg.eval.use_async_envs, trust_remote_code=cfg.trust_remote_code,
    )
    vec_env = next(e for tasks in envs.values() for e in tasks.values())
    env_pre, env_post = le.make_env_pre_post_processors(env_cfg=cfg.env, policy_cfg=cfg.policy)

    descriptor = getattr(vec_env, "descriptor", None)
    if descriptor is not None:
        le.validate_checkpoint(descriptor, cfg.policy, cfg.rename_map,
                               env_preprocessor=env_pre, env_postprocessor=env_post)

    policy = le.make_policy(cfg=cfg.policy, env_cfg=cfg.env, rename_map=cfg.rename_map)
    policy.eval()
    pre, post = le.make_pre_post_processors(
        policy_cfg=cfg.policy,
        pretrained_path=cfg.policy.pretrained_path,
        pretrained_revision=cfg.policy.pretrained_revision,
        preprocessor_overrides={
            "device_processor": {"device": str(policy.config.device)},
            "rename_observations_processor": {"rename_map": cfg.rename_map},
        },
    )
    return vec_env, policy, env_pre, env_post, pre, post


def run_condition(cfg, components, sigma: float, own) -> list[dict]:
    vec_env, policy, env_pre, env_post, pre, post = components
    env = EvalEnv(vec_env, sigma=sigma, noise_seed=own.noise_seed, pos_key=own.pos_key)
    common = dict(env=env, policy=policy, env_preprocessor=env_pre, env_postprocessor=env_post,
                  preprocessor=pre, postprocessor=post)
    seeds = list(range(cfg.seed, cfg.seed + cfg.eval.n_episodes))

    results, runtimes = [], []
    with torch.no_grad():
        if own.mode == "parity":
            t0 = time.perf_counter()
            info = le.eval_policy(**common, n_episodes=len(seeds), start_seed=seeds[0])
            results = info["per_episode"]
            runtimes = [float("nan")] * len(results)  # not measurable per episode in one call
            print(f"[parity] total {time.perf_counter() - t0:.1f}s")
        else:
            for i, s in enumerate(seeds):
                le.set_seed(s)
                vdir = Path(cfg.output_dir) / "videos" / f"sigma_{sigma:g}" / f"seed_{s}" if i < own.n_videos else None
                t0 = time.perf_counter()
                info = le.eval_policy(**common, n_episodes=1, start_seed=s,
                                      max_episodes_rendered=1 if vdir else 0, videos_dir=vdir)
                runtimes.append(time.perf_counter() - t0)
                results.append(info["per_episode"][0])
                print(f"  sigma={sigma:g} seed={s} success={bool(results[-1]['success'])} "
                      f"cov_capped={env.episodes[-1]['final_coverage_capped']:.3f} "
                      f"steps={env.episodes[-1]['episode_steps']} ({runtimes[-1]:.1f}s)")

    assert len(results) == len(env.episodes), (len(results), len(env.episodes))
    rows = []
    for idx, (res, rec, rt) in enumerate(zip(results, env.episodes, runtimes)):
        rows.append({
            "condition": f"agent_pos_noise_sigma_{sigma:g}", "sigma": sigma, "mode": own.mode,
            "env_seed": rec["env_seed"], "noise_seed": own.noise_seed, "episode_index": idx,
            "success": bool(res["success"]),
            "final_coverage_capped": rec["final_coverage_capped"],
            "max_coverage_capped": rec["max_coverage_capped"],
            "sum_reward": float(res["sum_reward"]), "max_reward": float(res["max_reward"]),
            "episode_steps": rec["episode_steps"], "runtime_seconds": rt,
            "lerobot_version": lerobot.__version__, "checkpoint": str(cfg.policy.pretrained_path),
        })
    return rows


def append_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


@parser.wrap()
def main(cfg: EvalPipelineConfig):
    own = OWN_ARGS
    if cfg.eval.batch_size != 1:
        raise ValueError("evaluate.py assumes --eval.batch_size=1")
    Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)
    components = build_components(cfg)
    try:
        for sigma in own.sigmas:
            rows = run_condition(cfg, components, sigma, own)
            append_csv(own.csv, rows)  # saved after each condition, so partial progress survives
            n_ok = sum(r["success"] for r in rows)
            print(f"== sigma={sigma:g}: {n_ok}/{len(rows)} success, "
                  f"mean final coverage (capped) {np.mean([r['final_coverage_capped'] for r in rows]):.3f}")
    finally:
        components[0].close()


if __name__ == "__main__":
    p = argparse.ArgumentParser(allow_abbrev=False)
    p.add_argument("--sigmas", type=float, nargs="+", required=True, help="agent_pos noise std, raw env units")
    p.add_argument("--noise-seed", type=int, default=0)
    p.add_argument("--mode", choices=["paired", "parity"], default="paired")
    p.add_argument("--csv", type=Path, required=True)
    p.add_argument("--n-videos", type=int, default=0, help="render the first N episodes per condition (paired mode)")
    p.add_argument("--pos-key", default="observation.state",
                   help="observation key holding the raw 2D agent position (LeRobot env naming)")
    OWN_ARGS, remaining = p.parse_known_args()
    sys.argv = [sys.argv[0], *remaining]  # leave only LeRobot's own args for its parser
    main()