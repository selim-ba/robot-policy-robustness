"""Observation perturbations for robustness evaluation on PushT.

Only the agent-facing 2D position is perturbed. Pixels, the simulator state,
rewards, success and `info` are never modified.

Noise types (same per-step marginal error N(0, sigma^2) per coordinate):
  iid    : fresh noise at every observation (white measurement noise)
  offset : one draw per episode, held constant (calibration bias; no fake velocity)
"""
from __future__ import annotations

import gymnasium as gym
import numpy as np

AGENT_POS_LOW, AGENT_POS_HIGH = 0.0, 512.0  # PushT observation_space bounds
NOISE_TYPES = ("iid", "offset")


def add_position_noise(
    pos: np.ndarray,
    sigma: float,
    rng: np.random.Generator | None,
    low: float = AGENT_POS_LOW,
    high: float = AGENT_POS_HIGH,
) -> np.ndarray:
    """Return pos + N(0, sigma^2) i.i.d. per coordinate, clipped to [low, high].

    sigma is in raw environment units (workspace is 512 x 512).
    sigma == 0 returns an unmodified copy and consumes no randomness.
    """
    if sigma < 0:
        raise ValueError(f"sigma must be >= 0, got {sigma}")
    pos = np.asarray(pos)
    if sigma == 0:
        return pos.copy()
    if rng is None:
        raise ValueError("sigma > 0 requires a seeded np.random.Generator")
    noisy = pos + rng.normal(0.0, sigma, size=pos.shape)
    return np.clip(noisy, low, high).astype(pos.dtype, copy=False)


class NoisyAgentPos(gym.vector.VectorObservationWrapper):
    """Perturb the agent-position entry of a vector env's observations, at reset and every step.

    Each sub-env gets its own RNG, re-seeded at every reset from (noise_seed, env_seed).
    An episode's noise therefore depends only on its env seed (not on earlier episodes)
    and never touches torch's global RNG used by diffusion sampling.
    """

    def __init__(
        self,
        env: gym.vector.VectorEnv,
        sigma: float,
        noise_seed: int = 0,
        pos_key: str = "agent_pos",
        noise_type: str = "iid",
    ):
        super().__init__(env)
        if sigma < 0:
            raise ValueError(f"sigma must be >= 0, got {sigma}")
        if noise_type not in NOISE_TYPES:
            raise ValueError(f"noise_type must be one of {NOISE_TYPES}, got {noise_type!r}")
        self.sigma = float(sigma)
        self.noise_seed = int(noise_seed)
        self.pos_key = pos_key
        self.noise_type = noise_type
        self._rngs: list[np.random.Generator | None] | None = None
        self._offsets: list[np.ndarray] | None = None
        self.last_clean_pos: np.ndarray | None = None  # for tests / debugging
        self.last_noisy_pos: np.ndarray | None = None

    def reset(self, *, seed=None, options=None):
        if seed is None:
            if self.sigma > 0:
                raise ValueError("NoisyAgentPos needs explicit env seeds for reproducible noise")
            env_seeds = [None] * self.num_envs
        elif isinstance(seed, int):
            env_seeds = [seed + i for i in range(self.num_envs)]  # gymnasium's convention
        else:
            env_seeds = list(seed)
        self._rngs = [
            np.random.default_rng([self.noise_seed, s]) if s is not None else None for s in env_seeds
        ]
        # Offsets are drawn before the reset observation is produced (it is perturbed too).
        self._offsets = [
            rng.normal(0.0, self.sigma, size=2) if (rng is not None and self.sigma > 0) else np.zeros(2)
            for rng in self._rngs
        ]
        return super().reset(seed=seed, options=options)

    def _perturb(self, i: int, pos: np.ndarray) -> np.ndarray:
        if self.noise_type == "offset":
            if self.sigma == 0:
                return pos.copy()
            return np.clip(pos + self._offsets[i], AGENT_POS_LOW, AGENT_POS_HIGH).astype(pos.dtype)
        return add_position_noise(pos, self.sigma, self._rngs[i])

    def observations(self, observations: dict) -> dict:
        if self._rngs is None:
            raise RuntimeError("reset() must be called before stepping")
        if self.pos_key not in observations:
            layout = {k: getattr(v, "shape", type(v).__name__) for k, v in observations.items()}
            raise KeyError(f"{self.pos_key!r} not in observations; got {layout}")
        clean = np.asarray(observations[self.pos_key])  # shape (num_envs, 2)
        noisy = np.stack([self._perturb(i, clean[i]) for i in range(self.num_envs)])
        self.last_clean_pos, self.last_noisy_pos = clean.copy(), noisy.copy()
        return {**observations, self.pos_key: noisy}