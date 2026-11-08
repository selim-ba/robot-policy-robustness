"""Unit tests for observation perturbations. No simulator or policy needed."""
import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from perturbations import AGENT_POS_HIGH, AGENT_POS_LOW, NoisyAgentPos, add_position_noise

POS = np.array([131.0, 258.0])  # agent_pos at reset(seed=1000), from inspect_env.py


# ---------- add_position_noise (pure function) ----------

def test_zero_sigma_is_exact_identity():
    out = add_position_noise(POS, 0.0, rng=None)
    assert np.array_equal(out, POS)
    assert out.dtype == POS.dtype
    assert out is not POS  # returns a copy, never the same object


def test_same_seed_reproducible_different_seed_differs():
    a = add_position_noise(POS, 10.0, np.random.default_rng(0))
    b = add_position_noise(POS, 10.0, np.random.default_rng(0))
    c = add_position_noise(POS, 10.0, np.random.default_rng(1))
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)


def test_noise_statistics_match_sigma():
    rng = np.random.default_rng(42)
    center = np.array([256.0, 256.0])  # far from bounds, so clipping never triggers
    samples = np.stack([add_position_noise(center, 10.0, rng) for _ in range(10_000)])
    offsets = samples - center
    assert np.allclose(offsets.mean(axis=0), 0.0, atol=0.5)
    assert np.allclose(offsets.std(axis=0), 10.0, rtol=0.05)


def test_output_clipped_to_workspace():
    rng = np.random.default_rng(0)
    corner = np.array([1.0, 511.0])
    out = np.stack([add_position_noise(corner, 100.0, rng) for _ in range(1_000)])
    assert out.min() >= AGENT_POS_LOW and out.max() <= AGENT_POS_HIGH


def test_input_not_mutated():
    pos = POS.copy()
    add_position_noise(pos, 10.0, np.random.default_rng(0))
    assert np.array_equal(pos, POS)


def test_invalid_arguments_raise():
    with pytest.raises(ValueError):
        add_position_noise(POS, -1.0, np.random.default_rng(0))
    with pytest.raises(ValueError):
        add_position_noise(POS, 5.0, rng=None)


# ---------- NoisyAgentPos wrapper, on a tiny fake PushT-like env ----------

class FakePushT(gym.Env):
    """Same obs/info structure as gym-pusht: obs = agent_pos + pixels, info['pos_agent'] = true state."""
    observation_space = spaces.Dict({
        "agent_pos": spaces.Box(0.0, 512.0, (2,), np.float64),
        "pixels": spaces.Box(0, 255, (4, 4, 3), np.uint8),
    })
    action_space = spaces.Box(0.0, 512.0, (2,), np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.pos = POS.copy()
        return self._obs(), {"pos_agent": self.pos.copy()}

    def step(self, action):
        self.pos = np.asarray(action, dtype=np.float64)  # PushT actions are target positions
        return self._obs(), 0.0, False, False, {"pos_agent": self.pos.copy()}

    def _obs(self):
        return {"agent_pos": self.pos.copy(), "pixels": np.full((4, 4, 3), 7, np.uint8)}


def make_vec(sigma, noise_seed=0):
    return NoisyAgentPos(gym.vector.SyncVectorEnv([FakePushT]), sigma=sigma, noise_seed=noise_seed)


def test_wrapper_zero_sigma_matches_unwrapped_env():
    clean_env = gym.vector.SyncVectorEnv([FakePushT])
    noisy_env = make_vec(sigma=0.0)
    obs_c, _ = clean_env.reset(seed=1000)
    obs_n, _ = noisy_env.reset(seed=1000)
    assert np.array_equal(obs_c["agent_pos"], obs_n["agent_pos"])
    action = np.array([[200.0, 300.0]], dtype=np.float32)
    obs_c, *_ = clean_env.step(action)
    obs_n, *_ = noisy_env.step(action)
    assert np.array_equal(obs_c["agent_pos"], obs_n["agent_pos"])


def test_wrapper_perturbs_observation_but_not_true_state_or_pixels():
    env = make_vec(sigma=10.0)
    obs, info = env.reset(seed=1000)
    assert not np.array_equal(obs["agent_pos"][0], POS)        # observation is noisy
    assert np.array_equal(info["pos_agent"][0], POS)           # ground truth untouched
    assert np.all(obs["pixels"] == 7)                          # image untouched
    obs, _, _, _, info = env.step(np.array([[200.0, 300.0]], dtype=np.float32))
    assert np.array_equal(info["pos_agent"][0], [200.0, 300.0])
    assert not np.array_equal(obs["agent_pos"][0], [200.0, 300.0])


def test_wrapper_noise_depends_only_on_env_seed():
    env = make_vec(sigma=10.0)
    first, _ = env.reset(seed=1003)
    env.reset(seed=1007)                  # an unrelated episode in between
    again, _ = env.reset(seed=1003)
    assert np.array_equal(first["agent_pos"], again["agent_pos"])
    other_noise, _ = make_vec(sigma=10.0, noise_seed=1).reset(seed=1003)
    assert not np.array_equal(first["agent_pos"], other_noise["agent_pos"])


def test_wrapper_refuses_unseeded_noise():
    with pytest.raises(ValueError):
        make_vec(sigma=10.0).reset(seed=None)