# How robust is a pretrained imitation-learning policy to imperfect position readings?

A small, controlled simulation study. I take the pretrained LeRobot **Diffusion Policy** for the **PushT** task, leave the simulator untouched, and corrupt only the 2D agent position that the policy *observes*. Then I measure how closed-loop task success changes on identical start states.

![Same start state (seed 1000), observation noise σ = 0, 12 and 32 from left to right](media/seed1000_sigma_0_12_32.gif)

*Same seed and start state in all three panels. Left: clean observations (solved in 129 steps). Middle: σ = 12. Right: σ = 32 (both fail within the 300-step limit). Rendered at 96×96, the resolution the policy sees, upscaled 3×.*
<!-- TODO(Selim): one sentence on what the agent visibly does in the noisy panels (jittery / drifting / hesitant). -->

## Key findings

- **Small, frame-to-frame jitter in the position reading breaks the policy.** With independent Gaussian noise of σ = 8 workspace units (≈ 1.5 image pixels), success drops from **17/30 to 5/30** on the same 30 start states (paired exact McNemar p = 0.008). Up to σ = 4 (< 1 pixel), no effect is detectable at this sample size.
- **A constant offset of the same size does not.** If the same σ = 8 error is drawn once per episode and held fixed, success is **16/30**, indistinguishable from clean (p = 1.0) and clearly better than i.i.d. jitter (p = 0.007).
- **Interpretation (hypothesis, not proven):** the policy conditions on its last 2 states. I.i.d. noise creates fake frame-to-frame motion (the difference of two noisy readings has std σ√2 ≈ 11 units at σ = 8), whereas a static bias does not. This is consistent with the policy relying on short-horizon motion cues from proprioception, even though the camera image (left clean) also shows the agent.

## Motivation

Policies learned from demonstrations are only as good as the observations they receive at run time. Real sensors drift, jitter and lag. This project measures, in a fully controlled setting, how one widely used pretrained policy reacts to one specific kind of observation error.

## Setup

| | |
|---|---|
| Task | [Gym-PushT](https://github.com/huggingface/gym-pusht): push a T-shaped block onto a fixed target in 2D. Success = target coverage > 95%. Episodes are cut at 300 steps (10 fps). |
| Policy | [`lerobot/diffusion_pusht`](https://huggingface.co/lerobot/diffusion_pusht) at commit `84a7c23178445c6bbf7e1a884ff497017910f653`, frozen. No training or fine-tuning. |
| Policy inputs | RGB image (3×96×96, 84×84 center crop at eval) + agent position (2D), last **2** observations |
| Policy outputs | 2D target positions. It predicts a chunk of 16 actions by denoising, executes 8, then re-plans. |
| Framework | [LeRobot](https://github.com/huggingface/lerobot) 0.7.0 |
| Hardware | MacBook Pro M4 Pro, macOS 15.6, PyTorch MPS backend |

### Checkpoint migration

The published checkpoint predates LeRobot's processor refactor (PR #1452) and does not load in 0.7.0 as is. `scripts/migrate_checkpoint.py` runs LeRobot's official migration script unchanged, except for one type fix: two config fields (`crop_shape`, `down_dims`) are stored as JSON lists and are converted to the tuples 0.7.0 expects. The migration log reports that all weight keys matched. Normalization statistics are extracted from the original weights, and the image preprocessing (84×84 center crop, no resize) is unchanged.

## Method

**Perturbation.** `src/perturbations.py` wraps the vectorized environment that LeRobot itself builds and adds noise to the **raw** agent position (workspace units, 0–512) **before** any normalization. The image, the simulator state, the reward and the success signal are never touched. Two noise types share the same per-step error distribution N(0, σ²) per coordinate:

- **i.i.d.** (sensor jitter): fresh noise at every observation
- **offset** (calibration bias): one draw per episode, held constant

The noise comes from its own NumPy generator, seeded from `(noise_seed, env_seed)`. It never touches PyTorch's global RNG, which drives diffusion sampling. Positions are clipped to the workspace bounds. 1 image pixel ≈ 5.3 workspace units.

**Evaluation harness.** `src/evaluate.py` builds the config, environment, policy and processors exactly as `lerobot-eval` does and runs LeRobot's own `eval_policy`. Only the observation wrapper is added.

**Validity checks.**
- **Parity with the official evaluator:** at σ = 0, the harness reproduces a plain `lerobot-eval` run (seeds 1000–1009) exactly: all 10 episode returns are identical to the last decimal.
- **Unit tests** (`tests/`, 11 tests): σ = 0 is an exact identity, noise is reproducible and has the requested std, the true simulator state and the image are never modified, offset noise is constant within an episode.
- **Determinism:** two independent paired runs gave identical results on the 10 seeds they share.

**Paired design.** Every condition uses the same 30 environment seeds (1000–1029). Before each episode the global seed is reset to the episode's env seed. A given seed therefore gets the same start state, the same diffusion sampling noise and the same observation noise in every condition, so differences come from the perturbation. Conditions are compared seed by seed with an **exact McNemar test**. Success rates carry **95% Wilson intervals**.

**Coverage.** LeRobot's environment does not forward gym-pusht's `coverage` field. Gym-pusht defines `reward = clip(coverage / 0.95, 0, 1)`, so coverage is recovered exactly as `reward × 0.95` below the success threshold, and is capped at 0.95 for successful episodes (`*_coverage_capped` columns).

## Results

### Success vs. noise level (i.i.d. noise)

![Success rate vs. noise σ](results/success_vs_sigma.png)

| Noise σ (units) | ≈ image px | Solved | 95% CI | Seeds lost / gained vs. clean | Exact McNemar p | Mean final coverage (capped) |
|---|---|---|---|---|---|---|
| 0 | 0 | 17/30 | 39–73% | — | — | 0.83 |
| 2 | 0.4 | 16/30 | 36–70% | 8 / 7 | 1.00 | 0.83 |
| 4 | 0.75 | 14/30 | 30–64% | 7 / 4 | 0.55 | 0.83 |
| 8 | 1.5 | 5/30 | 7–34% | 15 / 3 | **0.008** | 0.51 |
| 12 *(pilot)* | 2.3 | 0/10 | 0–28% | 7 / 0 | 0.016 | 0.28 |
| 32 *(pilot)* | 6 | 0/10 | 0–28% | 7 / 0 | 0.016 | 0.06 |

Pilot rows use seeds 1000–1009 only and are exploratory. Across the four main-experiment comparisons, σ = 8 stays significant under a Bonferroni correction (0.008 < 0.05 / 4).

### Ablation: jitter vs. constant bias (σ = 8, same 30 seeds)

| Condition | Solved | Mean final coverage (capped) |
|---|---|---|
| Clean | 17/30 | 0.83 |
| Constant offset, σ = 8 | 16/30 | 0.84 |
| i.i.d. noise, σ = 8 | 5/30 | 0.51 |

| Paired comparison | Only first solved | Only second solved | Exact McNemar p |
|---|---|---|---|
| Offset vs. clean | 4 | 5 | 1.00 |
| Offset vs. i.i.d. | 13 | 2 | 0.007 |

Raw per-episode data: `results/main_paired.csv`, `results/pilot_paired.csv`, `results/ablation_offset.csv`. Summaries: `results/summary.csv`, `results/ablation_summary.csv`.

## Limitations

- **One toy task, one checkpoint.** A 2D pushing task in simulation. Nothing here transfers automatically to other policies, tasks or real robots.
- **Small samples.** 30 episodes per main condition, 10 in the pilot. Effects smaller than roughly 20–30 percentage points could go undetected. The intervals only reflect episode-to-episode variability of a single trained policy, not variation across training runs.
- **The policy is stochastic.** Even on clean observations, the same start state can succeed or fail depending on diffusion sampling noise: seeds 1000–1009 gave 8/10 with LeRobot's default seeding and 7/10 with per-episode seeding. The paired design controls for this, but absolute success rates should be read with their intervals.
- **Simplified error model.** Gaussian noise on position only. The camera image is clean, and there is no latency, dropout or correlated drift.
- **Mechanism.** The fake-velocity explanation is supported by one ablation at one noise level. It is not a proven causal account.
- **Baseline not compared to published numbers.** I verified the migration (weights, statistics, preprocessing) and parity with LeRobot's evaluator, but not the checkpoint's originally reported success rate.

## What I would test next with real robot data

- **Temporal structure of sensor errors:** jitter vs. bias vs. drift vs. latency (timestamp misalignment between camera and proprioception), using the same paired protocol.
- **Mitigations:** filtering proprioception (low-pass or Kalman) before the policy, training with noise augmentation, or a single-step state history. Does robustness improve without hurting clean performance?
- **Modality reliance:** does the policy recover when the state is dropped entirely, i.e. how much does it actually use the camera for self-localization?
- **Data quality in demonstrations:** the same perturbations applied to training data rather than at test time.
- **Hardware validation:** none of this replaces running the protocol on a physical robot.

## Reproduce

Tested on Apple Silicon. Make sure your terminal runs natively (`uname -m` prints `arm64`, not `x86_64` under Rosetta), otherwise PyTorch cannot use MPS.

```bash
# 1. Environment
conda create -n robot-bench python=3.12 -y
conda activate robot-bench
conda install -c conda-forge ffmpeg -y
pip install 'lerobot[evaluation,diffusion,pusht]==0.7.0' matplotlib pytest
# exact versions used: requirements-lock.txt

# 2. Migrate the checkpoint (writes checkpoints/diffusion_pusht_migrated, about 1 GB, git-ignored)
python scripts/migrate_checkpoint.py

# 3. Tests
pytest -q

# 4. Parity check: sigma=0 must reproduce plain lerobot-eval exactly
COMMON="--policy.path=checkpoints/diffusion_pusht_migrated --env.type=pusht --eval.batch_size=1 \
  --policy.device=mps --policy.use_amp=false --seed=1000"
python src/evaluate.py --mode parity --sigmas 0 --csv results/parity_sigma0.csv \
  $COMMON --eval.n_episodes=10 --output_dir=outputs/parity_sigma0

# 5. Experiments (paired mode)
python src/evaluate.py --sigmas 0 2 4 8 --csv results/main_paired.csv \
  $COMMON --eval.n_episodes=30 --output_dir=outputs/main_paired
python src/evaluate.py --sigmas 0 12 32 --n-videos 2 --csv results/pilot_paired.csv \
  $COMMON --eval.n_episodes=10 --output_dir=outputs/pilot_paired
python src/evaluate.py --sigmas 8 --noise-type offset --csv results/ablation_offset.csv \
  $COMMON --eval.n_episodes=30 --output_dir=outputs/ablation_offset

# 6. Analysis (tables + plot)
python src/analyze.py
```

`evaluate.py` appends to the CSV, so delete an existing file before regenerating it. Measured runtime on an M4 Pro (MPS): about 10–25 s for a successful episode and about 34–36 s for an episode that runs the full 300 steps. The main experiment (4 levels × 30 episodes) took roughly an hour.

## Repository layout

```text
scripts/migrate_checkpoint.py   checkpoint conversion to the LeRobot 0.7.0 format
src/perturbations.py            noise functions and the observation wrapper
src/evaluate.py                 evaluation harness (parity and paired modes), one CSV row per episode
src/analyze.py                  Wilson CIs, exact McNemar tests, determinism check, plot
tests/test_perturbations.py     unit tests, no simulator needed
results/                        raw per-episode CSVs, summaries, plot
media/                          demo GIF
```

## Versions

Python 3.12.16 · PyTorch 2.15.0 · LeRobot 0.7.0 · gym-pusht 0.1.8 · Gymnasium 1.4.0 · NumPy 2.2.6 · pandas 2.3.3 · Matplotlib 3.11.2 · ffmpeg 5.1 (conda-forge).

Known harmless warnings: `objc[...] Class ... is implemented in both ...` comes from OpenCV, PyAV and pygame each bundling their own copies of FFmpeg and SDL libraries.

`results/parity_sigma0.csv` was produced before the coverage columns were added. Only its `success` and `sum_reward` columns are used.

## Acknowledgements

Built on [LeRobot](https://github.com/huggingface/lerobot), [Gym-PushT](https://github.com/huggingface/gym-pusht) and the pretrained checkpoint [`lerobot/diffusion_pusht`](https://huggingface.co/lerobot/diffusion_pusht). Diffusion Policy: Chi et al., *Diffusion Policy: Visuomotor Policy Learning via Action Diffusion*, RSS 2023.
