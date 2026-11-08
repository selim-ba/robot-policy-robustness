"""Analyze paired robustness runs.

Per noise level: success rate with 95% Wilson CI, mean capped coverage, and a paired
comparison against sigma=0 on the same env seeds (seeds lost/gained, exact McNemar p).
Also checks that paired-mode runs are deterministic across invocations.
"""
import argparse
from math import comb, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import NullLocator, PercentFormatter

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
MAIN_COLOR, PILOT_COLOR = "#2a78d6", "#8a8984"


def wilson_ci(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return float("nan"), float("nan")
    p, z2 = k / n, z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    return max(0.0, center - half), min(1.0, center + half)


def mcnemar_exact(lost: int, gained: int) -> float:
    """Two-sided exact McNemar (binomial sign test on discordant pairs)."""
    n = lost + gained
    if n == 0:
        return 1.0
    k = min(lost, gained)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2**n)


def summarize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    clean = df[df.sigma == 0].set_index("env_seed")[["success"]]
    rows = []
    for sigma, g in df.groupby("sigma"):
        k, n = int(g.success.sum()), len(g)
        lo, hi = wilson_ci(k, n)
        row = {
            "source": source, "sigma": sigma, "n": n, "successes": k,
            "success_rate": k / n, "ci95_low": lo, "ci95_high": hi,
            "mean_final_coverage_capped": g.final_coverage_capped.mean(),
            "median_steps": g.episode_steps.median(),
        }
        if sigma != 0 and not clean.empty:
            paired = g.set_index("env_seed")[["success"]].join(clean, rsuffix="_clean", how="inner")
            lost = int((paired.success_clean & ~paired.success).sum())
            gained = int((~paired.success_clean & paired.success).sum())
            row.update(n_paired=len(paired), seeds_lost=lost, seeds_gained=gained,
                       mcnemar_p=mcnemar_exact(lost, gained))
        rows.append(row)
    return pd.DataFrame(rows)


def check_determinism(a: pd.DataFrame, b: pd.DataFrame) -> None:
    """Paired-mode sigma=0 rows sharing an env_seed must be identical across runs."""
    cols = ["env_seed", "success", "sum_reward", "episode_steps"]
    m = a[a.sigma == 0][cols].merge(b[b.sigma == 0][cols], on="env_seed", suffixes=("_a", "_b"))
    if m.empty:
        print("[determinism] no overlapping sigma=0 seeds to compare")
        return
    same = (m.sum_reward_a == m.sum_reward_b) & (m.episode_steps_a == m.episode_steps_b)
    print(f"[determinism] sigma=0, {len(m)} shared seeds: {'IDENTICAL' if same.all() else 'MISMATCH'}")
    if not same.all():
        print(m[~same])


def plot(summary: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.2), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    for source, color, label, filled in [
        ("main", MAIN_COLOR, "30 episodes per level", True),
        ("pilot", PILOT_COLOR, "Pilot, 10 episodes per level", False),
    ]:
        s = summary[(summary.source == source) & ((source == "main") | (summary.sigma > 0))]
        s = s.sort_values("sigma")
        if s.empty:
            continue
        yerr = [s.success_rate - s.ci95_low, s.ci95_high - s.success_rate]
        ax.errorbar(s.sigma, s.success_rate, yerr=yerr, color=color, lw=2 if filled else 0,
                    marker="o", ms=8, mfc=color if filled else SURFACE, mec=color, mew=2,
                    capsize=4, elinewidth=1.5, label=label)

    ax.set_xscale("symlog", linthresh=2)
    ticks = sorted(summary.sigma.unique())
    ax.set_xticks(ticks, [f"{t:g}" for t in ticks])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_ylim(-0.03, 1.03)
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.set_xlabel("Position noise σ (workspace units; 1 image pixel ≈ 5.3 units)", color=INK_2)
    ax.set_ylabel("Episodes solved (95% CI)", color=INK_2)
    ax.set_title("Task success vs. noise in the robot's position reading",
                 color=INK, fontsize=12, loc="left", pad=12)

    ax.grid(axis="y", color=GRID, lw=0.8)
    ax.tick_params(colors=INK_2)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.legend(frameon=False, labelcolor=INK_2, fontsize=9, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=200, facecolor=SURFACE)
    print(f"[plot] {path}")


def compare_paired(a: pd.DataFrame, b: pd.DataFrame, name_a: str, name_b: str) -> dict:
    """Paired comparison of two conditions on shared env seeds (exact McNemar)."""
    m = a.set_index("env_seed")[["success"]].join(
        b.set_index("env_seed")[["success"]], lsuffix="_a", rsuffix="_b", how="inner")
    a_only = int((m.success_a & ~m.success_b).sum())
    b_only = int((~m.success_a & m.success_b).sum())
    return {"condition_a": name_a, "condition_b": name_b, "n_paired": len(m),
            "solved_a": int(m.success_a.sum()), "solved_b": int(m.success_b.sum()),
            "only_a_solved": a_only, "only_b_solved": b_only,
            "mcnemar_p": mcnemar_exact(a_only, b_only)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--main", type=Path, default=Path("results/main_paired.csv"))
    p.add_argument("--pilot", type=Path, default=Path("results/pilot_paired.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("results"))
    p.add_argument("--ablation", type=Path, default=Path("results/ablation_offset.csv"))
    args = p.parse_args()

    main_df = pd.read_csv(args.main)
    parts = [summarize(main_df, "main")]
    if args.pilot.exists():
        pilot = pd.read_csv(args.pilot)
        parts.append(summarize(pilot, "pilot"))
        check_determinism(main_df, pilot)
    summary = pd.concat(parts, ignore_index=True)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "summary.csv", index=False)
    with pd.option_context("display.width", 200, "display.max_columns", None, "display.precision", 3):
        print(summary)
    plot(summary, args.out_dir / "success_vs_sigma.png")

    if args.ablation.exists():
        abl = pd.read_csv(args.ablation)
        sigma = float(abl.sigma.iloc[0])
        comps = pd.DataFrame([
            compare_paired(abl, main_df[main_df.sigma == 0], f"offset sigma={sigma:g}", "clean"),
            compare_paired(abl, main_df[main_df.sigma == sigma], f"offset sigma={sigma:g}", f"iid sigma={sigma:g}"),
        ])
        comps.to_csv(args.out_dir / "ablation_summary.csv", index=False)
        with pd.option_context("display.width", 200, "display.max_columns", None, "display.precision", 3):
            print("\n[ablation] constant offset vs. clean and vs. i.i.d. noise (paired on env seeds)")
            print(comps)


if __name__ == "__main__":
    main()