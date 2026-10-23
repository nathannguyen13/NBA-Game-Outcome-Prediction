"""
02_exploratory_analysis.py
==========================
Project : Predicting NBA Game Outcomes Using Recent Team Performance (DS 4002, Project 2)

PURPOSE
    Exploratory data analysis (EDA) that answers the questions posed in MI2 and
    produces the explanatory plots used in DATA/README.md:
      1. What does the distribution of points scored look like?
      2. Does the previous-five-game point differential differ before wins vs losses?
      3. How does each team's scoring move over the season (the time series that
         ARIMA will model)?
    Also prints a per-team summary table to the console.

INPUTS   DATA/all_games_clean.csv, DATA/model_dataset.csv
         (created by 01_clean_and_engineer_features.py)
OUTPUTS  OUTPUT/fig01_points_distribution.png
         OUTPUT/fig02_last5_point_diff_by_outcome.png
         OUTPUT/fig03_points_time_series_by_team.png

HOW TO RUN (from the repository root)
    python SCRIPTS/02_exploratory_analysis.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import plot_style as ps

ROOT = Path(__file__).resolve().parents[1]
# Folder layout used by this repository: all CSVs (raw and processed) sit directly
# in DATA/, and all figures and tables sit directly in OUTPUT/.
PROC_DIR = ROOT / "DATA"
FIG_DIR = ROOT / "OUTPUT"
FIG_DIR.mkdir(parents=True, exist_ok=True)

ps.apply()

games = pd.read_csv(PROC_DIR / "all_games_clean.csv", parse_dates=["game_date"])
model_df = pd.read_csv(PROC_DIR / "model_dataset.csv", parse_dates=["game_date"])
print(f"All games: {len(games)}   modeling rows: {len(model_df)}")

# ---------------------------------------------------------------------------
# Per-team season summary (printed only)
# ---------------------------------------------------------------------------
summary = (games.groupby("team")
           .agg(games=("win", "size"), wins=("win", "sum"),
                avg_points=("points", "mean"), avg_opp_points=("opponent_points", "mean"),
                avg_point_diff=("point_diff", "mean"), avg_rebounds=("rebounds", "mean"),
                avg_turnovers=("turnovers", "mean"),
                home_win_pct=("win", lambda s: s[games.loc[s.index, "home"] == 1].mean()),
                away_win_pct=("win", lambda s: s[games.loc[s.index, "home"] == 0].mean()))
           .reindex(ps.TEAM_ORDER))
summary["win_pct"] = summary["wins"] / summary["games"]
summary = summary.round(3)
print("\nPer-team summary:\n", summary.to_string())

# ---------------------------------------------------------------------------
# Figure 1: distribution of points scored (all 410 games)
# ---------------------------------------------------------------------------
mean_pts = games["points"].mean()
fig, ax = plt.subplots(figsize=(7, 4))
bins = np.arange(80, 155, 5)
ax.hist(games["points"], bins=bins, color=ps.TEAM_COLORS["BOS"], edgecolor=ps.SURFACE, linewidth=2)
ax.axvline(mean_pts, color=ps.INK, linestyle="--", linewidth=1.5)
ax.text(mean_pts + 1, ax.get_ylim()[1] * 0.92, f"mean = {mean_pts:.2f}", color=ps.INK, fontsize=9)
ax.set_title("Points scored per game, all five teams (n = 410)")
ax.set_xlabel("Points scored")
ax.set_ylabel("Number of games")
ax.grid(axis="x", visible=False)
fig.savefig(FIG_DIR / "fig01_points_distribution.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 2: previous-5 average point differential, before wins vs before losses
# ---------------------------------------------------------------------------
win_vals = model_df.loc[model_df["win"] == 1, "avg_point_diff_last5"]
loss_vals = model_df.loc[model_df["win"] == 0, "avg_point_diff_last5"]

fig, ax = plt.subplots(figsize=(7, 4.2))
rng = np.random.default_rng(4002)        # fixed seed so the jitter is reproducible
for i, (vals, color, label) in enumerate([(win_vals, ps.WIN_COLOR, "Win"),
                                          (loss_vals, ps.LOSS_COLOR, "Loss")]):
    x = i + rng.uniform(-0.18, 0.18, len(vals))
    ax.scatter(x, vals, s=14, color=color, alpha=0.45, edgecolor="none")
    ax.boxplot(vals, positions=[i], widths=0.5, showfliers=False,
               medianprops=dict(color=ps.INK, linewidth=2),
               boxprops=dict(color=ps.INK_2), whiskerprops=dict(color=ps.INK_2),
               capprops=dict(color=ps.INK_2))
    ax.text(i + 0.3, vals.mean(), f"mean {vals.mean():+.2f}\n(n = {len(vals)})",
            va="center", fontsize=9, color=ps.INK)
ax.axhline(0, color=ps.NEUTRAL, linewidth=1)
ax.set_xticks([0, 1], ["Next game: Win", "Next game: Loss"])
ax.set_xlim(-0.6, 1.9)
ax.set_ylabel("Avg point differential, previous 5 games")
ax.set_title("Recent point differential before wins vs. losses")
ax.grid(axis="x", visible=False)
fig.savefig(FIG_DIR / "fig02_last5_point_diff_by_outcome.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 3: points scored as a time series, one panel per team (small multiples)
# ---------------------------------------------------------------------------
split_date = pd.Timestamp("2026-03-15")
fig, axes = plt.subplots(5, 1, figsize=(9, 11), sharex=True, sharey=True)
for ax, team in zip(axes, ps.TEAM_ORDER):
    g = games[games["team"] == team]
    color = ps.TEAM_COLORS[team]
    ax.plot(g["game_date"], g["points"], color=color, alpha=0.35, linewidth=1.2,
            marker="o", markersize=3)
    ax.plot(g["game_date"], g["points"].rolling(5).mean(), color=color, linewidth=2.2)
    ax.axvspan(split_date, g["game_date"].max() + pd.Timedelta(days=2), color="#efeee9", zorder=0)
    ax.set_title(f"{team}   season mean {g['points'].mean():.1f}", fontsize=10)
    ax.set_ylabel("Points")
axes[0].text(split_date + pd.Timedelta(days=2), 150, "test period", fontsize=8, color=ps.INK_2)
axes[-1].set_xlabel("Game date (thin line = each game, thick line = 5-game rolling mean)")
fig.suptitle("Points scored over the 2025-26 season", x=0.01, ha="left",
             fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig03_points_time_series_by_team.png")
plt.close(fig)

print(f"\nMean points per game: {mean_pts:.2f}")
print(f"Mean prev-5 point diff before wins: {win_vals.mean():+.2f}  before losses: {loss_vals.mean():+.2f}")
print("Figures saved to OUTPUT/")
