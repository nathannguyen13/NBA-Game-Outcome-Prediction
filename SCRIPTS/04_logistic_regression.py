"""
04_logistic_regression.py
=========================
Project : Predicting NBA Game Outcomes Using Recent Team Performance (DS 4002, Project 2)

PURPOSE
    Answer the research question: can a team's previous five games predict whether it
    wins its next game? Fits logistic regression on the chronological training set
    (games through 2026-03-14) and predicts win/loss for the held-out test set
    (2026-03-15 to 2026-04-12).

MODELS COMPARED
    * Home-team baseline        : predict a win whenever the team is at home (MI2 benchmark)
    * Always-predict-win        : predict the most common training outcome for every game
                                  (reference point: the accuracy you get for free)
    * LR: rolling-5 + home      : logistic regression on the previous-5-game averages of
                                  points, point differential, rebounds, turnovers + home flag
    * LR: + ARIMA points        : the same plus the ARIMA next-game points forecast (script 03)

METRIC
    Test-set accuracy (MI2 goal: >= 65% and better than the home-team baseline), with a
    95% Wilson confidence interval because the test set is small (76 games).

INPUTS   DATA/processed/model_dataset.csv     (script 01)
         DATA/processed/arima_forecasts.csv   (script 03)
OUTPUTS  DATA/processed/final_analysis_dataset.csv
         OUTPUT/tables/model_comparison.csv
         OUTPUT/tables/test_predictions.csv
         OUTPUT/figures/fig06_model_accuracy_comparison.png
         OUTPUT/figures/fig07_confusion_matrices.png

HOW TO RUN (from the repository root)
    python SCRIPTS/04_logistic_regression.py
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.proportion import proportion_confint

import plot_style as ps

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "DATA" / "processed"
FIG_DIR = ROOT / "OUTPUT" / "figures"
TAB_DIR = ROOT / "OUTPUT" / "tables"
ps.apply()

TARGET_ACCURACY = 0.65    # quantitative goal from MI2

# ---------------------------------------------------------------------------
# Load and merge: rolling features (script 01) + ARIMA forecast (script 03)
# ---------------------------------------------------------------------------
model_df = pd.read_csv(PROC_DIR / "model_dataset.csv", parse_dates=["game_date"])
arima = pd.read_csv(PROC_DIR / "arima_forecasts.csv", parse_dates=["game_date"])
df = model_df.merge(arima[["team", "game_date", "arima_points_forecast"]],
                    on=["team", "game_date"], how="left", validate="one_to_one")
assert df["arima_points_forecast"].notna().all(), "missing ARIMA forecast for a modeling row"
df = df.sort_values(["game_date", "team"]).reset_index(drop=True)
df.to_csv(PROC_DIR / "final_analysis_dataset.csv", index=False)

train = df[df["split"] == "train"]
test = df[df["split"] == "test"]
print(f"Train: {len(train)} games ({train['win'].mean():.1%} wins)   "
      f"Test: {len(test)} games ({test['win'].mean():.1%} wins)")

BASE = ["avg_points_last5", "avg_point_diff_last5", "avg_rebounds_last5",
        "avg_turnovers_last5", "home"]
FEATURE_SETS = {
    "LR: rolling-5 + home": BASE,
    "LR: + ARIMA points": BASE + ["arima_points_forecast"],
}

# ---------------------------------------------------------------------------
# Predictions on the test set
# ---------------------------------------------------------------------------
preds = test[["team", "game_date", "opponent", "home", "win"]].copy()

# Baselines (no fitting needed)
majority = int(train["win"].mean() >= 0.5)            # 1 = win is the common outcome
preds["Home-team baseline"] = test["home"].astype(int)
preds["Always predict win"] = majority
train_acc = {"Home-team baseline": accuracy_score(train["win"], train["home"]),
             "Always predict win": accuracy_score(train["win"], np.full(len(train), majority))}

# Logistic regression: standardize features, then fit (scikit-learn defaults)
for name, feats in FEATURE_SETS.items():
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(train[feats], train["win"])
    preds[f"{name} prob"] = model.predict_proba(test[feats])[:, 1]   # P(win)
    preds[name] = (preds[f"{name} prob"] >= 0.5).astype(int)          # predicted win/loss
    train_acc[name] = accuracy_score(train["win"], model.predict(train[feats]))
    print(f"  {name}: predicts a win for {preds[name].mean():.0%} of test games")
preds.to_csv(TAB_DIR / "test_predictions.csv", index=False)

# ---------------------------------------------------------------------------
# Accuracy table with 95% Wilson confidence intervals
# ---------------------------------------------------------------------------
MODELS = ["Home-team baseline", "Always predict win"] + list(FEATURE_SETS)
rows = []
for m in MODELS:
    correct = int((preds[m] == preds["win"]).sum())
    lo, hi = proportion_confint(correct, len(test), alpha=0.05, method="wilson")
    rows.append({"model": m, "train_accuracy": train_acc[m],
                 "test_correct": correct, "test_games": len(test),
                 "test_accuracy": correct / len(test), "ci95_low": lo, "ci95_high": hi,
                 "share_predicted_win": preds[m].mean()})
results = pd.DataFrame(rows)
home_acc = results.loc[results["model"] == "Home-team baseline", "test_accuracy"].item()
results["meets_65pct_goal"] = results["test_accuracy"] >= TARGET_ACCURACY
results["beats_home_baseline"] = results["test_accuracy"] > home_acc
results.round(4).to_csv(TAB_DIR / "model_comparison.csv", index=False)
print("\nModel comparison:\n", results.round(3).to_string(index=False))

# ---------------------------------------------------------------------------
# Figure 6: test accuracy with 95% CI (dot-and-interval)
# ---------------------------------------------------------------------------
plot_df = results.iloc[::-1].reset_index(drop=True)
fig, ax = plt.subplots(figsize=(8.5, 3.6))
for i, r in plot_df.iterrows():
    color = ps.NEUTRAL if r["model"] in ("Home-team baseline", "Always predict win") else ps.WIN_COLOR
    ax.errorbar(r["test_accuracy"], i,
                xerr=[[r["test_accuracy"] - r["ci95_low"]], [r["ci95_high"] - r["test_accuracy"]]],
                fmt="o", color=color, markersize=9, capsize=4, elinewidth=2,
                markeredgecolor=ps.SURFACE)
    ax.text(r["ci95_high"] + 0.01, i, f"{r['test_accuracy']:.1%}", va="center",
            fontsize=9, color=ps.INK)
ax.axvline(TARGET_ACCURACY, color=ps.INK, linestyle="--", linewidth=1.2)
ax.text(TARGET_ACCURACY - 0.005, -0.75, "65% goal", fontsize=8, color=ps.INK, ha="right")
ax.set_ylim(-1, len(plot_df) - 0.5)
ax.set_yticks(range(len(plot_df)), plot_df["model"])
ax.set_xlim(0.4, 0.95)
ax.set_xlabel(f"Test-set accuracy (n = {len(test)} games), whiskers = 95% CI  "
              "(gray = baseline, blue = logistic regression)")
ax.set_title("Model accuracy on held-out games, Mar 15 - Apr 12, 2026")
ax.grid(axis="y", visible=False)
fig.savefig(FIG_DIR / "fig06_model_accuracy_comparison.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Figure 7: confusion matrices (rows = actual, columns = predicted)
# ---------------------------------------------------------------------------
cm_models = ["Home-team baseline"] + list(FEATURE_SETS)
fig, axes = plt.subplots(1, len(cm_models), figsize=(10, 3.6))
for ax, m in zip(axes, cm_models):
    cm = confusion_matrix(preds["win"], preds[m], labels=[0, 1])
    ax.imshow(cm, cmap="Blues", vmin=0, vmax=len(test))
    for (r, c), v in np.ndenumerate(cm):
        ax.text(c, r, str(v), ha="center", va="center", fontsize=13,
                color="white" if v > len(test) * 0.45 else ps.INK)
    ax.set_xticks([0, 1], ["Pred L", "Pred W"])
    ax.set_yticks([0, 1], ["Actual L", "Actual W"])
    ax.grid(False)
    ax.set_title(f"{m}\naccuracy {(preds[m] == preds['win']).mean():.1%}", fontsize=9)
fig.suptitle("Confusion matrices, test set", x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig07_confusion_matrices.png")
plt.close(fig)

print("\nSaved model tables and figures to OUTPUT/.")
