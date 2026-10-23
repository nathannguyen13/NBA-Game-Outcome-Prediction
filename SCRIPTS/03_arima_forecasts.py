"""
03_arima_forecasts.py
=====================
Project : Predicting NBA Game Outcomes Using Recent Team Performance (DS 4002, Project 2)

PURPOSE
    Time-series component of the project. For every team we treat its game-by-game
    points scored as a time series and use ARIMA [statsmodels] to forecast the NEXT
    game's points using only the games already played. These one-step-ahead
    forecasts become an extra predictor in the logistic regression (script 04).

STEPS
    1. Stationarity check: Augmented Dickey-Fuller (ADF) test on each team's
       training-period points series (games before 2026-03-15).
    2. ACF / PACF plots of the training-period series.
    3. Order selection: the ADF tests reject a unit root for every team, so the
       differencing order is fixed at d = 0. We fit every ARIMA(p,0,q) with
       p,q in {0,1,2} to each team's TRAINING series and choose the single (p,q)
       with the lowest total AIC across the five teams. Only training data is used,
       so the test period never influences the model choice.
    4. Walk-forward (expanding-window) forecasting: for game k of a team, refit the
       chosen ARIMA on that team's games 1..k-1 and forecast game k. Nothing from
       game k or later is ever used to forecast game k.
    5. Forecast check: compare ARIMA's mean absolute error (MAE) with the simple
       previous-5-game average.

INPUTS   DATA/processed/all_games_clean.csv   (from script 01)
OUTPUTS  DATA/processed/arima_forecasts.csv   (one-step-ahead forecast for every game)
         OUTPUT/tables/arima_adf_tests.csv
         OUTPUT/tables/arima_order_selection.csv
         OUTPUT/tables/arima_forecast_accuracy.csv
         OUTPUT/figures/fig04_acf_pacf_points.png
         OUTPUT/figures/fig05_arima_forecast_vs_actual.png

HOW TO RUN (from the repository root)
    python SCRIPTS/03_arima_forecasts.py        (about 30 seconds)

NOTES
    * Forecasts for each team's first five games are not used anywhere (those games
      are dropped from the modeling table in script 01). If a fit ever fails, the
      forecast falls back to the mean of the games played so far and is flagged.
"""

import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

import plot_style as ps

# ARIMA on short series emits many harmless convergence / version warnings.
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
PROC_DIR = ROOT / "DATA" / "processed"
FIG_DIR = ROOT / "OUTPUT" / "figures"
TAB_DIR = ROOT / "OUTPUT" / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)
TAB_DIR.mkdir(parents=True, exist_ok=True)
SPLIT_DATE = pd.Timestamp("2026-03-15")
MIN_HISTORY = 5                            # need at least 5 past games to forecast

ps.apply()
games = pd.read_csv(PROC_DIR / "all_games_clean.csv", parse_dates=["game_date"])
games = games.sort_values(["team", "game_date"]).reset_index(drop=True)
train_games = games[games["game_date"] < SPLIT_DATE]


def train_series(team: str) -> np.ndarray:
    """A team's points scored in the training period, in date order."""
    return train_games.loc[train_games["team"] == team, "points"].to_numpy(dtype=float)


# ---------------------------------------------------------------------------
# 1. Stationarity: ADF test per team, training period only
# ---------------------------------------------------------------------------
adf_rows = []
for team in ps.TEAM_ORDER:
    y = train_series(team)
    stat, pval, *_ = adfuller(y, autolag="AIC")
    adf_rows.append({"team": team, "n_games": len(y), "adf_statistic": round(stat, 3),
                     "p_value": round(pval, 4), "stationary_at_5pct": pval < 0.05})
adf = pd.DataFrame(adf_rows)
adf.to_csv(TAB_DIR / "arima_adf_tests.csv", index=False)
print("ADF stationarity tests on points (training period):\n", adf.to_string(index=False))

# ---------------------------------------------------------------------------
# 2. ACF / PACF of points, training period
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(5, 2, figsize=(10, 12))
for row, team in enumerate(ps.TEAM_ORDER):
    y = train_series(team)
    color = ps.TEAM_COLORS[team]
    plot_acf(y, lags=12, ax=axes[row, 0], color=color, vlines_kwargs={"colors": color},
             title=f"{team}: ACF of points")
    plot_pacf(y, lags=12, ax=axes[row, 1], color=color, vlines_kwargs={"colors": color},
              title=f"{team}: PACF of points", method="ywm")
    for ax in axes[row]:
        ax.set_ylim(-0.6, 1.05)
        ax.title.set_fontsize(10)
for ax in axes[-1]:
    ax.set_xlabel("Lag (games)")
fig.suptitle("Autocorrelation of points scored, training period (shaded band = 95% significance)",
             x=0.01, ha="left", fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig04_acf_pacf_points.png")
plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Order selection by total AIC over the five teams' training series
# ---------------------------------------------------------------------------
def fit_arima(y: np.ndarray, order: tuple):
    """Fit ARIMA with a constant (d = 0, so the series fluctuates around a mean)."""
    return ARIMA(y, order=order, trend="c").fit()


assert adf["stationary_at_5pct"].all(), "A series is non-stationary: revisit the choice d = 0"
sel_rows = []
for p, q in itertools.product([0, 1, 2], [0, 1, 2]):
    order = (p, 0, q)
    total_aic = sum(fit_arima(train_series(t), order).aic for t in ps.TEAM_ORDER)
    sel_rows.append({"order": str(order), "p": p, "d": 0, "q": q, "total_aic": round(total_aic, 2)})
sel = pd.DataFrame(sel_rows).sort_values("total_aic").reset_index(drop=True)
best = sel.iloc[0]
ORDER = (int(best["p"]), 0, int(best["q"]))
sel["chosen"] = sel.index == 0
sel.to_csv(TAB_DIR / "arima_order_selection.csv", index=False)
print(f"\nChosen ARIMA order for points: {ORDER}  (total AIC {best['total_aic']})")


# ---------------------------------------------------------------------------
# 4. Walk-forward one-step-ahead forecasts (expanding window)
# ---------------------------------------------------------------------------
def walk_forward(y: np.ndarray, order: tuple):
    """Return (forecasts, fallback_flags) where forecast[k] uses only y[:k]."""
    preds = np.full(len(y), np.nan)
    fallback = np.zeros(len(y), dtype=bool)
    for k in range(MIN_HISTORY, len(y)):
        history = y[:k]
        try:
            preds[k] = fit_arima(history, order).forecast(1)[0]
            if not np.isfinite(preds[k]):
                raise ValueError("non-finite forecast")
        except Exception:
            preds[k] = history.mean()
            fallback[k] = True
    return preds, fallback


out = games[["team", "game_date", "points"]].copy()
out["arima_points_forecast"] = np.nan
out["arima_fallback"] = False
for team in ps.TEAM_ORDER:
    idx = out.index[out["team"] == team]
    preds, fb = walk_forward(out.loc[idx, "points"].to_numpy(dtype=float), ORDER)
    out.loc[idx, "arima_points_forecast"] = preds
    out.loc[idx, "arima_fallback"] = fb
    print(f"  {team}: {np.isfinite(preds).sum()} forecasts, {fb.sum()} fallbacks")

# Simple comparison forecast: average of the previous 5 games
out["prev5_points_avg"] = out.groupby("team")["points"].transform(
    lambda s: s.shift(1).rolling(5, min_periods=5).mean())
out["arima_order"] = str(ORDER)
out.to_csv(PROC_DIR / "arima_forecasts.csv", index=False)


# ---------------------------------------------------------------------------
# 5. Forecast accuracy (MAE) on train and test periods
# ---------------------------------------------------------------------------
valid = out.dropna(subset=["prev5_points_avg"]).copy()
valid["period"] = np.where(valid["game_date"] >= SPLIT_DATE, "test", "train")
acc_rows = []
for period, g in valid.groupby("period"):
    for name, col in [(f"ARIMA{ORDER}", "arima_points_forecast"),
                      ("Previous-5 average", "prev5_points_avg")]:
        acc_rows.append({"period": period, "forecaster": name, "n": len(g),
                         "MAE": round((g["points"] - g[col]).abs().mean(), 2)})
acc = pd.DataFrame(acc_rows)
acc.to_csv(TAB_DIR / "arima_forecast_accuracy.csv", index=False)
print("\nOne-step-ahead forecast accuracy (points):\n", acc.to_string(index=False))

# ---------------------------------------------------------------------------
# Figure 5: ARIMA forecast vs actual points, test period, per team
# ---------------------------------------------------------------------------
test = out[out["game_date"] >= SPLIT_DATE]
fig, axes = plt.subplots(5, 1, figsize=(9, 11), sharex=True, sharey=True)
for ax, team in zip(axes, ps.TEAM_ORDER):
    g = test[test["team"] == team]
    ax.plot(g["game_date"], g["points"], color=ps.TEAM_COLORS[team], marker="o", markersize=5,
            linewidth=1.2, alpha=0.55, label="Actual points")
    ax.plot(g["game_date"], g["arima_points_forecast"], color=ps.INK, linewidth=2,
            linestyle="--", label=f"ARIMA{ORDER} forecast")
    mae = (g["points"] - g["arima_points_forecast"]).abs().mean()
    ax.set_title(f"{team}   test MAE {mae:.1f} pts", fontsize=10)
    ax.set_ylabel("Points")
axes[0].legend(loc="upper left", ncol=2)
axes[-1].set_xlabel("Game date (test period: 2026-03-15 to 2026-04-12)")
fig.suptitle("One-step-ahead ARIMA forecasts of points vs. actual, test period",
             x=0.01, ha="left", fontsize=13, fontweight="bold")
fig.tight_layout()
fig.savefig(FIG_DIR / "fig05_arima_forecast_vs_actual.png")
plt.close(fig)
print("Saved DATA/processed/arima_forecasts.csv and ARIMA tables/figures.")
