"""
03_arima_forecasts.py
=====================
Project : Predicting NBA Game Outcomes Using Recent Team Performance (DS 4002, Project 2)

PURPOSE
    Time-series component of the project. For every team we treat its game-by-game
    points scored as a time series and use ARIMA [statsmodels] to forecast the NEXT
    game's points using only the games already played. These one-step-ahead
    forecasts become an extra predictor in the logistic regression (script 04).

STEPS (each step saves its results to OUTPUT/ so they can be checked later)
    1. Stationarity: Augmented Dickey-Fuller (ADF) test on each team's
       training-period points series (games before 2026-03-15).
          -> OUTPUT/arima_adf_tests.csv
    2. Identification: ACF and PACF of each team's training series, used to justify
       the candidate range p, q in {0, 1, 2}.
          -> OUTPUT/fig04_acf_pacf_points.png, OUTPUT/arima_acf_pacf_values.csv
    3. Order selection: fit every ARIMA(p,0,q), p,q in {0,1,2}, to each team's
       TRAINING series. Every fit's convergence is checked; an order is eligible only
       if it converged for all five teams. The eligible order with the lowest total
       AIC is chosen. d = 0 because step 1 finds every series stationary.
          -> OUTPUT/arima_order_selection.csv
    4. Residual checks: refit the chosen order on each team's training series and run
       Ljung-Box tests on the residuals (lags 5 and 10). Large p-values mean no
       autocorrelation is left over, i.e. the model captures the series' structure.
          -> OUTPUT/arima_ljungbox.csv
    5. Walk-forward forecasting: game k of a team is forecast from games 1..k-1 only.
       * A forecast is made by ARIMA only when the team has at least MIN_HISTORY (20)
         earlier games. With fewer games a model with up to 6 estimated parameters
         (ARIMA(2,0,2) + constant + variance) is not reliable, so the forecast is the
         running mean of the games so far, labelled "warmup_running_mean".
       * Every ARIMA fit is checked for convergence warnings and for the optimizer's
         own "converged" flag. A fit that did not converge, raised an error, or gave a
         non-finite forecast is replaced by the running mean and labelled as a
         fallback, so no bad forecast can slip through unnoticed.
          -> DATA/arima_forecasts.csv (with forecast_source and n_history columns)
          -> OUTPUT/arima_fit_log.csv (how many forecasts came from each source)
    6. Comparison with simple averages: ARIMA forecasts are compared with the running
       (season-to-date) mean and the previous-5-game mean, using MAE and RMSE on the
       games where ARIMA made the forecast. The largest absolute gap between the
       ARIMA forecast and the running mean is also reported: if the chosen order is
       ARIMA(0,0,0), ARIMA is just a running mean and this gap is ~0.
          -> OUTPUT/arima_forecast_accuracy.csv, OUTPUT/fig05_arima_forecast_vs_actual.png

INPUTS   DATA/all_games_clean.csv   (from script 01)

HOW TO RUN (from the repository root)
    python SCRIPTS/03_arima_forecasts.py        (about 30-60 seconds)
"""

import itertools
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tools.sm_exceptions import ConvergenceWarning
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, adfuller, pacf

import plot_style as ps

# Only silence the version-change notice adfuller prints; every other warning,
# including ConvergenceWarning, is captured and checked inside fit_arima() below.
warnings.filterwarnings("ignore", category=FutureWarning)

# Folder layout: all CSVs (raw and processed) sit directly in DATA/,
# and all figures and tables sit directly in OUTPUT/.
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "DATA"
OUT_DIR = ROOT / "OUTPUT"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SPLIT_DATE = pd.Timestamp("2026-03-15")
MIN_HISTORY = 20            # fewest earlier games before ARIMA is allowed to forecast
PQ_RANGE = [0, 1, 2]        # candidate AR and MA orders
ACF_LAGS = 10               # lags shown in the ACF/PACF plots and table
LB_LAGS = [5, 10]           # Ljung-Box lags

ps.apply()
games = pd.read_csv(DATA_DIR / "all_games_clean.csv", parse_dates=["game_date"])
games = games.sort_values(["team", "game_date"]).reset_index(drop=True)
train_games = games[games["game_date"] < SPLIT_DATE]


def train_series(team: str) -> np.ndarray:
    """A team's points scored in the training period, in date order."""
    return train_games.loc[train_games["team"] == team, "points"].to_numpy(dtype=float)


def fit_arima(y: np.ndarray, order: tuple):
    """Fit ARIMA(p,0,q) with a constant and report whether the fit can be trusted.

    Returns (result, ok, reason). ok is False when the optimizer raised a
    ConvergenceWarning, reported converged=False, produced non-finite parameters,
    or crashed. Warnings are recorded here instead of being switched off globally.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            res = ARIMA(y, order=order, trend="c").fit()
        except Exception as exc:                       # noqa: BLE001
            return None, False, f"error: {type(exc).__name__}"
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        return res, False, "convergence_warning"
    if not res.mle_retvals.get("converged", True):
        return res, False, "optimizer_not_converged"
    if not np.all(np.isfinite(res.params)):
        return res, False, "non_finite_parameters"
    return res, True, "ok"


# ---------------------------------------------------------------------------
# 1. Stationarity: ADF test per team, training period only
# ---------------------------------------------------------------------------
adf_rows = []
for team in ps.TEAM_ORDER:
    y = train_series(team)
    stat, pval, used_lag, nobs, *_ = adfuller(y, autolag="AIC")
    adf_rows.append({"team": team, "n_games": len(y), "adf_statistic": round(stat, 3),
                     "p_value": round(pval, 4), "lags_used": used_lag,
                     "stationary_at_5pct": pval < 0.05})
adf = pd.DataFrame(adf_rows)
adf.to_csv(OUT_DIR / "arima_adf_tests.csv", index=False)
print("1. ADF stationarity tests (training period):\n", adf.to_string(index=False))
assert adf["stationary_at_5pct"].all(), "A series is non-stationary: revisit the choice d = 0"

# ---------------------------------------------------------------------------
# 2. ACF / PACF: figure + table of values, to justify the p, q range
# ---------------------------------------------------------------------------
acf_rows = []
fig, axes = plt.subplots(5, 2, figsize=(12, 14))
for row, team in enumerate(ps.TEAM_ORDER):
    y = train_series(team)
    bound = 1.96 / np.sqrt(len(y))               # approximate 95% significance bound
    acf_vals = acf(y, nlags=ACF_LAGS)
    pacf_vals = pacf(y, nlags=ACF_LAGS, method="ywm")
    for lag in range(1, ACF_LAGS + 1):
        acf_rows.append({"team": team, "lag": lag, "acf": round(acf_vals[lag], 3),
                         "pacf": round(pacf_vals[lag], 3), "bound_95": round(bound, 3),
                         "acf_significant": abs(acf_vals[lag]) > bound,
                         "pacf_significant": abs(pacf_vals[lag]) > bound})
    color = ps.TEAM_COLORS[team]
    plot_acf(y, lags=ACF_LAGS, ax=axes[row, 0], color=color,
             vlines_kwargs={"colors": color}, title=f"{team}: ACF of points")
    plot_pacf(y, lags=ACF_LAGS, ax=axes[row, 1], color=color,
              vlines_kwargs={"colors": color}, title=f"{team}: PACF of points", method="ywm")
    for ax in axes[row]:
        ax.set_ylim(-0.6, 1.05)
        ax.title.set_fontsize(13)
        ax.tick_params(labelsize=11)
for ax in axes[-1]:
    ax.set_xlabel("Lag (games)", fontsize=12)
fig.suptitle("ACF and PACF of points scored, training period (shaded band = 95% significance)",
             x=0.01, ha="left", fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig04_acf_pacf_points.png")
plt.close(fig)
acf_tab = pd.DataFrame(acf_rows)
acf_tab.to_csv(OUT_DIR / "arima_acf_pacf_values.csv", index=False)
sig = acf_tab.groupby("team")[["acf_significant", "pacf_significant"]].sum().reindex(ps.TEAM_ORDER)
print(f"\n2. Significant ACF/PACF lags out of {ACF_LAGS} (training period):\n", sig.to_string())

# ---------------------------------------------------------------------------
# 3. Order selection by total AIC, convergence checked for every fit
# ---------------------------------------------------------------------------
sel_rows = []
for p, q in itertools.product(PQ_RANGE, PQ_RANGE):
    order = (p, 0, q)
    row = {"order": str(order), "p": p, "d": 0, "q": q}
    all_ok, total_aic = True, 0.0
    for team in ps.TEAM_ORDER:
        res, ok, reason = fit_arima(train_series(team), order)
        row[f"aic_{team}"] = round(res.aic, 2) if res is not None else np.nan
        row[f"fit_{team}"] = reason
        all_ok &= ok
        total_aic += res.aic if res is not None else np.inf
    row["all_teams_converged"] = all_ok
    row["total_aic"] = round(total_aic, 2) if all_ok else np.nan
    sel_rows.append(row)
sel = pd.DataFrame(sel_rows).sort_values("total_aic", na_position="last").reset_index(drop=True)
best = sel.iloc[0]
ORDER = (int(best["p"]), 0, int(best["q"]))
sel["chosen"] = sel.index == 0
sel.to_csv(OUT_DIR / "arima_order_selection.csv", index=False)
print(f"\n3. Orders converged for all teams: {int(sel['all_teams_converged'].sum())} of {len(sel)}")
print(sel[["order", "all_teams_converged", "total_aic"]].to_string(index=False))
print(f"   Chosen ARIMA order: {ORDER}  (total AIC {best['total_aic']})")

# ---------------------------------------------------------------------------
# 4. Residual checks: Ljung-Box on the chosen model's training residuals
# ---------------------------------------------------------------------------
lb_rows = []
for team in ps.TEAM_ORDER:
    res, ok, reason = fit_arima(train_series(team), ORDER)
    lb = acorr_ljungbox(res.resid, lags=LB_LAGS, model_df=ORDER[0] + ORDER[2], return_df=True)
    for lag, r in lb.iterrows():
        lb_rows.append({"team": team, "order": str(ORDER), "fit_status": reason, "lag": lag,
                        "lb_stat": round(r["lb_stat"], 3), "p_value": round(r["lb_pvalue"], 4),
                        "residuals_look_white_noise": r["lb_pvalue"] > 0.05})
ljung = pd.DataFrame(lb_rows)
ljung.to_csv(OUT_DIR / "arima_ljungbox.csv", index=False)
print("\n4. Ljung-Box tests on residuals of the chosen model:\n", ljung.to_string(index=False))


# ---------------------------------------------------------------------------
# 5. Walk-forward one-step-ahead forecasts with warm-up and fit checks
# ---------------------------------------------------------------------------
def walk_forward(y: np.ndarray, order: tuple):
    """forecast[k] uses only y[:k]; returns forecasts and where each one came from."""
    preds = np.full(len(y), np.nan)
    source = np.array(["none"] * len(y), dtype=object)
    for k in range(1, len(y)):
        history = y[:k]
        running_mean = history.mean()
        if k < MIN_HISTORY:                          # too little data for ARIMA
            preds[k], source[k] = running_mean, "warmup_running_mean"
            continue
        res, ok, reason = fit_arima(history, order)
        fc = res.forecast(1)[0] if res is not None else np.nan
        if ok and np.isfinite(fc):
            preds[k], source[k] = fc, "arima"
        else:                                        # flagged, never silently used
            preds[k], source[k] = running_mean, f"fallback_{reason}"
    return preds, source


out = games[["team", "game_date", "points"]].copy()
out["n_history"] = out.groupby("team").cumcount()          # earlier games available
out["arima_points_forecast"] = np.nan
out["forecast_source"] = "none"
for team in ps.TEAM_ORDER:
    idx = out.index[out["team"] == team]
    preds, source = walk_forward(out.loc[idx, "points"].to_numpy(dtype=float), ORDER)
    out.loc[idx, "arima_points_forecast"] = preds
    out.loc[idx, "forecast_source"] = source

# Simple comparison forecasts, built from earlier games only
out["running_mean_forecast"] = out.groupby("team")["points"].transform(
    lambda s: s.shift(1).expanding(min_periods=1).mean())
out["prev5_points_avg"] = out.groupby("team")["points"].transform(
    lambda s: s.shift(1).rolling(5, min_periods=5).mean())
out["arima_order"] = str(ORDER)
out.to_csv(DATA_DIR / "arima_forecasts.csv", index=False)

fit_log = (out.assign(period=np.where(out["game_date"] >= SPLIT_DATE, "test", "train"))
           .groupby(["team", "period", "forecast_source"]).size()
           .rename("n_games").reset_index())
fit_log.to_csv(OUT_DIR / "arima_fit_log.csv", index=False)
print(f"\n5. Forecast sources (warm-up = fewer than {MIN_HISTORY} earlier games):")
print(out["forecast_source"].value_counts().to_string())

# ---------------------------------------------------------------------------
# 6. Compare ARIMA with simple averages on games ARIMA actually forecast
# ---------------------------------------------------------------------------
cmp = out[out["forecast_source"] == "arima"].copy()
cmp["period"] = np.where(cmp["game_date"] >= SPLIT_DATE, "test", "train")
acc_rows = []
for period in ["train", "test"]:
    g = cmp[cmp["period"] == period]
    for name, col in [(f"ARIMA{ORDER}", "arima_points_forecast"),
                      ("Running (season-to-date) mean", "running_mean_forecast"),
                      ("Previous-5-game mean", "prev5_points_avg")]:
        err = g["points"] - g[col]
        acc_rows.append({"period": period, "forecaster": name, "n_games": len(g),
                         "MAE": round(err.abs().mean(), 3),
                         "RMSE": round(np.sqrt((err ** 2).mean()), 3),
                         "max_abs_gap_vs_running_mean":
                             round((g[col] - g["running_mean_forecast"]).abs().max(), 4)})
acc = pd.DataFrame(acc_rows)
acc.to_csv(OUT_DIR / "arima_forecast_accuracy.csv", index=False)
print("\n6. One-step-ahead forecast accuracy (games forecast by ARIMA):\n", acc.to_string(index=False))

# Figure 5: ARIMA forecast vs running mean vs actual, test period
test = out[out["game_date"] >= SPLIT_DATE]
fig, axes = plt.subplots(5, 1, figsize=(11, 13), sharex=True, sharey=True)
for ax, team in zip(axes, ps.TEAM_ORDER):
    g = test[test["team"] == team]
    ax.plot(g["game_date"], g["points"], color=ps.TEAM_COLORS[team], marker="o", markersize=6,
            linewidth=1.4, alpha=0.6, label="Actual points")
    ax.plot(g["game_date"], g["arima_points_forecast"], color=ps.INK, linewidth=2.4,
            linestyle="--", label=f"ARIMA{ORDER} forecast")
    ax.plot(g["game_date"], g["running_mean_forecast"], color=ps.NEUTRAL, linewidth=2,
            linestyle=":", label="Running mean")
    mae = (g["points"] - g["arima_points_forecast"]).abs().mean()
    ax.set_title(f"{team}   test MAE {mae:.1f} pts", fontsize=13)
    ax.set_ylabel("Points", fontsize=12)
    ax.tick_params(labelsize=11)
axes[0].legend(loc="upper left", ncol=3, fontsize=11)
axes[-1].set_xlabel("Game date (test period: 2026-03-15 to 2026-04-12)", fontsize=12)
fig.suptitle("One-step-ahead ARIMA forecasts vs. running mean vs. actual points, test period",
             x=0.01, ha="left", fontsize=15, fontweight="bold")
fig.tight_layout()
fig.savefig(OUT_DIR / "fig05_arima_forecast_vs_actual.png")
plt.close(fig)
print("\nSaved DATA/arima_forecasts.csv, ARIMA tables and figures in OUTPUT/.")
