"""
01_clean_and_engineer_features.py
=================================
Project : Predicting NBA Game Outcomes Using Recent Team Performance (DS 4002, Project 2)
Group   : The Good Guys -- Nathan Nguyen (lead), Keegan Jewell, Brian Ryu

PURPOSE
    Turn the five raw BoxScore Lab game-log CSVs (BOS, TOR, LAL, SAS, OKC; 2025-26
    regular season) into one tidy modeling table with leakage-free five-game
    rolling averages.

INPUTS   DATA/<TEAM>_game_logs_2025-26.csv       (5 raw files, 82 games each)
OUTPUTS  DATA/all_games_clean.csv                (all 410 games, with rolling features)
         DATA/model_dataset.csv                (385 games with full 5-game history,
                                                  plus the chronological train/test label)

HOW TO RUN (from the repository root)
    python SCRIPTS/01_clean_and_engineer_features.py

NOTES
    * Rolling averages use shift(1) BEFORE rolling(5), so the game being predicted
      never contributes to its own features (no data leakage).
    * Each team's first five games have no complete 5-game history and are dropped
      from the modeling table (410 -> 385 rows).
    * Train/test split is chronological: games on or before 2026-03-14 are TRAIN,
      games on or after 2026-03-15 are TEST (about 80/20). Because the split is by
      date, both sides of a game between two of our five teams always land in the
      same split.
"""

from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths are built relative to this file so the script works from any folder.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
# Folder layout used by this repository: all CSVs (raw and processed) sit directly
# in DATA/, and all figures and tables sit directly in OUTPUT/.
RAW_DIR = ROOT / "DATA"     # the five raw team CSVs
PROC_DIR = ROOT / "DATA"    # processed CSVs are written next to them

TEAMS = ["BOS", "TOR", "LAL", "SAS", "OKC"]
WINDOW = 5                       # number of previous games in each rolling average
SPLIT_DATE = pd.Timestamp("2026-03-15")   # first date that belongs to the test set
ROLL_STATS = ["points", "point_diff", "rebounds", "turnovers"]


def load_raw() -> pd.DataFrame:
    """Read and stack the five raw team CSVs into one DataFrame."""
    frames = []
    for team in TEAMS:
        path = RAW_DIR / f"{team}_game_logs_2025-26.csv"
        df = pd.read_csv(path)
        frames.append(df)
        print(f"  loaded {path.name}: {len(df)} rows")
    return pd.concat(frames, ignore_index=True)


def validate(df: pd.DataFrame) -> pd.DataFrame:
    """Basic integrity checks. Anything that fails is reported and dropped
    (we never fill values with information from later games)."""
    df = df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"])

    # 'home' arrives as the strings "True"/"False"; convert to a 0/1 integer.
    df["home"] = df["home"].astype(str).str.strip().str.lower().map({"true": 1, "false": 0})

    # Missing values check
    n_missing = int(df.isna().sum().sum())
    print(f"  missing cells: {n_missing}")
    if n_missing:
        print(df[df.isna().any(axis=1)])
        df = df.dropna()

    # Duplicate team-date rows check
    dups = df.duplicated(subset=["team", "game_date"]).sum()
    print(f"  duplicate team/date rows: {dups}")
    df = df.drop_duplicates(subset=["team", "game_date"])

    # Internal consistency: point_diff should equal points - opponent_points,
    # and result should agree with the sign of point_diff.
    bad_diff = (df["points"] - df["opponent_points"] != df["point_diff"]).sum()
    bad_result = ((df["point_diff"] > 0) != (df["result"] == "W")).sum()
    print(f"  rows with inconsistent point_diff: {bad_diff}")
    print(f"  rows with result not matching point_diff: {bad_result}")

    # Home flag should agree with the matchup text ("vs." = home, "@" = away)
    home_from_text = df["matchup"].str.contains("vs.", regex=False).astype(int)
    print(f"  rows where home flag disagrees with matchup text: {(home_from_text != df['home']).sum()}")
    return df


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add target and five-game rolling averages computed from PRIOR games only."""
    df = df.sort_values(["team", "game_date"]).reset_index(drop=True)
    df["win"] = (df["result"] == "W").astype(int)            # target: 1 = win, 0 = loss
    df["game_number"] = df.groupby("team").cumcount() + 1    # 1..82 within each team

    for stat in ROLL_STATS:
        df[f"avg_{stat}_last5"] = (
            df.groupby("team")[stat]
              .transform(lambda s: s.shift(1).rolling(WINDOW, min_periods=WINDOW).mean())
        )

    # Days since the team's previous game (rest). Kept as a descriptive column only;
    # it is NOT one of the planned model features.
    df["rest_days"] = df.groupby("team")["game_date"].diff().dt.days
    return df


def main() -> None:
    print("Loading raw data ...")
    raw = load_raw()
    print(f"Total raw rows: {len(raw)}")

    print("Validating ...")
    clean = validate(raw)

    print("Engineering rolling features ...")
    feats = add_rolling_features(clean)
    feats.to_csv(PROC_DIR / "all_games_clean.csv", index=False)

    # Drop games without a full 5-game history (each team's first five games)
    model_df = feats.dropna(subset=[f"avg_{s}_last5" for s in ROLL_STATS]).copy()
    model_df["split"] = (model_df["game_date"] >= SPLIT_DATE).map({False: "train", True: "test"})
    model_df.to_csv(PROC_DIR / "model_dataset.csv", index=False)

    n_train = (model_df["split"] == "train").sum()
    n_test = (model_df["split"] == "test").sum()
    tr = model_df[model_df["split"] == "train"]["game_date"]
    te = model_df[model_df["split"] == "test"]["game_date"]
    print(f"Modeling rows: {len(model_df)}  (dropped {len(feats) - len(model_df)} early-season rows)")
    print(f"  train: {n_train} ({n_train/len(model_df):.2%})  {tr.min().date()} to {tr.max().date()}")
    print(f"  test : {n_test} ({n_test/len(model_df):.2%})  {te.min().date()} to {te.max().date()}")
    print(f"Saved -> {PROC_DIR.relative_to(ROOT)}/all_games_clean.csv, model_dataset.csv")


if __name__ == "__main__":
    main()
