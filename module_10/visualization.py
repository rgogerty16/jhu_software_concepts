"""Module 10 - NFL data dashboard: what gives a team the edge since 1999?

This script loads 25+ seasons of NFL game results (the open-source nflverse
"games" dataset) and produces the four exploratory visualizations that drive
the Module 10 dashboard.  Together they answer one leading question:

    "What gives an NFL team the edge - and how has the game changed since 1999?"

Pipeline
--------
1. Load the raw one-row-per-game table into a Pandas DataFrame (a local
   snapshot, falling back to a one-time download from nflverse).
2. Clean the data: drop unplayed games, coerce scores to numbers, fold
   relocated franchises into their current code and keep the 1999-2025 seasons.
3. Reshape the games into a one-row-per-team-per-game frame so that team,
   season and home/away statistics can be aggregated.
4. Summarize the headline numbers (home-field edge, best franchise, scoring
   growth) to the console so the README never hard-codes a result.
5. Write three Seaborn ``.png`` figures and one interactive, animated Plotly
   ``.html`` figure, all sharing a single colour palette.

Only regular-season games are used for rate statistics so win percentages are
not distorted by the uneven number of playoff games each team plays.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

# --- Configuration ---------------------------------------------------------
HERE = Path(__file__).resolve().parent
DATA_FILE = HERE / "nfl_games.csv"
GAMES_URL = (
    "https://raw.githubusercontent.com/nflverse/nfldata/master/data/games.csv"
)

# Shared, colour-blind-safe palette reused by every Seaborn and Plotly figure.
NAVY = "#1b3a5b"             # primary series / home team
ORANGE = "#e07a2e"           # accent / away team / callouts
REFERENCE_GREY = "#8c8c8c"   # 50% baseline and league-average guide lines
SEQUENTIAL_SCALE = "Blues"   # continuous scale for win % / point differential

FIRST_SEASON = 1999          # first season present in the dataset
LAST_SEASON = 2025           # 2026 rows are unplayed and are dropped
EMPTY_STADIUM_SEASON = 2020  # COVID season played with few or no fans
MIN_TEAM_GAMES = 100         # franchises need a real sample for the bar chart

# Historical team codes folded into each franchise's current code.
FRANCHISE_RENAMES = {"OAK": "LV", "SD": "LAC", "STL": "LA"}

# Decade buckets used for the scoring-era box plot (right-inclusive edges).
DECADE_BINS = [1998, 2009, 2019, 2025]
DECADE_LABELS = ["1999-2009", "2010-2019", "2020-2025"]

# Short, recognisable nickname for each current franchise code.
TEAM_NICKNAMES = {
    "ARI": "Cardinals", "ATL": "Falcons", "BAL": "Ravens", "BUF": "Bills",
    "CAR": "Panthers", "CHI": "Bears", "CIN": "Bengals", "CLE": "Browns",
    "DAL": "Cowboys", "DEN": "Broncos", "DET": "Lions", "GB": "Packers",
    "HOU": "Texans", "IND": "Colts", "JAX": "Jaguars", "KC": "Chiefs",
    "LA": "Rams", "LAC": "Chargers", "LV": "Raiders", "MIA": "Dolphins",
    "MIN": "Vikings", "NE": "Patriots", "NO": "Saints", "NYG": "Giants",
    "NYJ": "Jets", "PHI": "Eagles", "PIT": "Steelers", "SEA": "Seahawks",
    "SF": "49ers", "TB": "Buccaneers", "TEN": "Titans", "WAS": "Commanders",
}


def load_games():
    """Return the raw games table, caching a snapshot on first run."""
    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)
    frame = pd.read_csv(GAMES_URL)
    frame.to_csv(DATA_FILE, index=False)
    return frame


def clean_games(games):
    """Drop unplayed games, coerce scores and normalise franchise codes."""
    frame = games.copy()
    for column in ("home_score", "away_score", "season", "result"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["home_score", "away_score"])
    frame = frame[frame["season"].between(FIRST_SEASON, LAST_SEASON)]
    frame["season"] = frame["season"].astype(int)
    frame["home_team"] = frame["home_team"].replace(FRANCHISE_RENAMES)
    frame["away_team"] = frame["away_team"].replace(FRANCHISE_RENAMES)
    frame["total_points"] = frame["home_score"] + frame["away_score"]
    return frame.reset_index(drop=True)


def build_team_game_frame(games):
    """Explode regular-season games into one row per team per game."""
    reg = games[games["game_type"] == "REG"]
    home = pd.DataFrame({
        "season": reg["season"],
        "team": reg["home_team"],
        "is_home": True,
        "points_for": reg["home_score"],
        "points_against": reg["away_score"],
        "margin": reg["result"],
    })
    away = pd.DataFrame({
        "season": reg["season"],
        "team": reg["away_team"],
        "is_home": False,
        "points_for": reg["away_score"],
        "points_against": reg["home_score"],
        "margin": -reg["result"],
    })
    team_games = pd.concat([home, away], ignore_index=True)
    team_games["win_value"] = np.where(
        team_games["margin"] > 0, 1.0,
        np.where(team_games["margin"] < 0, 0.0, 0.5),
    )
    team_games["side"] = np.where(team_games["is_home"], "Home", "Away")
    return team_games


def home_win_pct_by_season(games):
    """Return the home-team win percentage for each regular season."""
    reg = games[games["game_type"] == "REG"].copy()
    reg["home_win"] = np.where(
        reg["result"] > 0, 1.0, np.where(reg["result"] < 0, 0.0, 0.5)
    )
    return reg.groupby("season")["home_win"].mean().mul(100)


def franchise_win_pct(team_games):
    """Return each franchise's win % and average point differential."""
    grouped = team_games.groupby("team")
    stats = pd.DataFrame({
        "win_pct": grouped["win_value"].mean().mul(100),
        "point_diff": grouped["points_for"].mean() - grouped["points_against"].mean(),
        "games": grouped.size(),
    })
    stats = stats[stats["games"] >= MIN_TEAM_GAMES]
    return stats.sort_values("win_pct", ascending=False)


def add_decade(team_games):
    """Return *team_games* with a categorical decade column added."""
    decade = pd.cut(team_games["season"], bins=DECADE_BINS, labels=DECADE_LABELS)
    return team_games.assign(decade=decade)


def team_season_stats(team_games):
    """Aggregate per-team, per-season offence, defence, record and win %."""
    grouped = team_games.groupby(["season", "team"])
    stats = grouped.agg(
        points_for=("points_for", "mean"),
        points_against=("points_against", "mean"),
        win_pct=("win_value", "mean"),
        games=("win_value", "size"),
        wins=("win_value", lambda v: int((v == 1).sum())),
        losses=("win_value", lambda v: int((v == 0).sum())),
        ties=("win_value", lambda v: int((v == 0.5).sum())),
    ).reset_index()
    stats["win_pct"] = stats["win_pct"] * 100
    stats["record"] = (
        stats["wins"].astype(str) + "-" + stats["losses"].astype(str)
        + "-" + stats["ties"].astype(str)
    )
    return stats


def plot_home_advantage(games, path):
    """Line-plot the home-team win % for every season (Seaborn)."""
    data = home_win_pct_by_season(games).reset_index(name="home_win_pct")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data, x="season", y="home_win_pct", marker="o",
                 color=NAVY, ax=ax, label="Home win %")
    ax.axhline(50, color=REFERENCE_GREY, linestyle="--", label="Coin flip (50%)")
    dip = data.loc[data["season"] == EMPTY_STADIUM_SEASON, "home_win_pct"]
    if not dip.empty:
        ax.annotate(
            "2020: empty\nstadiums (COVID)",
            xy=(EMPTY_STADIUM_SEASON, float(dip.iloc[0])),
            xytext=(EMPTY_STADIUM_SEASON - 7, 41),
            arrowprops={"arrowstyle": "->", "color": ORANGE},
            color=ORANGE, fontsize=9,
        )
    ax.set_title("NFL Home-Field Advantage by Season (1999-2025)")
    ax.set_xlabel("Season")
    ax.set_ylabel("Home-team win percentage (%)")
    ax.set_ylim(35, 70)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_team_win_pct(team_games, path):
    """Bar-chart each franchise's win %, shaded by point differential."""
    stats = franchise_win_pct(team_games).reset_index()
    stats["label"] = stats["team"].map(TEAM_NICKNAMES).fillna(stats["team"])

    norm = Normalize(stats["point_diff"].min(), stats["point_diff"].max())
    cmap = colormaps[SEQUENTIAL_SCALE]
    colours = {row.label: cmap(norm(row.point_diff)) for row in stats.itertuples()}

    fig, ax = plt.subplots(figsize=(9, 10))
    sns.barplot(data=stats, x="win_pct", y="label", hue="label",
                order=stats["label"].tolist(), palette=colours, legend=False, ax=ax)
    ax.axvline(50, color=REFERENCE_GREY, linestyle="--", label="Coin flip (50%)")
    ax.set_title("Regular-Season Win % by Franchise (1999-2025)")
    ax.set_xlabel("Win percentage (%)")
    ax.set_ylabel("Franchise")
    ax.legend(loc="lower right")

    scalar = ScalarMappable(cmap=cmap, norm=norm)
    scalar.set_array([])
    fig.colorbar(scalar, ax=ax, label="Avg point differential (pts/game)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_points_by_decade(team_games, path):
    """Box-plot per-team points scored per game across decades (Seaborn)."""
    data = add_decade(team_games).dropna(subset=["decade"])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(data=data, x="decade", y="points_for", hue="side",
                palette={"Home": NAVY, "Away": ORANGE}, ax=ax)
    ax.set_title("Points Scored per Team per Game by Decade")
    ax.set_xlabel("Decade")
    ax.set_ylabel("Points scored (per team, per game)")
    ax.legend(title="Team location", loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def build_offense_vs_defense_fig(team_games):
    """Return an interactive, animated Plotly scatter of offence vs defence."""
    stats = team_season_stats(team_games)
    stats["nickname"] = stats["team"].map(TEAM_NICKNAMES).fillna(stats["team"])
    mean_pf = float(stats["points_for"].mean())
    mean_pa = float(stats["points_against"].mean())
    pad = 2.0
    x_range = [stats["points_for"].min() - pad, stats["points_for"].max() + pad]
    y_range = [stats["points_against"].min() - pad, stats["points_against"].max() + pad]
    order = [str(season) for season in sorted(stats["season"].unique())]
    stats["season"] = stats["season"].astype(str)

    fig = px.scatter(
        stats, x="points_for", y="points_against", animation_frame="season",
        color="win_pct", size="games", hover_name="nickname",
        hover_data={"record": True, "win_pct": ":.1f", "points_for": ":.1f",
                    "points_against": ":.1f", "games": False, "season": False},
        color_continuous_scale=SEQUENTIAL_SCALE, range_color=(0, 100),
        range_x=x_range, range_y=y_range, size_max=18,
        category_orders={"season": order},
        labels={"points_for": "Points scored per game",
                "points_against": "Points allowed per game", "win_pct": "Win %"},
        title="NFL Team Offense vs. Defense by Season (drag the slider to play)",
    )
    fig.add_vline(x=mean_pf, line_dash="dot", line_color=REFERENCE_GREY)
    fig.add_hline(y=mean_pa, line_dash="dot", line_color=REFERENCE_GREY)
    fig.update_layout(template="plotly_white")
    return fig


def summarize(games, team_games):
    """Print the headline numbers so the README never hard-codes a result."""
    home_by_season = home_win_pct_by_season(games)
    franchises = franchise_win_pct(team_games)
    decade = add_decade(team_games).dropna(subset=["decade"])
    points = decade.groupby("decade", observed=True)["points_for"].mean().mul(2)

    print(f"Games (played, {FIRST_SEASON}-{LAST_SEASON}): {len(games):,}")
    print(f"Regular-season team-games: {len(team_games):,}")
    print(f"Average season home win %: {home_by_season.mean():.1f}")
    if EMPTY_STADIUM_SEASON in home_by_season.index:
        print(f"Home win % in {EMPTY_STADIUM_SEASON}: "
              f"{home_by_season.loc[EMPTY_STADIUM_SEASON]:.1f}")
    best = franchises.index[0]
    print(f"Best franchise: {TEAM_NICKNAMES.get(best, best)} "
          f"({franchises.iloc[0]['win_pct']:.1f}% win)")
    print(f"Avg combined points/game {DECADE_LABELS[0]}: {points.iloc[0]:.1f}")
    print(f"Avg combined points/game {DECADE_LABELS[-1]}: {points.iloc[-1]:.1f}")


def main():
    """Run the pipeline and write every visualization deliverable."""
    matplotlib.use("Agg")  # render to files without a display
    sns.set_theme(style="whitegrid")

    games = clean_games(load_games())
    team_games = build_team_game_frame(games)
    summarize(games, team_games)

    plot_home_advantage(games, HERE / "home_advantage_by_season.png")
    plot_team_win_pct(team_games, HERE / "team_win_pct.png")
    plot_points_by_decade(team_games, HERE / "points_per_game_by_decade.png")

    figure = build_offense_vs_defense_fig(team_games)
    figure.write_html(HERE / "offense_vs_defense.html", include_plotlyjs=True,
                      auto_play=False)
    print(f"All visualizations written to {HERE}")


if __name__ == "__main__":
    main()
