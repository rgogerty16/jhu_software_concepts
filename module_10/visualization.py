"""Module 10 - Peak vs. Longevity: LeBron James vs. Michael Jordan.

This script loads season-by-season and roster data (collected from the public
Basketball-Reference database) and produces the four visualizations that drive
the Module 10 dashboard.  Together they explore one leading question:

    "Peak vs. Longevity: Is LeBron the GOAT? A data comparison with Jordan."

Pipeline
--------
1. Load each player's cleaned season table and LeBron's teammate table from the
   committed CSV snapshots.
2. Derive cumulative career points, games-weighted per-game averages and each
   teammate's birth decade.
3. Print a console summary of the headline numbers so the README and dashboard
   tiles are generated from code, never hard-coded.
4. Write three Seaborn ``.png`` figures and one interactive, animated Plotly
   ``.html`` figure, all sharing a single colour palette.

The comparison is descriptive: "GOAT" is inherently subjective, so the figures
present Jordan's peak-scoring case and LeBron's longevity case side by side
rather than declaring a winner.
"""

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colormaps
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns

# --- Configuration ---------------------------------------------------------
HERE = Path(__file__).resolve().parent
LEBRON_FILE = HERE / "lebron_seasons.csv"
JORDAN_FILE = HERE / "jordan_seasons.csv"
TEAMMATES_FILE = HERE / "lebron_teammates.csv"

# Shared, team-evocative palette reused by every figure.
LEBRON = "#552583"          # Lakers purple
JORDAN = "#C8102E"          # Bulls red
REFERENCE_GREY = "#8c8c8c"  # guide / reference lines
DECADE_SCALE = "Purples"    # sequential shades for teammate birth-decades

KAREEM_RECORD = 38387       # Kareem's total that LeBron passed on 2023-02-07

# Per-game statistics compared in the career "fingerprint" chart.
STAT_LABELS = {"ppg": "PPG", "rpg": "RPG", "apg": "APG",
               "spg": "SPG", "bpg": "BPG"}


def load_seasons(path):
    """Load a player's season table, cleaned and cumulatively scored."""
    frame = pd.read_csv(path).dropna(subset=["games", "pts_total"])
    frame = frame.sort_values("age").reset_index(drop=True)
    frame["age"] = frame["age"].astype(int)
    frame["cum_points"] = frame["pts_total"].cumsum()
    return frame


def load_teammates(path):
    """Load LeBron's teammates, one row per player with a birth decade."""
    frame = pd.read_csv(path).dropna(subset=["birth_year"]).copy()
    frame["birth_year"] = frame["birth_year"].astype(int)
    frame["decade"] = (frame["birth_year"] // 10 * 10).astype(str) + "s"
    return frame


def career_averages(seasons):
    """Return games-weighted career per-game averages for one player."""
    games = seasons["games"]
    return {stat: float((seasons[stat] * games).sum() / games.sum())
            for stat in STAT_LABELS}


def player_key_stats(seasons):
    """Return the headline career numbers used in summaries and tiles."""
    return {
        "seasons": int(len(seasons)),
        "points": int(seasons["pts_total"].sum()),
        "ppg": float(seasons["pts_total"].sum() / seasons["games"].sum()),
        "peak_ppg": float(seasons["ppg"].max()),
        "age_min": int(seasons["age"].min()),
        "age_max": int(seasons["age"].max()),
    }


def plot_scoring_by_age(lebron, jordan, path):
    """Line-plot points per game against age for both players (Seaborn)."""
    data = pd.concat([lebron.assign(player="LeBron James"),
                      jordan.assign(player="Michael Jordan")])

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.lineplot(data=data, x="age", y="ppg", hue="player", marker="o",
                 palette={"LeBron James": LEBRON, "Michael Jordan": JORDAN},
                 ax=ax)
    ax.set_title("Scoring by Age: Jordan's Peak vs. LeBron's Longevity")
    ax.set_xlabel("Age (season)")
    ax.set_ylabel("Points per game (PPG)")
    ax.legend(title="Player", loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_career_fingerprint(lebron, jordan, path):
    """Grouped bar of games-weighted career per-game averages (Seaborn)."""
    rows = []
    for name, seasons in (("LeBron James", lebron), ("Michael Jordan", jordan)):
        for stat, value in career_averages(seasons).items():
            rows.append({"player": name, "stat": STAT_LABELS[stat],
                         "value": value})
    data = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=data, x="stat", y="value", hue="player",
                palette={"LeBron James": LEBRON, "Michael Jordan": JORDAN},
                ax=ax)
    for container in ax.containers:
        ax.bar_label(container, fmt="%.1f", padding=2, fontsize=8)
    ax.set_title("Career Per-Game Averages: Scorer vs. All-Arounder")
    ax.set_xlabel("Statistic (per game)")
    ax.set_ylabel("Value")
    ax.legend(title="Player", loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_teammate_span(teammates, path):
    """Strip-plot every teammate by birth year, coloured by decade (Seaborn)."""
    order = sorted(teammates["decade"].unique())
    shades = colormaps[DECADE_SCALE](np.linspace(0.35, 0.95, len(order)))
    palette = dict(zip(order, shades))
    data = teammates.assign(row="Teammates")
    oldest = teammates.loc[teammates["birth_year"].idxmin()]
    youngest = teammates.loc[teammates["birth_year"].idxmax()]
    note = ", his son" if youngest["player"] == "Bronny James" else ""

    np.random.seed(42)  # reproducible jitter
    fig, ax = plt.subplots(figsize=(11, 5))
    sns.stripplot(data=data, x="birth_year", y="row", hue="decade",
                  hue_order=order, palette=palette, jitter=0.35, size=6,
                  alpha=0.85, ax=ax)
    ax.annotate(f"Oldest: {oldest['player']} ({oldest['birth_year']})",
                xy=(oldest["birth_year"], 0), xytext=(oldest["birth_year"], -0.38),
                ha="center", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": REFERENCE_GREY})
    ax.annotate(f"Youngest: {youngest['player']} ({youngest['birth_year']}{note})",
                xy=(youngest["birth_year"], 0), xytext=(youngest["birth_year"], 0.42),
                ha="center", fontsize=9,
                arrowprops={"arrowstyle": "->", "color": REFERENCE_GREY})
    ax.set_title(f"LeBron's {len(teammates)} Teammates Span {len(order)} Birth Decades")
    ax.set_xlabel("Teammate birth year")
    ax.set_ylabel("")
    ax.legend(title="Born in", loc="upper center", ncol=len(order),
              bbox_to_anchor=(0.5, -0.18))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def build_scoring_chase_fig(lebron, jordan):
    """Return an animated Plotly race of cumulative career points by age.

    Both players are given a point at every age in the range (0 before their
    debut) so each line is present in the first frame, which Plotly Express
    requires for the trace to animate.
    """
    players = {"LeBron James": lebron, "Michael Jordan": jordan}
    low = int(min(lebron["age"].min(), jordan["age"].min()))
    high = int(max(lebron["age"].max(), jordan["age"].max()))
    rows = []
    for name, seasons in players.items():
        pts_by_age = seasons.set_index("age")["pts_total"]
        running = 0
        for age in range(low, high + 1):
            running += int(pts_by_age.get(age, 0))
            rows.append({"player": name, "age": age, "cum_points": running})
    grid = pd.DataFrame(rows)
    snaps = pd.concat(
        [grid[grid["age"] <= frame].assign(frame=frame)
         for frame in range(low, high + 1)],
        ignore_index=True,
    )

    fig = px.line(
        snaps, x="age", y="cum_points", color="player", animation_frame="frame",
        markers=True, range_x=[low - 1, high + 1], range_y=[0, 45000],
        color_discrete_map={"LeBron James": LEBRON, "Michael Jordan": JORDAN},
        labels={"age": "Age (season)", "cum_points": "Cumulative career points",
                "player": "Player"},
        title="The Scoring Chase: Cumulative Career Points by Age (press play)",
    )
    fig.add_hline(y=KAREEM_RECORD, line_dash="dot", line_color=REFERENCE_GREY,
                  annotation_text=f"Kareem's record ({KAREEM_RECORD:,})",
                  annotation_position="top left")
    fig.update_layout(template="plotly_white")
    return fig


def summarize(lebron, jordan, teammates):
    """Print the headline numbers used by the README and dashboard tiles."""
    leb, jor = player_key_stats(lebron), player_key_stats(jordan)
    print(f"LeBron: {leb['seasons']} seasons, {leb['points']:,} pts, "
          f"{leb['ppg']:.1f} PPG, peak {leb['peak_ppg']:.1f}, "
          f"ages {leb['age_min']}-{leb['age_max']}")
    print(f"Jordan: {jor['seasons']} seasons, {jor['points']:,} pts, "
          f"{jor['ppg']:.1f} PPG, peak {jor['peak_ppg']:.1f}, "
          f"ages {jor['age_min']}-{jor['age_max']}")
    decades = sorted(teammates["decade"].unique())
    print(f"Teammates: {len(teammates)} across {len(decades)} decades "
          f"({teammates['birth_year'].min()}-{teammates['birth_year'].max()}): "
          f"{', '.join(decades)}")


def main():
    """Load the data, print the summary and write every visualization."""
    matplotlib.use("Agg")  # render to files without a display
    sns.set_theme(style="whitegrid")

    lebron = load_seasons(LEBRON_FILE)
    jordan = load_seasons(JORDAN_FILE)
    teammates = load_teammates(TEAMMATES_FILE)
    summarize(lebron, jordan, teammates)

    plot_scoring_by_age(lebron, jordan, HERE / "scoring_by_age.png")
    plot_career_fingerprint(lebron, jordan, HERE / "career_fingerprint.png")
    plot_teammate_span(teammates, HERE / "teammate_birth_span.png")

    figure = build_scoring_chase_fig(lebron, jordan)
    figure.write_html(HERE / "scoring_chase.html", include_plotlyjs=True,
                      auto_play=False)
    print(f"All visualizations written to {HERE}")


if __name__ == "__main__":
    main()
