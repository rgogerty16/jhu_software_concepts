"""Module 10 - Dash dashboard for the NFL "team edge" analysis.

A single-page Dash application that presents the four Module 10 figures: the
three saved Seaborn ``.png`` charts plus the live, interactive and animated
Plotly scatter built in :mod:`visualization`.  The page title states the
research question and a short caption guides the viewer toward the conclusion.

Run with ``python dashboard.py`` and open http://127.0.0.1:8050 in a browser.
"""

import base64
from pathlib import Path

from dash import Dash, dcc, html

from visualization import (
    NAVY,
    build_offense_vs_defense_fig,
    build_team_game_frame,
    clean_games,
    load_games,
)

# --- Configuration ---------------------------------------------------------
HERE = Path(__file__).resolve().parent

RESEARCH_QUESTION = (
    "What gives an NFL team the edge - and how has the game changed since 1999?"
)

# The guiding conclusion text (kept to three sentences, per the brief).
GUIDE_TEXT = (
    "Home teams won about 56% of regular-season games since 1999 - an edge that "
    "collapsed to a coin flip in the empty-stadium 2020 season, hinting that "
    "crowds matter. A few franchises (led by the Patriots) stayed well above "
    ".500 across 25 years by outscoring opponents, while leaguewide scoring "
    "drifted upward into today's offense-friendly era. Together the charts "
    "suggest a team's edge comes mostly from a sustained scoring margin and, to "
    "a smaller degree, from playing at home."
)

# Saved Seaborn PNG panels: (filename, one-line caption).
PANELS = [
    ("home_advantage_by_season.png",
     "Home win % by season: the home edge holds near 56% but falls to a coin "
     "flip in 2020's empty stadiums."),
    ("team_win_pct.png",
     "Regular-season win % by franchise, shaded by average point differential: "
     "steady winners outscore their opponents."),
    ("points_per_game_by_decade.png",
     "Points scored per team per game by decade: medians drift upward, and home "
     "teams outscore visitors in every era."),
]

CONTAINER_STYLE = {"maxWidth": "980px", "margin": "0 auto", "padding": "24px",
                   "fontFamily": "Arial, Helvetica, sans-serif"}
TITLE_STYLE = {"color": NAVY, "textAlign": "center", "marginBottom": "8px"}
GUIDE_STYLE = {"fontSize": "17px", "lineHeight": "1.5", "color": "#333",
               "textAlign": "center", "maxWidth": "820px", "margin": "0 auto 8px"}
IMAGE_STYLE = {"width": "100%", "maxWidth": "900px", "display": "block",
               "margin": "0 auto", "border": "1px solid #e2e2e2"}
CAPTION_STYLE = {"textAlign": "center", "color": "#555", "marginTop": "8px"}
FIGURE_STYLE = {"margin": "36px 0"}


def _encoded_image(filename):
    """Return a base64 ``data:`` URI for the PNG *filename* in this folder."""
    payload = base64.b64encode((HERE / filename).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _plotly_figure():
    """Build the interactive Plotly figure from the cleaned dataset."""
    team_games = build_team_game_frame(clean_games(load_games()))
    return build_offense_vs_defense_fig(team_games)


def _image_panel(filename, caption):
    """Return one captioned image panel for a saved Seaborn PNG."""
    return html.Figure(
        style=FIGURE_STYLE,
        children=[
            html.Img(src=_encoded_image(filename), style=IMAGE_STYLE),
            html.Figcaption(caption, style=CAPTION_STYLE),
        ],
    )


def build_layout():
    """Assemble and return the single-page dashboard layout."""
    return html.Div(
        style=CONTAINER_STYLE,
        children=[
            html.H1(RESEARCH_QUESTION, style=TITLE_STYLE),
            html.P(GUIDE_TEXT, style=GUIDE_STYLE),
            dcc.Graph(figure=_plotly_figure()),
            html.Figcaption(
                "Interactive: hover a dot for a team's record, then press play "
                "or drag the slider; elite teams sit in the lower-right (score "
                "a lot, allow little).",
                style=CAPTION_STYLE,
            ),
            *[_image_panel(name, caption) for name, caption in PANELS],
            html.Footer(
                "Data: open-source nflverse games dataset, 1999-2025 regular "
                "seasons. Figures show associations, not causal claims.",
                style={"textAlign": "center", "color": "#888",
                       "marginTop": "24px", "fontSize": "13px"},
            ),
        ],
    )


def build_app():
    """Create and configure the Dash application."""
    app = Dash(__name__)
    app.title = "NFL Team Edge Dashboard"
    app.layout = build_layout()
    return app


if __name__ == "__main__":
    build_app().run(debug=False, port=8050)
