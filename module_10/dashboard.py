"""Module 10 - Dash dashboard for the LeBron-vs-Jordan "GOAT" analysis.

A single-page Dash application that presents the four Module 10 figures: a row
of fun-fact "longevity" tiles, the live interactive/animated Plotly scoring
chase built in :mod:`visualization`, and the three saved Seaborn ``.png``
charts.  The page title states the research question and a short caption guides
the viewer toward the peak-vs-longevity conclusion.

Run with ``python dashboard.py`` and open http://127.0.0.1:8050 in a browser.
"""

import base64
from pathlib import Path

from dash import Dash, dcc, html

from visualization import (
    JORDAN_FILE,
    LEBRON,
    LEBRON_FILE,
    TEAMMATES_FILE,
    build_scoring_chase_fig,
    load_seasons,
    load_teammates,
    player_key_stats,
)

# --- Configuration ---------------------------------------------------------
HERE = Path(__file__).resolve().parent

RESEARCH_QUESTION = (
    "Peak vs. Longevity: Is LeBron the GOAT? A Data Comparison with Michael Jordan"
)

# Guiding conclusion text (three sentences, per the brief).
GUIDE_TEXT = (
    "Michael Jordan owns the sharper peak - a record 30.1 career PPG and a higher "
    "scoring apex - while LeBron James owns the longevity: 23 seasons, the all-time "
    "scoring record, and more career rebounds and assists. The animation shows "
    "LeBron's point total climbing past Jordan's finish and Kareem's record, and his "
    "teammates span five birth decades, from Scott Williams (1968) to his own son "
    "(2004). Who is the GOAT is a matter of taste - the data simply lays out the peak "
    "case and the longevity case."
)

# Externally cited longevity facts (not derivable from the season tables).
PRESIDENTS = 4    # G.W. Bush, Obama, Trump, Biden - from his Oct 2003 debut onward
NBA_SHARE = 37    # % of all NBA players he has shared the floor with (Yahoo/StatMuse)

# Saved Seaborn PNG panels: (filename, one-line caption).
PANELS = [
    ("scoring_by_age.png",
     "Points per game by age: Jordan's red line peaks higher in his 20s, but "
     "LeBron keeps scoring into his 40s."),
    ("career_fingerprint.png",
     "Career per-game averages: Jordan leads scoring and steals; LeBron leads "
     "rebounds and assists - scorer vs. all-arounder."),
    ("teammate_birth_span.png",
     "Every LeBron teammate by birth year: from Scott Williams (1968) to his son "
     "Bronny (2004), five decades apart."),
]

CONTAINER_STYLE = {"maxWidth": "1000px", "margin": "0 auto", "padding": "24px",
                   "fontFamily": "Arial, Helvetica, sans-serif"}
TITLE_STYLE = {"color": LEBRON, "textAlign": "center", "marginBottom": "8px"}
GUIDE_STYLE = {"fontSize": "17px", "lineHeight": "1.5", "color": "#333",
               "textAlign": "center", "maxWidth": "860px", "margin": "0 auto 8px"}
TILES_ROW_STYLE = {"display": "flex", "flexWrap": "wrap",
                   "justifyContent": "center", "margin": "10px 0 24px"}
TILE_STYLE = {"flex": "1 1 160px", "minWidth": "150px", "maxWidth": "200px",
              "textAlign": "center", "padding": "16px 12px", "margin": "6px",
              "borderRadius": "10px", "background": "#f3effa",
              "border": f"1px solid {LEBRON}"}
TILE_VALUE_STYLE = {"fontSize": "26px", "fontWeight": "bold", "color": LEBRON}
TILE_LABEL_STYLE = {"fontSize": "13px", "color": "#444", "marginTop": "4px"}
IMAGE_STYLE = {"width": "100%", "maxWidth": "900px", "display": "block",
               "margin": "0 auto", "border": "1px solid #e2e2e2"}
CAPTION_STYLE = {"textAlign": "center", "color": "#555", "marginTop": "8px"}
FIGURE_STYLE = {"margin": "36px 0"}


def _encoded_image(filename):
    """Return a base64 ``data:`` URI for the PNG *filename* in this folder."""
    payload = base64.b64encode((HERE / filename).read_bytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _fact_tiles():
    """Return the fun-fact tiles: data-derived numbers plus cited facts."""
    stats = player_key_stats(load_seasons(LEBRON_FILE))
    teammates = load_teammates(TEAMMATES_FILE)
    facts = [
        (f"{stats['seasons']}", "Seasons played (NBA record)"),
        (f"{stats['points']:,}", "Career points (all-time #1)"),
        (f"{teammates['decade'].nunique()}", "Decades of teammate birth years"),
        (f"{PRESIDENTS}", "U.S. presidents during his career"),
        (f"~{NBA_SHARE}%", "of all NBA players shared his floor"),
    ]
    return [html.Div(style=TILE_STYLE, children=[
        html.Div(value, style=TILE_VALUE_STYLE),
        html.Div(label, style=TILE_LABEL_STYLE),
    ]) for value, label in facts]


def _plotly_figure():
    """Build the interactive, animated scoring-chase figure."""
    return build_scoring_chase_fig(load_seasons(LEBRON_FILE),
                                   load_seasons(JORDAN_FILE))


def _image_panel(filename, caption):
    """Return one captioned image panel for a saved Seaborn PNG."""
    return html.Figure(style=FIGURE_STYLE, children=[
        html.Img(src=_encoded_image(filename), style=IMAGE_STYLE),
        html.Figcaption(caption, style=CAPTION_STYLE),
    ])


def build_layout():
    """Assemble and return the single-page dashboard layout."""
    return html.Div(
        style=CONTAINER_STYLE,
        children=[
            html.H1(RESEARCH_QUESTION, style=TITLE_STYLE),
            html.P(GUIDE_TEXT, style=GUIDE_STYLE),
            html.Div(_fact_tiles(), style=TILES_ROW_STYLE),
            dcc.Graph(figure=_plotly_figure()),
            html.Figcaption(
                "Interactive: hover a point for that season, then press play or "
                "drag the slider to watch LeBron's total pass Jordan and Kareem.",
                style=CAPTION_STYLE,
            ),
            *[_image_panel(name, caption) for name, caption in PANELS],
            html.Footer(
                "Data: season and roster tables from Basketball-Reference "
                "(through 2025-26). Descriptive comparison, not a definitive verdict.",
                style={"textAlign": "center", "color": "#888",
                       "marginTop": "24px", "fontSize": "13px"},
            ),
        ],
    )


def build_app():
    """Create and configure the Dash application."""
    app = Dash(__name__)
    app.title = "LeBron vs. Jordan - GOAT Dashboard"
    app.layout = build_layout()
    return app


if __name__ == "__main__":
    build_app().run(debug=False, port=8050)
