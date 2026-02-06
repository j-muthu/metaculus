"""
Baden-Württemberg Polling Data Dashboard
Interactive Plotly Dash web app for analyzing state-level polling data.
"""

import pandas as pd
import numpy as np
from dash import Dash, dcc, html, Input, Output, State, callback
import plotly.graph_objects as go
import plotly.express as px
from scipy import stats

# ---------------------------------------------------------------------------
# Data Loading & Preprocessing
# ---------------------------------------------------------------------------

CSV_PATH = "bw_polls.csv"

df = pd.read_csv(CSV_PATH)
df["fieldwork_end"] = pd.to_datetime(df["fieldwork_end"], errors="coerce")
df = df.sort_values("fieldwork_end").reset_index(drop=True)

PARTY_COLS = ["gruene", "cdu", "spd", "fdp", "afd", "linke", "fw", "bsw", "others"]
PARTY_LABELS = {
    "gruene": "Grüne",
    "cdu": "CDU",
    "spd": "SPD",
    "fdp": "FDP",
    "afd": "AfD",
    "linke": "Linke",
    "fw": "FW",
    "bsw": "BSW",
    "others": "Others",
}
PARTY_COLORS = {
    "gruene": "#1B7A2B",
    "cdu": "#000000",
    "spd": "#E3000F",
    "fdp": "#FFED00",
    "afd": "#009EE0",
    "linke": "#BE3075",
    "fw": "#FF6600",
    "bsw": "#8B0000",
    "others": "#808080",
}

for col in PARTY_COLS:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Separate elections from polls
df["is_federal_election"] = df["is_federal_election"].fillna(0).astype(int)
df["is_state_election"] = df["is_state_election"].fillna(0).astype(int)

elections = df[(df["is_federal_election"] == 1) | (df["is_state_election"] == 1)].copy()
polls = df[(df["is_federal_election"] == 0) & (df["is_state_election"] == 0)].copy()

polling_firms = sorted(polls["polling_firm"].dropna().unique().tolist())

min_date = df["fieldwork_end"].min()
max_date = df["fieldwork_end"].max()

# ---------------------------------------------------------------------------
# App Layout
# ---------------------------------------------------------------------------

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "BW Polling Dashboard"

app.layout = html.Div(
    style={"fontFamily": "Segoe UI, Roboto, sans-serif", "margin": "0 auto", "maxWidth": "1400px", "padding": "20px"},
    children=[
        html.H1("Baden-Württemberg Polling Dashboard", style={"textAlign": "center", "marginBottom": "5px"}),
        html.P(
            "Interactive analysis of state-level polling data",
            style={"textAlign": "center", "color": "#666", "marginTop": "0"},
        ),
        dcc.Tabs(
            id="tabs",
            value="tab-timeseries",
            children=[
                dcc.Tab(label="Polling Time Series", value="tab-timeseries"),
                dcc.Tab(label="Correlation Analysis", value="tab-correlation"),
            ],
        ),
        html.Div(id="tab-content"),
    ],
)

# ---------------------------------------------------------------------------
# Tab layouts
# ---------------------------------------------------------------------------


def build_timeseries_tab():
    return html.Div(
        [
            html.Div(
                style={"display": "flex", "flexWrap": "wrap", "gap": "20px", "marginTop": "15px", "marginBottom": "10px"},
                children=[
                    html.Div(
                        [
                            html.Label("Moving Average Window", style={"fontWeight": "bold"}),
                            dcc.Slider(
                                id="ma-slider",
                                min=2,
                                max=15,
                                step=1,
                                value=5,
                                marks={i: str(i) for i in range(2, 16)},
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                        style={"flex": "1", "minWidth": "300px"},
                    ),
                    html.Div(
                        [
                            html.Label("Date Range", style={"fontWeight": "bold"}),
                            dcc.DatePickerRange(
                                id="date-range",
                                min_date_allowed=min_date,
                                max_date_allowed=max_date,
                                start_date=min_date,
                                end_date=max_date,
                                display_format="YYYY-MM-DD",
                            ),
                        ],
                        style={"minWidth": "250px"},
                    ),
                    html.Div(
                        [
                            html.Label("Polling Firm", style={"fontWeight": "bold"}),
                            dcc.Dropdown(
                                id="firm-dropdown",
                                options=[{"label": "All", "value": "All"}]
                                + [{"label": f, "value": f} for f in polling_firms],
                                value="All",
                                clearable=False,
                                style={"width": "220px"},
                            ),
                        ],
                    ),
                ],
            ),
            html.Div(
                [
                    html.Label("Parties", style={"fontWeight": "bold"}),
                    dcc.Checklist(
                        id="party-checklist",
                        options=[{"label": PARTY_LABELS[p], "value": p} for p in PARTY_COLS],
                        value=PARTY_COLS.copy(),
                        inline=True,
                        style={"display": "flex", "flexWrap": "wrap", "gap": "12px"},
                    ),
                ],
                style={"marginBottom": "10px"},
            ),
            dcc.Loading(dcc.Graph(id="timeseries-graph", style={"height": "600px"})),
        ]
    )


def build_correlation_tab():
    party_options = [{"label": PARTY_LABELS[p], "value": p} for p in PARTY_COLS]
    return html.Div(
        [
            html.Div(
                style={"display": "flex", "flexWrap": "wrap", "gap": "20px", "marginTop": "15px", "marginBottom": "10px"},
                children=[
                    html.Div(
                        [
                            html.Label("Party X", style={"fontWeight": "bold"}),
                            dcc.Dropdown(id="corr-party-x", options=party_options, value="gruene", clearable=False, style={"width": "180px"}),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Party Y", style={"fontWeight": "bold"}),
                            dcc.Dropdown(id="corr-party-y", options=party_options, value="cdu", clearable=False, style={"width": "180px"}),
                        ]
                    ),
                    html.Div(
                        [
                            html.Label("Rolling Window", style={"fontWeight": "bold"}),
                            dcc.Slider(
                                id="corr-window",
                                min=3,
                                max=20,
                                step=1,
                                value=8,
                                marks={i: str(i) for i in [3, 5, 8, 10, 15, 20]},
                                tooltip={"placement": "bottom"},
                            ),
                        ],
                        style={"flex": "1", "minWidth": "250px"},
                    ),
                ],
            ),
            html.Div(
                style={"display": "flex", "flexWrap": "wrap", "gap": "15px"},
                children=[
                    dcc.Loading(dcc.Graph(id="corr-heatmap", style={"flex": "1", "minWidth": "450px", "height": "500px"})),
                    dcc.Loading(dcc.Graph(id="corr-scatter", style={"flex": "1", "minWidth": "450px", "height": "500px"})),
                ],
            ),
            dcc.Loading(dcc.Graph(id="corr-rolling", style={"height": "400px"})),
        ]
    )


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------


@callback(Output("tab-content", "children"), Input("tabs", "value"))
def render_tab(tab):
    if tab == "tab-timeseries":
        return build_timeseries_tab()
    return build_correlation_tab()


# ---- Time Series ----


@callback(
    Output("timeseries-graph", "figure"),
    Input("ma-slider", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
    Input("firm-dropdown", "value"),
    Input("party-checklist", "value"),
)
def update_timeseries(ma_window, start_date, end_date, firm, selected_parties):
    fig = go.Figure()

    # Filter polls
    mask = (polls["fieldwork_end"] >= start_date) & (polls["fieldwork_end"] <= end_date)
    if firm != "All":
        mask &= polls["polling_firm"] == firm
    filtered = polls[mask].copy()

    # Election markers (vertical bands)
    for _, row in elections.iterrows():
        edate = row["fieldwork_end"]
        if pd.isna(edate) or edate < pd.Timestamp(start_date) or edate > pd.Timestamp(end_date):
            continue
        is_fed = row["is_federal_election"] == 1
        color = "rgba(0,0,200,0.12)" if is_fed else "rgba(200,0,0,0.12)"
        label = row["polling_firm"] if pd.notna(row["polling_firm"]) else ("Federal" if is_fed else "State")
        fig.add_vrect(
            x0=edate - pd.Timedelta(days=1),
            x1=edate + pd.Timedelta(days=1),
            fillcolor=color,
            line_width=0,
            annotation_text=label,
            annotation_position="top left",
            annotation_font_size=9,
            annotation_textangle=-90,
        )

    # Plot each party
    for party in selected_parties:
        label = PARTY_LABELS[party]
        color = PARTY_COLORS[party]
        series = filtered[["fieldwork_end", party]].dropna(subset=[party])
        if series.empty:
            continue

        # Scatter points
        fig.add_trace(
            go.Scatter(
                x=series["fieldwork_end"],
                y=series[party],
                mode="markers",
                name=label,
                marker=dict(color=color, size=6, opacity=0.6),
                legendgroup=party,
                hovertemplate=f"{label}: %{{y:.1f}}%<br>%{{x|%Y-%m-%d}}<extra></extra>",
            )
        )

        # Moving average line
        if len(series) >= ma_window:
            ma = series[party].rolling(window=ma_window, min_periods=ma_window).mean()
            fig.add_trace(
                go.Scatter(
                    x=series["fieldwork_end"],
                    y=ma,
                    mode="lines",
                    name=f"{label} (MA-{ma_window})",
                    line=dict(color=color, width=2.5),
                    legendgroup=party,
                    showlegend=True,
                    hovertemplate=f"{label} MA: %{{y:.1f}}%<br>%{{x|%Y-%m-%d}}<extra></extra>",
                )
            )

        # Election result markers for this party
        for _, erow in elections.iterrows():
            edate = erow["fieldwork_end"]
            val = erow[party]
            if pd.isna(edate) or pd.isna(val):
                continue
            if edate < pd.Timestamp(start_date) or edate > pd.Timestamp(end_date):
                continue
            is_fed = erow["is_federal_election"] == 1
            symbol = "diamond" if is_fed else "star"
            fig.add_trace(
                go.Scatter(
                    x=[edate],
                    y=[val],
                    mode="markers",
                    marker=dict(color=color, size=12, symbol=symbol, line=dict(width=1.5, color="white")),
                    legendgroup=party,
                    showlegend=False,
                    hovertemplate=f"{label} ({'Federal' if is_fed else 'State'} election): {val:.1f}%<br>%{{x|%Y-%m-%d}}<extra></extra>",
                )
            )

    fig.update_layout(
        title="Polling Time Series – Baden-Württemberg",
        xaxis_title="Date",
        yaxis_title="Vote Share (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5),
        margin=dict(b=120),
        template="plotly_white",
    )
    return fig


# ---- Correlation Tab ----


@callback(
    Output("corr-heatmap", "figure"),
    Output("corr-scatter", "figure"),
    Output("corr-rolling", "figure"),
    Input("corr-party-x", "value"),
    Input("corr-party-y", "value"),
    Input("corr-window", "value"),
)
def update_correlation(party_x, party_y, window):
    # Use polls only (exclude elections)
    corr_df = polls[PARTY_COLS].dropna(how="all")

    # --- Heatmap ---
    corr_matrix = corr_df.corr(method="pearson")
    labels = [PARTY_LABELS[p] for p in PARTY_COLS]
    heatmap_fig = go.Figure(
        go.Heatmap(
            z=corr_matrix.values,
            x=labels,
            y=labels,
            colorscale="RdBu_r",
            zmid=0,
            zmin=-1,
            zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            hovertemplate="(%{x}, %{y}): %{z:.2f}<extra></extra>",
        )
    )
    heatmap_fig.update_layout(
        title="Pearson Correlation Matrix (Polls Only)",
        template="plotly_white",
        margin=dict(l=80, r=20, t=50, b=80),
    )

    # --- Pairwise Scatter ---
    pair_df = polls[[party_x, party_y]].dropna()
    scatter_fig = go.Figure()
    scatter_fig.add_trace(
        go.Scatter(
            x=pair_df[party_x],
            y=pair_df[party_y],
            mode="markers",
            marker=dict(color=PARTY_COLORS[party_x], size=7, opacity=0.7),
            name="Polls",
            hovertemplate=f"{PARTY_LABELS[party_x]}: %{{x:.1f}}%<br>{PARTY_LABELS[party_y]}: %{{y:.1f}}%<extra></extra>",
        )
    )

    # Regression line
    if len(pair_df) >= 2:
        slope, intercept, r_value, p_value, std_err = stats.linregress(pair_df[party_x], pair_df[party_y])
        x_range = np.linspace(pair_df[party_x].min(), pair_df[party_x].max(), 50)
        scatter_fig.add_trace(
            go.Scatter(
                x=x_range,
                y=intercept + slope * x_range,
                mode="lines",
                line=dict(color="red", dash="dash", width=2),
                name=f"OLS (r={r_value:.2f})",
            )
        )

    scatter_fig.update_layout(
        title=f"{PARTY_LABELS[party_x]} vs {PARTY_LABELS[party_y]}",
        xaxis_title=f"{PARTY_LABELS[party_x]} (%)",
        yaxis_title=f"{PARTY_LABELS[party_y]} (%)",
        template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=60),
    )

    # --- Rolling Correlation ---
    roll_df = polls[["fieldwork_end", party_x, party_y]].dropna().sort_values("fieldwork_end").reset_index(drop=True)
    rolling_fig = go.Figure()

    if len(roll_df) >= window:
        rolling_corr = roll_df[party_x].rolling(window=window).corr(roll_df[party_y])
        rolling_fig.add_trace(
            go.Scatter(
                x=roll_df["fieldwork_end"],
                y=rolling_corr,
                mode="lines+markers",
                marker=dict(size=4),
                line=dict(width=2, color=PARTY_COLORS[party_x]),
                name=f"Rolling r (window={window})",
                hovertemplate="r = %{y:.2f}<br>%{x|%Y-%m-%d}<extra></extra>",
            )
        )
        rolling_fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)

    rolling_fig.update_layout(
        title=f"Rolling Correlation: {PARTY_LABELS[party_x]} & {PARTY_LABELS[party_y]} (window={window})",
        xaxis_title="Date",
        yaxis_title="Pearson r",
        yaxis=dict(range=[-1.05, 1.05]),
        template="plotly_white",
        margin=dict(l=60, r=20, t=50, b=60),
    )

    return heatmap_fig, scatter_fig, rolling_fig


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    app.run(debug=True)
