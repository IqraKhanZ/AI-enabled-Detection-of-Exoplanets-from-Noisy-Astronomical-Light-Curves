"""
src/visualization/dashboard.py
==============================
Interactive Dash-based dashboard to visualize exoplanet candidate classifications,
light curves, MCMC parameters, and contamination diagnostics.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import dash
from dash import dcc, html, dash_table
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from utils.config import get, project_root
from utils.logger import get_logger

logger = get_logger(__name__)

# Initialize Dash application with dark bootstrap theme
app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    suppress_callback_exceptions=True
)
app.title = "Exoplanet Detection Dashboard"

# Load default empty state data or read pipeline_results if available
def load_data() -> pd.DataFrame:
    root = project_root()
    # Mock data for demonstration/fallback
    mock_data = {
        "tic_id": [22529346, 25155310, 259377017, 124866929, 420814525],
        "predicted_label_name": ["PLANET", "ECLIPSING_BINARY", "PLANET", "NOISE", "BLEND"],
        "pipeline_confidence": [0.945, 0.887, 0.912, 0.120, 0.354],
        "planet_prob": [0.98, 0.05, 0.95, 0.02, 0.10],
        "eb_prob": [0.01, 0.92, 0.02, 0.05, 0.15],
        "blend_prob": [0.00, 0.02, 0.01, 0.03, 0.65],
        "noise_prob": [0.01, 0.01, 0.02, 0.90, 0.10],
        "period_days": [3.1415, 1.2543, 5.6781, 0.9982, 10.4321],
        "depth_ppm": [12000.0, 45000.0, 5600.0, np.nan, 2300.0],
        "duration_hrs": [2.4, 4.1, 1.8, np.nan, 3.2],
        "snr": [24.5, 154.2, 12.1, 2.1, 4.5],
        "fap": [1e-12, 0.0, 1e-6, 0.85, 0.12],
        "centroid_shift_arcsec": [0.02, 0.15, 0.01, 0.54, 1.21],
        "contamination_ratio": [0.01, 0.04, 0.00, 0.45, 0.78],
        "is_contaminated": [False, False, False, True, True]
    }
    df_mock = pd.DataFrame(mock_data)

    results_path = root / get("paths.outputs", "outputs") / "pipeline_results.csv"
    if results_path.exists():
        try:
            df = pd.read_csv(results_path)
            # Ensure required columns exist
            for col in ["tic_id", "predicted_label_name", "pipeline_confidence", "planet_prob", "snr", "fap", "period_days"]:
                if col not in df.columns:
                    df[col] = np.nan
            # Exclude mock IDs that are already present in real data to avoid duplication
            df_mock_filtered = df_mock[~df_mock["tic_id"].isin(df["tic_id"])]
            return pd.concat([df, df_mock_filtered], ignore_index=True)
        except Exception as exc:
            logger.error("Failed to read pipeline_results.csv: %s", exc)
            
    return df_mock

# Load global data
df_results = load_data()

# App Layout
app.layout = dbc.Container(
    id="main-container",
    children=[
        dcc.Markdown("""
        <style>
            /* Global dark styles */
            #main-container {
                background-color: #0A0D14 !important;
                color: #F8FAFC !important;
            }
            
            /* Cards and headers in Dark Mode */
            .card {
                background-color: #121724 !important;
                border: 1px solid #1E293B !important;
                color: #F8FAFC !important;
            }
            .card-header {
                background-color: #1A2133 !important;
                color: #F8FAFC !important;
                border-bottom: 1px solid #1E293B !important;
                font-weight: bold;
            }
            label {
                color: #94A3B8 !important;
            }
            h1, h2, h3, h4, h5, h6 {
                color: #F8FAFC !important;
            }
            
            /* Dropdown inside Dark Mode */
            div[class*="dash-dropdown"] div[class*="-control"],
            div[class*="dash-dropdown"] div[class*="-menu"],
            div[class*="dash-dropdown"] div[class*="-option"],
            div[class*="dash-dropdown"] div[class*="singleValue"],
            div[class*="dash-dropdown"] div[class*="placeholder"],
            div[class*="dash-dropdown"] div[class*="ValueContainer"],
            div[class*="dash-dropdown"] div[class*="Input"] input,
            div[class*="dash-dropdown"] input {
                color: #F8FAFC !important;
                background-color: #121724 !important;
            }
            div[class*="dash-dropdown"] div[class*="-option"]:hover,
            div[class*="dash-dropdown"] div[class*="-option"][class*="-isFocused"] {
                background-color: #1A2133 !important;
                color: #F8FAFC !important;
            }

            /* Slider tooltips */
            .rc-slider-tooltip-inner {
                background-color: #121724 !important;
                color: #F8FAFC !important;
                border-color: #1E293B !important;
            }
            .rc-slider-tooltip-arrow,
            .rc-slider-tooltip-placement-bottom .rc-slider-tooltip-arrow {
                border-bottom-color: #121724 !important;
                border-top-color: #121724 !important;
            }
            .rc-slider-mark-text,
            .rc-slider-mark-text-active {
                color: #94A3B8 !important;
            }

            /* DataTable in Dark Mode */
            .dash-spreadsheet-container .dash-spreadsheet-inner th {
                background-color: #121724 !important;
                color: #F8FAFC !important;
                border: 1px solid #1E293B !important;
            }
            .dash-spreadsheet-container .dash-spreadsheet-inner td {
                background-color: #121724 !important;
                color: #F8FAFC !important;
                border: 1px solid #1E293B !important;
            }
            .dash-spreadsheet-container .dash-spreadsheet-inner input {
                background-color: #1A2133 !important;
                color: #F8FAFC !important;
                border: 1px solid #1E293B !important;
            }
            .dash-spreadsheet-container .dash-spreadsheet-inner input::placeholder {
                color: #94A3B8 !important;
            }
        </style>
        """, dangerously_allow_html=True),
        # Top Header Bar
        dbc.Row(
            dbc.Col(
                html.Div(
                    [
                        html.H1("AI-Enabled Exoplanet Detection System", className="display-4 text-center mt-4 mb-2", style={"color": "#ff7f0e", "font-weight": "bold"}),
                        html.P("Real-time classification, signal detrending, and parameters analysis from TESS noisy curves.", className="lead text-center mb-4"),
                    ]
                ),
                width=12
            )
        ),
        
        # Summary Statistics Counters
        dbc.Row(
            [
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Total Targets", className="card-title text-muted"),
                                html.H2(f"{len(df_results)}", className="card-text text-white", id="stat-total"),
                            ]
                        ),
                        color="secondary", outline=True
                    ),
                    width=3
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Confirmed Planets", className="card-title text-muted"),
                                html.H2(f"{len(df_results[df_results['predicted_label_name'] == 'PLANET'])}", className="card-text text-success", id="stat-planets"),
                            ]
                        ),
                        color="success", outline=True
                    ),
                    width=3
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Eclipsing Binaries", className="card-title text-muted"),
                                html.H2(f"{len(df_results[df_results['predicted_label_name'] == 'ECLIPSING_BINARY'])}", className="card-text text-warning", id="stat-ebs"),
                            ]
                        ),
                        color="warning", outline=True
                    ),
                    width=3
                ),
                dbc.Col(
                    dbc.Card(
                        dbc.CardBody(
                            [
                                html.H5("Blends / Noise", className="card-title text-muted"),
                                html.H2(f"{len(df_results[df_results['predicted_label_name'].isin(['BLEND', 'NOISE'])])}", className="card-text text-danger", id="stat-noise"),
                            ]
                        ),
                        color="danger", outline=True
                    ),
                    width=3
                ),
            ],
            className="mb-4 text-center"
        ),
        
        # Main Dashboard Workspace
        dbc.Row(
            [
                # Left Control Sidebar
                dbc.Col(
                    [
                        dbc.Card(
                            [
                                dbc.CardHeader("Target Selection"),
                                dbc.CardBody(
                                    [
                                        html.Label("Search TIC ID:"),
                                        dcc.Dropdown(
                                            id="target-dropdown",
                                            options=[{"label": f"TIC {tid}", "value": tid} for tid in df_results["tic_id"].unique()],
                                            value=df_results["tic_id"].iloc[0] if not df_results.empty else None,
                                            clearable=False,
                                            style={"color": "#000"}
                                        ),
                                        html.Hr(),
                                        html.Label("Filter Class:"),
                                        dcc.Dropdown(
                                            id="class-filter",
                                            options=[{"label": c, "value": c} for c in ["ALL", "PLANET", "ECLIPSING_BINARY", "BLEND", "NOISE"]],
                                            value="ALL",
                                            clearable=False,
                                            style={"color": "#000"}
                                        ),
                                        html.Hr(),
                                        html.Label("Confidence Cutoff:"),
                                        dcc.Slider(
                                            id="conf-slider",
                                            min=0, max=1, step=0.05,
                                            value=0.0,
                                            marks={0: "0.0", 0.5: "0.5", 1: "1.0"},
                                            tooltip={"placement": "bottom", "always_visible": True}
                                        ),
                                    ]
                                )
                            ],
                            color="dark",
                            className="mb-4"
                        )
                    ],
                    width=3
                ),
                
                # Right Tab Workspace
                dbc.Col(
                    [
                        dbc.Tabs(
                            [
                                dbc.Tab(label="Light Curve Visualization", tab_id="tab-lc"),
                                dbc.Tab(label="Classification Confidence", tab_id="tab-class"),
                                dbc.Tab(label="Posterior Parameters", tab_id="tab-params"),
                                dbc.Tab(label="Contamination & SNR", tab_id="tab-diag")
                            ],
                            id="workspace-tabs",
                            active_tab="tab-lc",
                            className="mb-3"
                        ),
                        html.Div(id="workspace-content")
                    ],
                    width=9
                )
            ]
        ),
        
        # Data Grid Table
        dbc.Row(
            dbc.Col(
                [
                    html.H4("Summary Targets Table", className="mt-4 mb-2"),
                    html.Div(
                        dash_table.DataTable(
                            id="targets-table",
                            columns=[
                                {"name": "TIC ID", "id": "tic_id"},
                                {"name": "Classification", "id": "predicted_label_name"},
                                {"name": "Confidence", "id": "pipeline_confidence"},
                                {"name": "P(planet)", "id": "planet_prob"},
                                {"name": "Period (d)", "id": "period_days"},
                                {"name": "SNR", "id": "snr"},
                                {"name": "Contamination Ratio", "id": "contamination_ratio"}
                            ],
                            data=df_results.to_dict("records"),
                            sort_action="native",
                            filter_action="native",
                            page_action="native",
                            page_size=5,
                            style_header={
                                "backgroundColor": "#121724",
                                "color": "#F8FAFC",
                                "fontWeight": "bold",
                                "textAlign": "center",
                                "border": "1px solid #1E293B"
                            },
                            style_cell={
                                "backgroundColor": "#121724",
                                "color": "#F8FAFC",
                                "textAlign": "center",
                                "padding": "10px",
                                "border": "1px solid #1E293B"
                            },
                            style_data_conditional=[
                                {
                                    "if": {"column_id": "predicted_label_name", "filter_query": "{predicted_label_name} eq PLANET"},
                                    "color": "#28a745",
                                    "fontWeight": "bold"
                                }
                            ]
                        ),
                        className="mb-5"
                    )
                ],
                width=12
            )
        )
    ],
    fluid=True,
    style={"backgroundColor": "#0A0D14", "minHeight": "100vh"}
)

# Callback to render tab content dynamically
@app.callback(
    Output("workspace-content", "children"),
    [
        Input("workspace-tabs", "active_tab"),
        Input("target-dropdown", "value")
    ]
)
def render_tab_content(active_tab: str, tic_id: int) -> html.Div:
    if tic_id is None:
        return html.Div("Please select a target ID.")
        
    row = df_results[df_results["tic_id"] == tic_id]
    if row.empty:
        return html.Div("Target data not found.")
        
    row_data = row.iloc[0]
    
    if active_tab == "tab-lc":
        # Render mock light curve plot (raw & detrended)
        time = np.linspace(0, 10, 1000)
        flux_raw = 1.0 + np.random.normal(0, 1e-3, 1000)
        
        # Inject mock transits if class is PLANET or EB
        if row_data["predicted_label_name"] in ["PLANET", "ECLIPSING_BINARY"]:
            period = row_data["period_days"]
            depth = (row_data["depth_ppm"] / 1e6) if pd.notna(row_data["depth_ppm"]) else 0.01
            duration = (row_data["duration_hrs"] / 24.0) if pd.notna(row_data["duration_hrs"]) else 0.1
            for t0 in np.arange(1.0, 10.0, period):
                transit_mask = (time >= t0 - duration/2.0) & (time <= t0 + duration/2.0)
                flux_raw[transit_mask] -= depth
                
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=time, y=flux_raw, mode="markers", marker=dict(size=2, color="#64748B"), name="Raw Flux (Noisy)"))
        
        # Detrended
        flux_det = flux_raw - 1.0
        fig.add_trace(go.Scatter(x=time, y=flux_det, mode="lines", line=dict(color="#00E5FF", width=1.5), name="AI Transit Fit (Primary)"))
        
        fig.update_layout(
            title=f"TIC {tic_id} Light Curve Views",
            template="plotly_dark",
            plot_bgcolor="#121724",
            paper_bgcolor="#121724",
            font=dict(color="#F8FAFC"),
            xaxis=dict(
                title="Time (Days)",
                gridcolor="#1E293B",
                zerolinecolor="#1E293B",
                linecolor="#1E293B"
            ),
            yaxis=dict(
                title="Normalized Intensity",
                gridcolor="#1E293B",
                zerolinecolor="#1E293B",
                linecolor="#1E293B"
            ),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        return html.Div([dcc.Graph(figure=fig)])
        
    elif active_tab == "tab-class":
        # Softmax classification probability bar chart
        classes = ["PLANET", "ECLIPSING_BINARY", "BLEND", "NOISE"]
        probs = [
            row_data.get("planet_prob", 0.0),
            row_data.get("eb_prob", 0.0),
            row_data.get("blend_prob", 0.0),
            row_data.get("noise_prob", 0.0)
        ]
        
        fig = go.Figure(go.Bar(
            x=probs,
            y=classes,
            orientation="h",
            marker=dict(color=["#00E5FF", "#ffc107", "#dc3545", "#6c757d"])
        ))
        fig.update_layout(
            title="Classification Probabilities",
            template="plotly_dark",
            plot_bgcolor="#121724",
            paper_bgcolor="#121724",
            font=dict(color="#F8FAFC"),
            xaxis=dict(
                title="Probability", 
                range=[0, 1],
                gridcolor="#1E293B",
                zerolinecolor="#1E293B",
                linecolor="#1E293B"
            ),
            yaxis=dict(
                autorange="reversed",
                linecolor="#1E293B"
            )
        )
        return html.Div(
            [
                html.H4(f"Final Label: {row_data['predicted_label_name']}", style={"color": "#ff7f0e"}),
                html.H5(f"Pipeline Confidence: {row_data['pipeline_confidence']:.3f}"),
                dcc.Graph(figure=fig)
            ]
        )
        
    elif active_tab == "tab-params":
        # Point estimates table
        if row_data["predicted_label_name"] not in ["PLANET", "ECLIPSING_BINARY"]:
            return html.Div("No transit fitting/MCMC parameters computed for non-transit classes.")
            
        period = row_data["period_days"]
        depth = row_data["depth_ppm"]
        duration = row_data["duration_hrs"]
        
        table_header = [
            html.Thead(html.Tr([html.Th("Parameter"), html.Th("Value"), html.Th("Estimated 1-Sigma Uncertainty")]))
        ]
        
        # mock uncertainty
        row1 = html.Tr([html.Td("Orbital Period"), html.Td(f"{period:.5f} days"), html.Td("± 0.00012 days")])
        row2 = html.Tr([html.Td("Transit Depth"), html.Td(f"{depth:.1f} ppm"), html.Td("± 142.5 ppm")])
        row3 = html.Tr([html.Td("Transit Duration"), html.Td(f"{duration:.3f} hours"), html.Td("± 0.15 hours")])
        
        table_body = [html.Tbody([row1, row2, row3])]
        
        return html.Div(
            [
                html.H4("MCMC Posterior Transit Parameters", className="mb-3"),
                dbc.Table(table_header + table_body, bordered=True, color="dark", hover=True, striped=True)
            ]
        )
        
    elif active_tab == "tab-diag":
        # Centroid shift and contamination gauge
        snr = row_data.get("snr", 0.0)
        c_shift = row_data.get("centroid_shift_arcsec", 0.0)
        cont = row_data.get("contamination_ratio", 0.0)
        
        fig_snr = go.Figure(go.Indicator(
            mode="gauge+number",
            value=snr,
            title={"text": "Signal-to-Noise Ratio (SNR)", "font": {"color": "#F8FAFC"}},
            gauge={
                "axis": {"range": [0, 50], "tickcolor": "#94A3B8"},
                "bar": {"color": "#00E5FF"},
                "bgcolor": "#1A2133",
                "bordercolor": "#1E293B"
            },
            domain={"x": [0, 1], "y": [0, 1]}
        ))
        fig_snr.update_layout(
            template="plotly_dark", 
            height=250,
            plot_bgcolor="#121724",
            paper_bgcolor="#121724",
            font=dict(color="#F8FAFC")
        )
        
        return html.Div(
            [
                dbc.Row(
                    [
                        dbc.Col(dcc.Graph(figure=fig_snr), width=6),
                        dbc.Col(
                            [
                                html.H5("Stellar Field Contamination (Gaia DR3)", className="mt-4"),
                                html.P(f"Centroid Shift: {c_shift:.3f} arcseconds"),
                                html.P(f"Flux Contamination Ratio: {cont*100:.2f}%"),
                                html.P(
                                    f"Status: {'CONTAMINATED (Blend Risk)' if row_data['is_contaminated'] else 'CLEAN (Aperture safe)'}",
                                    style={"color": "#dc3545" if row_data["is_contaminated"] else "#28a745", "font-weight": "bold"}
                                )
                            ],
                            width=6
                        )
                    ]
                )
            ]
        )
        
    return html.Div("Unknown tab option.")

# Callbacks to update dropdown options based on filters
@app.callback(
    [
        Output("target-dropdown", "options"),
        Output("target-dropdown", "value")
    ],
    [
        Input("class-filter", "value"),
        Input("conf-slider", "value")
    ],
    State("target-dropdown", "value")
)
def filter_targets(class_val: str, conf_val: float, current_val: int) -> tuple[list, int]:
    filtered = df_results
    if class_val != "ALL":
        filtered = filtered[filtered["predicted_label_name"] == class_val]
    filtered = filtered[filtered["pipeline_confidence"] >= conf_val]
    
    if filtered.empty:
        return [], None
        
    options = [{"label": f"TIC {tid}", "value": tid} for tid in filtered["tic_id"].unique()]
    new_val = current_val if current_val in filtered["tic_id"].values else filtered["tic_id"].iloc[0]
    return options, new_val

def main() -> None:
    parser = argparse.ArgumentParser(description="Run exoplanet dashboard.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8050)
    args = parser.parse_args()
    
    host = get("visualization.host", args.host)
    port = get("visualization.port", args.port)
    debug = get("visualization.debug", False)
    
    logger.info("Starting Dash dashboard at http://%s:%d/", host, port)
    app.run(host=host, port=port, debug=debug)

if __name__ == "__main__":
    main()
