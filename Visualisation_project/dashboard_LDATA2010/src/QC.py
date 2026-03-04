###########################################################################
# IMPORTS
###########################################################################

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shiny import ui
from color_plot_config import OKABE_ITO_COLORS


def get_quality_metrics(df):
    df = df.replace([np.nan, None, '', 'NA', 'N/A', 'na', 'n/a'], np.nan)
    return {
        'n_duplicates': df.duplicated().sum(),
        'pct_missing': (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100 # porxi (don't take everything into account, but visual does)
    }


def plot_missingness(df, drop_no_missing_cols=False):
    df = df.replace([np.nan, None, '', 'NA', 'N/A', 'na', 'n/a'], np.nan)
    missing = df.isnull().astype(int)

    if drop_no_missing_cols:
        missing = missing.loc[:, missing.sum(axis=0) > 0]

    var_order = missing.sum(axis=0).sort_values(ascending=False).index
    patient_order = missing.sum(axis=1).sort_values(ascending=False).index
    missing = missing.loc[patient_order, var_order]

    missing_per_patient = missing.sum(axis=1)
    missing_per_var = missing.sum(axis=0)

    # color-blind friendly
    col_missing = OKABE_ITO_COLORS[1]
    col_present = OKABE_ITO_COLORS[0]

    fig = make_subplots(
        rows=2, cols=2,
        row_heights=[0.18, 0.82],
        column_widths=[0.82, 0.18],
        specs=[[{"type": "bar"}, None],
               [{"type": "heatmap"}, {"type": "bar"}]]
    )

    fig.add_trace(
        go.Bar(
            x=missing.index.astype(str),
            y=missing_per_patient.values,
            marker_color=col_missing,
            showlegend=False
        ),
        row=1, col=1
    )
    fig.update_xaxes(showticklabels=False, row=1, col=1)

    fig.add_trace(
        go.Heatmap(
            z=missing.T.values,
            x=missing.index.astype(str),
            y=missing.columns.astype(str),
            colorscale=[[0, col_present], [1, col_missing]], 
            showscale=False
        ),
        row=2, col=1
    )
    fig.update_yaxes(autorange="reversed", row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=2, col=1)

    fig.add_trace(
        go.Bar(
            x=missing_per_var.values,
            y=missing_per_var.index.astype(str),
            orientation='h',
            marker_color=col_missing,        
            showlegend=False
        ),
        row=2, col=2
    )
    fig.update_yaxes(matches='y2', row=2, col=2)
    fig.update_yaxes(autorange="reversed", row=2, col=2)
    fig.update_yaxes(showticklabels=False, row=2, col=2)

    fig.update_layout(
            template='plotly_white'
        )

    return fig

def metric_card(label, output_id):
    return ui.div(
        ui.span(label, class_="text-muted", style="font-size: 0.8rem;"),
        ui.span(ui.output_text(output_id),
                class_="fw-bold",
                style="font-size: 1.5rem;"),
        style=(
            "border: 1px solid #ddd; "
            "border-radius: 6px; "
            "padding: 10px 12px; "
            "background: white; "
        )
    )

