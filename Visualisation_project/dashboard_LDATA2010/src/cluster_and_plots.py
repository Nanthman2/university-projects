import pandas as pd
import numpy as np

from color_plot_config import OKABE_ITO_COLORS
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

from pandas.api.types import is_numeric_dtype

from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.neighbors import NearestNeighbors


def plot_cluster_map(embedding, embedding_cols, labels, metadata_df):
    #Scatter plot
    plot_df = pd.DataFrame(embedding, columns=embedding_cols)
    plot_df['Cluster'] = labels.astype(str)
    
    hover_cols = []
    if 'patient_id' in metadata_df.columns:
        plot_df['Patient ID'] = metadata_df['patient_id'].values
        hover_cols.append('Patient ID')
    if 'pam50_+_claudin-low_subtype' in metadata_df.columns:
        plot_df['Subtype'] = metadata_df['pam50_+_claudin-low_subtype'].values
        hover_cols.append('Subtype')

    n_dims = len(embedding_cols)

    if n_dims == 3:
        fig = px.scatter_3d(
            plot_df, x=embedding_cols[0], y=embedding_cols[1], z=embedding_cols[2],
            color='Cluster', hover_data=hover_cols, template='plotly_white', 
            color_discrete_sequence=OKABE_ITO_COLORS 
        )
        fig.update_traces(marker=dict(size=3))
    else:
        fig = px.scatter(
            plot_df, x=embedding_cols[0], y=embedding_cols[1],
            color='Cluster', hover_data=hover_cols, template='plotly_white', 
            color_discrete_sequence=OKABE_ITO_COLORS 
        )
        fig.update_traces(marker=dict(size=6, opacity=0.8))
    
    return fig

def plot_elbow_curve(inertias, current_k):
    #Elbow plot
    ks = list(range(2, 2 + len(inertias)))
    fig = px.line(
        x=ks, y=inertias, markers=True, title='Elbow Method',
        labels={'x': 'k', 'y': 'Inertia'}, template='plotly_white', 
        color_discrete_sequence=OKABE_ITO_COLORS 
    )
    
    current_inertia = inertias[current_k - 2]
    
    marker_color = OKABE_ITO_COLORS[5] if len(OKABE_ITO_COLORS) > 5 else 'red'
    
    fig.add_trace(go.Scatter(
        x=[current_k], y=[current_inertia], mode='markers',
        marker=dict(color=marker_color, size=12, symbol='x'), name=f'Selected k={current_k}'
    ))
    return fig

def plot_cluster_univariate(df_subset, labels, var_main, var_sub, format_label_func=str):
    #Plot univariate (the clusters )
    df_plot = df_subset.copy()
    df_plot['Cluster'] = "Cluster " + labels.astype(str)
    df_plot = df_plot.sort_values('Cluster')
    has_subgroup = (var_sub is not None) and (var_sub != "none")
    
    if is_numeric_dtype(df_plot[var_main]):
        fig = px.box(
            df_plot, x="Cluster", y=var_main, color=var_sub if has_subgroup else "Cluster",
            template='plotly_white',
            color_discrete_sequence=OKABE_ITO_COLORS 
        )
        if has_subgroup: fig.update_layout(boxmode='group')
    else:
        fig = px.histogram(
            df_plot, x="Cluster", color=var_main, facet_col=var_sub if has_subgroup else None,
            facet_col_wrap=2, barmode='group', 
            template='plotly_white', text_auto=True,
            color_discrete_sequence=OKABE_ITO_COLORS 
        )
        if has_subgroup: fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))    
    return fig

def plot_dendrogram(X, method, metric, orientation='bottom'):
    #Dendogramme
    if method in ['ward', 'centroid']: 
        metric = 'euclidean'
    
    def custom_linkage(x):
        return linkage(x, method=method, metric=metric)
    
    fig = ff.create_dendrogram(
        X, 
        orientation=orientation,
        linkagefun=custom_linkage
    )
    
    fig.update_layout(
        title='', 
        template='plotly_white',
        autosize=True,
        margin=dict(l=20, r=20, t=20, b=20),
        showlegend=False
    )
    
    if orientation == 'bottom':
        fig.update_layout(
            xaxis_title="Patients (Leaves)", 
            yaxis_title="Distance"
        )
        fig.update_xaxes(showticklabels=False)
    else:
        fig.update_layout(
            xaxis_title="Distance", 
            yaxis_title="Patients (Leaves)"
        )
        fig.update_yaxes(showticklabels=False)
    
    return fig

def plot_k_distance(X, k):
    #K-distance graph for DBSCAN
    neigh = NearestNeighbors(n_neighbors=k)
    nbrs = neigh.fit(X)
    distances, _ = nbrs.kneighbors(X)
    k_distances = np.sort(distances[:, k-1])
    
    fig = px.line(
        x=range(len(k_distances)), y=k_distances,
        labels={"x": "Points (sorted by distance)", "y": f"Distance to {k}-th NN"},
        template='plotly_white',
        color_discrete_sequence=OKABE_ITO_COLORS 
    )
    fig.add_annotation(text="Look for the 'knee' to set Epsilon", xref="paper", yref="paper", x=0.1, y=0.9, showarrow=False, font=dict(color="gray", size=12))
    return fig

def plot_kmeans_metrics(ks, ch_scores, gap_scores, current_k, best_ch_k=None, best_gap_k=None):
    # Colors from OKABE_ITO (color friendly)
    col_blue = OKABE_ITO_COLORS[4]
    col_green = OKABE_ITO_COLORS[2]
    col_red = OKABE_ITO_COLORS[5]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(go.Scatter(x=ks, y=ch_scores, name="CH Score", mode='lines+markers', marker=dict(color=col_blue, size=6), line=dict(color=col_blue, width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=ks, y=gap_scores, name="Gap Stat", mode='lines+markers', marker=dict(color=col_green, size=6), line=dict(color=col_green, width=2)), secondary_y=True)

    if best_ch_k in ks:
        idx = ks.index(best_ch_k)
        fig.add_trace(go.Scatter(x=[best_ch_k], y=[ch_scores[idx]], mode='markers', marker=dict(color=col_blue, size=14, symbol='star'), name="Best CH"), secondary_y=False)
    if best_gap_k in ks:
        idx = ks.index(best_gap_k)
        fig.add_trace(go.Scatter(x=[best_gap_k], y=[gap_scores[idx]], mode='markers', marker=dict(color=col_green, size=14, symbol='star'), name="Best Gap"), secondary_y=True)

    if current_k in ks:
        idx = ks.index(current_k)
        red_cross = dict(color=col_red, size=10, symbol='x', line=dict(color=col_red, width=1))
        fig.add_trace(go.Scatter(x=[current_k], y=[ch_scores[idx]], mode='markers', marker=red_cross, showlegend=False), secondary_y=False)
        fig.add_trace(go.Scatter(x=[current_k], y=[gap_scores[idx]], mode='markers', marker=red_cross, showlegend=False), secondary_y=True)

    fig.update_layout(template='plotly_white', legend=dict(orientation="h", y=1.1, x=0.5, xanchor="center"), hovermode="x unified", margin=dict(t=10, l=10, r=10, b=10))
    fig.update_xaxes(title_text="Clusters (k)", tickmode='linear')
    fig.update_yaxes(title_text="CH", title_font=dict(color=col_blue), secondary_y=False)
    fig.update_yaxes(title_text="Gap", title_font=dict(color=col_green), secondary_y=True)
    return fig

def plot_hc_metrics(ks, ch_scores, gap_scores, current_k, best_ch_k=None, best_gap_k=None):
    
    col_blue = OKABE_ITO_COLORS[4]
    col_green = OKABE_ITO_COLORS[2]
    col_red = OKABE_ITO_COLORS[5]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Scatter(x=ks, y=ch_scores, name="CH Score", mode='lines+markers', marker=dict(color=col_blue, size=6), line=dict(color=col_blue, width=2)), secondary_y=False)
    fig.add_trace(go.Scatter(x=ks, y=gap_scores, name="Gap Stat", mode='lines+markers', marker=dict(color=col_green, size=6), line=dict(color=col_green, width=2)), secondary_y=True)

    if best_ch_k in ks:
        idx = ks.index(best_ch_k)
        fig.add_trace(go.Scatter(x=[best_ch_k], y=[ch_scores[idx]], mode='markers', marker=dict(color=col_blue, size=14, symbol='star'), name="Best CH"), secondary_y=False)
    if best_gap_k in ks:
        idx = ks.index(best_gap_k)
        fig.add_trace(go.Scatter(x=[best_gap_k], y=[gap_scores[idx]], mode='markers', marker=dict(color=col_green, size=14, symbol='star'), name="Best Gap"), secondary_y=True)

    if current_k in ks:
        idx = ks.index(current_k)
        red_cross = dict(color=col_red, size=10, symbol='x', line=dict(color=col_red, width=1))
        fig.add_trace(go.Scatter(x=[current_k], y=[ch_scores[idx]], mode='markers', marker=red_cross, showlegend=False), secondary_y=False)
        fig.add_trace(go.Scatter(x=[current_k], y=[gap_scores[idx]], mode='markers', marker=red_cross, showlegend=False), secondary_y=True)

    fig.update_layout(template='plotly_white', legend=dict(orientation="v", y=0.5, x=1.2, xanchor="left"), hovermode="x unified", margin=dict(t=10, l=10, r=10, b=10))
    fig.update_xaxes(title_text="Clusters (k)", tickmode='linear')
    fig.update_yaxes(title_text="CH", title_font=dict(color=col_blue), secondary_y=False)
    fig.update_yaxes(title_text="Gap", title_font=dict(color=col_green), secondary_y=True)
    return fig