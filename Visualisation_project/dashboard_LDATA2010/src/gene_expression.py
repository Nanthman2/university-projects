###########################################################################
# IMPORTS
###########################################################################

import numpy as np
import pandas as pd
import plotly.express as px
from scipy.stats import ttest_ind

def create_volcano_plot(df, gene_vars, group_var, fc_thresh=1.0, p_thresh=0.05):
    df[group_var] = df[group_var].fillna(0)
    groups = df[group_var].unique()

    group_A = df[df[group_var] == groups[0]]
    group_B = df[df[group_var] == groups[1]]
    
    if len(group_A) < 3 or len(group_B) < 3:
        raise ValueError(f"Not enough samples: {len(group_A)} vs {len(group_B)}") 
    
    results = []
    
    for gene in gene_vars:
        # Z-scores
        mean_A = group_A[gene].mean()
        mean_B = group_B[gene].mean()
        
        log2_fc = mean_A - mean_B 
        
        # T-test
        t_stat, p_value = ttest_ind(
            group_A[gene].dropna(), 
            group_B[gene].dropna(), 
            equal_var=False
        )
        
        results.append({
            "gene": gene,
            "log2_fold_change": log2_fc,
            "p_value": p_value,
            "neg_log10_p": -np.log10(p_value)
        })
    
    df_volcano = pd.DataFrame(results)
    
    # Classify significance
    df_volcano["significance"] = "Not significant"
    df_volcano.loc[
        (df_volcano["log2_fold_change"] >= fc_thresh) & (df_volcano["p_value"] < p_thresh),
        "significance"
    ] = "Up"
    df_volcano.loc[
        (df_volcano["log2_fold_change"] <= -fc_thresh) & (df_volcano["p_value"] < p_thresh),
        "significance"
    ] = "Down"

    # Plot
    fig = px.scatter(
        df_volcano,
        x="log2_fold_change",
        y="neg_log10_p",
        color="significance",
        hover_data={
            "gene": True,
            "p_value": ":.2e",
            "log2_fold_change": ":.2f"
        }
    )
    
    # Threshold lines
    fig.add_hline(
        y=-np.log10(p_thresh),
        line_dash="dash",
        line_color="gray",
        annotation_text=f"p = {p_thresh}"
    )
    fig.add_vline(x=fc_thresh, line_dash="dash", line_color="gray")
    fig.add_vline(x=-fc_thresh, line_dash="dash", line_color="gray")
    
    fig.update_layout(
        template='plotly_white',
        xaxis_title="Log2 Fold-Change",
        yaxis_title="-Log10(p-value)"
    )
    
    return fig
