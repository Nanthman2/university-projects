###########################################################################
# IMPORTS
###########################################################################


import plotly.express as px
import pandas as pd

from color_plot_config import OKABE_ITO_COLORS # colorblind-friendly color palette
from utils import format_label

# To determin dynamically the chart to choose

def is_numeric(df, column):
    return pd.api.types.is_numeric_dtype(df[column])


# Plot func

def plot_bivariate(df, x_var, y_var, color_by=None):
    columns = [x_var, y_var]
    if color_by and color_by != 'none':
        columns.append(color_by)
    
    plot_df = df[columns].copy()

    # CASE 1: Both numeric = SCATTER

    if is_numeric(df, x_var) and is_numeric(df, y_var):
        fig = px.scatter(
            plot_df,
            x=x_var,
            y=y_var,
            color=color_by if color_by != 'none' else None,
            opacity=0.6,
            color_discrete_sequence=OKABE_ITO_COLORS,
            labels={
                x_var: format_label(x_var),
                y_var: format_label(y_var),
                color_by: format_label(color_by) if color_by and color_by != 'none' else None
            }
        )
            
        fig.update_traces(marker=dict(size=5))
        fig.update_layout(template='plotly_white')
        return fig
    
    # CASE 2: Both categorical = STACKED BARPLOT

    elif not is_numeric(df, x_var) and not is_numeric(df, y_var):
        # Count combinations
        counts = plot_df.groupby([x_var, y_var]).size().reset_index(name='count')
    
        fig = px.histogram(
            counts,
            x=x_var,
            y='count',
            color=y_var,
            barnorm='percent',  # Stacked bar chart normalized to 100%
            color_discrete_sequence=OKABE_ITO_COLORS,
            labels={
                x_var: format_label(x_var),
                y_var: format_label(y_var)
            }
        )
        fig.update_layout(
            template='plotly_white',
            yaxis_title='Percentage (%)'
        )
        return fig
    
    # CASE 3: One numeric, one categorical = BOXPLOT
    
    else:
        # Determine which is which
        if is_numeric(df, x_var) and not is_numeric(df, y_var):
            # Swap so categorical is on x-axis
            x_var, y_var = y_var, x_var
        
        fig = px.box(
            plot_df,
            x=x_var,
            y=y_var,
            color=color_by if color_by != 'none' else None,
            points='outliers',
            color_discrete_sequence=OKABE_ITO_COLORS,
            labels={
                x_var: format_label(x_var),
                y_var: format_label(y_var),
                color_by: format_label(color_by) if color_by and color_by != 'none' else None
            }
        )
        
        fig.update_layout(template='plotly_white')
        return fig