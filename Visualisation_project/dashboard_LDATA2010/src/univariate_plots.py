###########################################################################
# IMPORTS
###########################################################################

import pandas as pd
import plotly.express as px
from color_plot_config import OKABE_ITO_COLORS
from utils import format_label

# Utils 
def detect_variable_type(df, column):
    if pd.api.types.is_numeric_dtype(df[column]):
        n_unique = df[column].nunique()
        if n_unique <= 10: # "security"
            return 'categorical'
        return 'numerical'
    return 'categorical'

# Plot functions
def plot_univariate(df, variable, group_by=None, figsize=(8, 5)):
    """
    Plot depending on the variable type and the (optional) group by variable.
    """
    var_type = detect_variable_type(df, variable)
    
    if var_type == 'numerical':
        if group_by and group_by != 'none':
            return plot_numerical_grouped(df, variable, group_by)
        else:
            return plot_numerical_simple(df, variable)
    else:
        if group_by and group_by != 'none':
            return plot_categorical_grouped(df, variable, group_by)
        else:
            return plot_categorical_simple(df, variable)


# Sub functions

def plot_numerical_simple(df, variable):
    # security :
    data = df[variable].dropna()
    
    # Create histogram
    fig = px.histogram(
        data,
        x=variable,
        title=format_label(variable),
        labels={variable: format_label(variable)},
        opacity=0.7,
        color_discrete_sequence=[OKABE_ITO_COLORS[0]]
    )
    
    fig.update_layout(
        title_x=0.5,
        template='plotly_white',
        yaxis_title='Count'
    )
    
    return fig


def plot_numerical_grouped(df, variable, group_by):
    # security :
    data = df[[variable, group_by]].dropna()
    
    fig = px.histogram(
        data,
        x=variable,
        color=group_by,
        barmode='overlay',
        opacity=0.6,
        color_discrete_sequence=OKABE_ITO_COLORS  
    )
    
    fig.update_layout(
        title=f'{format_label(variable)} by {format_label(group_by)}',
        xaxis_title=format_label(variable),
        title_x=0.5,
        yaxis_title='Count',
        template='plotly_white',
        legend_title=format_label(group_by)
    )

    return fig


def plot_categorical_simple(df, variable):
    # Compute percentages (for additional display)
    value_counts = df[variable].value_counts()
    total = value_counts.sum()
    percentages = (100 * value_counts / total).round(1)
    
    # Create df
    plot_df = pd.DataFrame({
        variable: value_counts.index,
        'Count': value_counts.values,
        'Percentage': percentages
    })
    
    # Bar plot 
    fig = px.bar(
        plot_df,
        x=variable,
        y='Count',
        text='Percentage',
        color_discrete_sequence=[OKABE_ITO_COLORS[0]],  
        title=format_label(variable),
        labels={variable: format_label(variable)},
    )

    fig.update_layout(
        title_x=0.5,
        template='plotly_white',
        showlegend=False
    )
    
    return fig


def plot_categorical_grouped(df, variable, group_by):
    """Plot distribution of categorical variable grouped by another category."""
    data = df[[variable, group_by]].dropna()
    
    # Calculate counts
    ct = pd.crosstab(data[variable], data[group_by])
    
    # Transform crosstab to long format pour px.bar
    ct_long = ct.reset_index().melt(id_vars=variable, var_name=group_by, value_name='Count')
    
    fig = px.bar(
        ct_long,
        x=variable,
        y='Count',
        color=group_by,
        barmode='group',
        text='Count',
        color_discrete_sequence=OKABE_ITO_COLORS,  
        title=f"{format_label(variable)} by {format_label(group_by)}",
        labels={variable: format_label(variable),
                group_by: format_label(group_by),
                'Count': 'Count'},
        height=450
    )
    
    fig.update_layout(
        template='plotly_white',
        legend_title=format_label(group_by)
    )
    
    return fig