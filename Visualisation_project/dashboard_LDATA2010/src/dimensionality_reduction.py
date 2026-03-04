###########################################################################
# IMPORTS
###########################################################################

# Standard imports 
import numpy as np
import pandas as pd

# Intrinsic dimensionality
import skdim

# DR Techniques
from sklearn.decomposition import PCA
import umap.umap_ as umap

# Quality Assessment
from scipy.integrate import trapezoid
from sklearn.metrics.pairwise import pairwise_distances

# Ploting
import plotly.express as px

from color_plot_config import OKABE_ITO_COLORS

###########################################################################
# DIMENSIONALITY REDUCTION
###########################################################################

# to extract (a subset) of the patient genomic prolie

def prepare_feature_matrix(df, gene_cols, sample_fraction=1.0, random_state=2003):
    X_full = df[gene_cols].values
    
    # deterministic split depanding on the  user-selected %data
    if sample_fraction < 1.0:
        n_samples = int(len(df) * sample_fraction)
        np.random.seed(random_state)
        indices = np.random.choice(len(df), n_samples, replace=False)
        X = X_full[indices]
    else:
        X = X_full
        indices = np.arange(len(df))
    
    return X, indices, len(gene_cols)

# DR

def compute_umap(X, n_components, n_neighbors, min_dist, metric='euclidean', random_state=2003):
    model = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state
    )
    return model.fit_transform(X)


def compute_pca(X, n_components=20):
    model = PCA(n_components=n_components)
    embedding = model.fit_transform(X)
    return embedding, model.explained_variance_ratio_

# Store results for plot

def create_projection_df(embedding, df_original, sample_indices, color_var=None, method="UMAP"):
    n_dims = embedding.shape[1]
    
    if n_dims == 2:
        df_plot = pd.DataFrame({
            f'{method}_1': embedding[:, 0],
            f'{method}_2': embedding[:, 1]
        })
    else:
        df_plot = pd.DataFrame({
            f'{method}_1': embedding[:, 0],
            f'{method}_2': embedding[:, 1],
            f'{method}_3': embedding[:, 2]
        })
    
    if color_var:
        df_plot[color_var] = df_original.iloc[sample_indices][color_var].values
    
    return df_plot


###########################################################################
# INTRINSIC DIMENSIONALITY 
# Code taken form : https://huggingface.co/blog/AmelieSchreiber/intrinsic-dimension-of-proteins
###########################################################################

methods = {
    "Fisher Separability": skdim.id.FisherS,
    "Exponential Correlation": skdim.id.CorrInt
}

def estimate_all_dimensions(X):
    results = []
    for method_name, method_class in methods.items():
        try:
            id_est = method_class().fit(X)
            dimension = id_est.dimension_
            results.append({'Method': method_name, 'Dimension': round(dimension, 2)})
        except Exception:
            results.append({'Method': method_name, 'Dimension': np.nan})
    return pd.DataFrame(results)


###########################################################################
# CO-RANKING MATRIX
# Code taken from pyDRMetrics package github (https://github.com/zhangys11/pyDRMetrics)
###########################################################################

def ranking_matrix(D, solver='fast'):
    D = np.array(D)
    
    if solver == 'fast':
        R = [np.argsort(np.argsort(row)) for row in D]
    else:
        R = np.zeros(D.shape)
        m = len(R)
        for i in range(m):
            for j in range(m):
                Rij = 0
                for k in range(m):
                    if (D[i,k] < D[i,j]) or (np.isclose(D[i,k], D[i,j]) and k < j):
                        Rij += 1
                R[i,j] = Rij
    
    return np.array(R, dtype='uint')


def coranking_matrix(R1, R2):
    R1 = np.array(R1)
    R2 = np.array(R2)
    assert R1.shape == R2.shape
    
    Q = np.zeros(R1.shape)
    m = len(Q)
    
    for i in range(m):
        for j in range(m):
            k = int(R1[i,j])
            l = int(R2[i,j])
            Q[k,l] += 1
    
    return Q

###########################################################################
# Quality Metrics
###########################################################################

# Benchmark :
def compute_trustworthiness(Q, K):
    """
    Inspired from : pyDRMetrics .
    """
    Q = Q[1:, 1:]  
    m = len(Q)
    
    k = K - 1
    Qs = Q[k:, :k]
    W = np.arange(Qs.shape[0]).reshape(-1, 1)  
    T = 1 - np.sum(Qs * W) / (k + 1) / m / (m - 1 - k)
    
    return T


def compute_continuity(Q, K):
    """
    Inspired from : pyDRMetrics .
    """
    Q = Q[1:, 1:] 
    m = len(Q)
    
    k = K - 1
    
    Qs = Q[:k, k:]
    W = np.arange(Qs.shape[1]).reshape(1, -1)  
    C = 1 - np.sum(Qs * W) / (k + 1) / m / (m - 1 - k)
    
    return C


# Metrics from Lee & Verleysen framework


def compute_QNX(Q, K):
    """Quality"""
    Q_subset = Q[1:, 1:]  
    N = Q_subset.shape[0]
    
    qnx = np.sum(Q_subset[:K, :K]) / (K * N)
    
    return qnx

def compute_RandQNX(Q, K):
    """Q_NX of a random embedding"""
    N = Q[1:, 1:].shape[0]
    return K / (N - 1)

def compute_RNX(Q, K):
    """Standardized Quality"""
    N = Q[1:, 1:].shape[0]
    qnx = compute_QNX(Q, K)
    
    rnx = ((N - 1) * qnx - K) / (N - 1 - K)
    
    return rnx


def compute_BNX(Q, K):
    """Behavior"""
    Q_subset = Q[1:, 1:]
    N = Q_subset.shape[0]
    
    UN = np.sum(Q_subset[K:2*K, :K]) / (K * N)
    UX = np.sum(Q_subset[:K, K:2*K]) / (K * N)
    
    bnx = UN - UX
    
    return bnx

# Compute for a number of neighbors K 

def compute_all_metrics_from_Q(Q, K_values):
    results = {'K': [],'T': [],'C': [],'Random': [],'B_NX': [],'R_NX': []}
    for K in range(1,K_values+1):
        results['K'].append(K)
        results['T'].append(compute_trustworthiness(Q, K))  
        results['C'].append(compute_continuity(Q, K))
        results['Random'].append(compute_RandQNX(Q, K))
        results['B_NX'].append(compute_BNX(Q, K))
        results['R_NX'].append(compute_RNX(Q, K))
    
    return results


# Store for plotting

def compute_neighborhood_preservation(X_high, X_low_umap, X_low_pca, K_values=None):
    """
    Args :
        X_high : Original high-dimensional data
        X_low_umap : UMAP embedding
        X_low_pca : PCA embedding
        K_values : K values to evaluate
        
    Returns :
        dict
            - 'metrics_df': [K, Metric, Method, Value]
            - 'Q_umap': Co-ranking matrix for UMAP
            - 'Q_pca': Co-ranking matrix for PCA (if provided)
    """
    N = X_high.shape[0]
    
    metrics_data = []
    
    # UMAP METRICS
    
    # Build co-ranking matrix
    D_high = pairwise_distances(X_high)
    D_umap = pairwise_distances(X_low_umap)
    R_high = ranking_matrix(D_high)
    R_umap = ranking_matrix(D_umap)
    Q_umap = coranking_matrix(R_high, R_umap)
    
    # Compute metrics from Q
    K_max = min(K_values, N - 2)  
    q_metrics = compute_all_metrics_from_Q(Q_umap, K_max)
    
    for K in range(1,K_max):

        metrics_data.append({'K': K, 'Metric': 'Trustworthiness', 'Method': 'UMAP', 'Value': q_metrics['T'][K-1]})
        metrics_data.append({'K': K, 'Metric': 'Continuity', 'Method': 'UMAP', 'Value': q_metrics['C'][K-1]})
        metrics_data.append({'K': K, 'Metric': 'Random', 'Method': '', 'Value': q_metrics['Random'][K-1]})
        metrics_data.append({'K': K, 'Metric': 'B_NX', 'Method': 'UMAP', 'Value': q_metrics['B_NX'][K-1]})
        metrics_data.append({'K': K, 'Metric': 'R_NX', 'Method': 'UMAP', 'Value': q_metrics['R_NX'][K-1]})
    
    # PCA METRICS
    
    Q_pca = None
    if X_low_pca is not None:
        D_pca = pairwise_distances(X_low_pca)
        R_pca = ranking_matrix(D_pca)
        Q_pca = coranking_matrix(R_high, R_pca)
        
        q_metrics_pca = compute_all_metrics_from_Q(Q_pca, K_values)
        
        for K in range(1,K_max):

            metrics_data.append({'K': K, 'Metric': 'Trustworthiness', 'Method': 'PCA', 'Value': q_metrics['T'][K-1]})
            metrics_data.append({'K': K, 'Metric': 'Continuity', 'Method': 'PCA', 'Value': q_metrics['C'][K-1]})
            #metrics_data.append({'K': K, 'Metric': 'Random', 'Method': 'PCA', 'Value': q_metrics_pca['Random'][K-1]})
            metrics_data.append({'K': K, 'Metric': 'B_NX', 'Method': 'PCA', 'Value': q_metrics_pca['B_NX'][K-1]})
            metrics_data.append({'K': K, 'Metric': 'R_NX', 'Method': 'PCA', 'Value': q_metrics_pca['R_NX'][K-1]})
    
    return {'metrics_df': pd.DataFrame(metrics_data), 'Q_umap': Q_umap, 'Q_pca':Q_pca}

# Summarize into a scalar using the area under the curve (AUC)

def compute_auc_metric(metrics_df, metric_name='R_NX', method='UMAP'):
    """Compute normalized AUC for a given metric and method"""
    
    # Filter by method AND metric
    df_filtered = metrics_df[
        (metrics_df['Method'] == method) & 
        (metrics_df['Metric'] == metric_name)
    ].copy()
    
    if df_filtered.empty:
        return np.nan
    
    # Sort by K
    df_filtered = df_filtered.sort_values('K')
    
    # Compute AUC using the 'Value' column
    auc_raw = trapezoid(y=df_filtered['Value'], x=df_filtered['K'])
    
    # Normalize by K range to get value between 0 and 1
    K_max = df_filtered['K'].max()
    K_min = df_filtered['K'].min()
    auc_normalized = auc_raw / (K_max - K_min)
    
    return auc_normalized

############################################################################
# PLOTTING
############################################################################

def plot_quality_metrics(df_preservation):
    fig = px.line(
        df_preservation,
        x='K',
        y='Value',
        color='Metric',
        line_dash='Method',
        color_discrete_sequence=OKABE_ITO_COLORS
    )
    
    fig.add_hline(y=0, line_dash='dot', line_color='gray', opacity=0.5)

    fig.update_layout(
        template='plotly_white',
        xaxis_title="K (# neighbors)",
        yaxis_title="Quality Metric",
    )
    
    return fig


def plot_scree(variance_explained):
    n_components = len(variance_explained)
    cumsum = np.cumsum(variance_explained)
    
    df_variance = pd.DataFrame({
        'Component': list(range(1, n_components + 1)),
        'Individual': variance_explained,
        'Cumulative': cumsum
    })

    fig = px.bar(
        df_variance,
        x='Component',
        y='Individual',
    )
    
    fig.add_scatter(
        x=df_variance['Component'],
        y=df_variance['Cumulative'],
        mode='lines+markers',
        name='Cumulative',
        yaxis='y2',
        marker=dict(size=8),
        line=dict(width=2)
    )
    
    fig.update_layout(
        template='plotly_white',
        xaxis_title="Principal Component",
        yaxis_title="Variance Explained",
        yaxis2=dict(
            title="Cumulative Variance",
            overlaying='y',
            side='right',
            range=[0, 1]
        ),
        height=500
    )
    
    return fig

def plot_correlation_dim(df_dim):
    fig = px.bar(
        df_dim,
        y='Method',
        x='Dimension',
        barmode='group',
        text='Dimension',
        orientation='h', 
    )
    fig.update_traces(textposition='outside', textfont_size=12)
    fig.update_layout(
        template='plotly_white',
        xaxis_title="Estimated Dimension",
        yaxis_title="Method",
        showlegend=True,
        yaxis={'categoryorder':'total ascending'}
    )
    return fig