from scipy.spatial.distance import cdist
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.sparse import csr_matrix
import numpy as np

def calculate_dbcv_manual(X, labels, metric='euclidean'):
    #Simplified version
    X = X.astype(np.float64)
    n_samples = X.shape[0]
    
    unique_labels = set(labels)
    if -1 in unique_labels:
        unique_labels.remove(-1)
    unique_labels = list(unique_labels)
    n_clusters = len(unique_labels)
    
    if n_clusters < 2:
        return np.nan

    try:
        dist_matrix = cdist(X, X, metric=metric)
    except Exception:
        return np.nan

    def get_density_sparseness(cluster_idx):
        mask = (labels == cluster_idx)
        cluster_data = X[mask]
        n_points = cluster_data.shape[0]
        
        if n_points <= 1:
            return 0
            
        internal_dist = dist_matrix[np.ix_(mask, mask)]
        mst = minimum_spanning_tree(csr_matrix(internal_dist))
        
        return np.max(mst.data)

    def get_density_separation(idx_1, idx_2):
        mask1 = (labels == idx_1)
        mask2 = (labels == idx_2)
        inter_dist = dist_matrix[np.ix_(mask1, mask2)]
        return np.min(inter_dist)

    dbcv_accum = 0
    total_samples = 0
    
    dsc_values = {lab: get_density_sparseness(lab) for lab in unique_labels}
    
    for label in unique_labels:
        cluster_mask = (labels == label)
        n_points_in_cluster = np.sum(cluster_mask)
        total_samples += n_points_in_cluster
        
        dsc = dsc_values[label]
        
        min_dsp = np.inf
        for other_label in unique_labels:
            if label == other_label:
                continue
            dsp = get_density_separation(label, other_label)
            if dsp < min_dsp:
                min_dsp = dsp
        
        if np.isinf(min_dsp): min_dsp = 0
        
        if max(dsc, min_dsp) == 0:
            validity = 0
        else:
            validity = (min_dsp - dsc) / max(min_dsp, dsc)
            
        dbcv_accum += validity * n_points_in_cluster

    return dbcv_accum / total_samples