import os
import pickle
import numpy as np
import scanpy as sc
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def pca(adata, use_reps=None, n_comps=10):
    
    """Dimension reduction with PCA algorithm"""
    
    from sklearn.decomposition import PCA
    from scipy.sparse.csc import csc_matrix
    from scipy.sparse.csr import csr_matrix
    pca = PCA(n_components=n_comps)

    if use_reps is not None:
       feat_pca = pca.fit_transform(adata.obsm[use_reps])
    else: 
       if isinstance(adata.X, csc_matrix) or isinstance(adata.X, csr_matrix):
          feat_pca = pca.fit_transform(adata.X.toarray()) 
       else:   
          feat_pca = pca.fit_transform(adata.X)
    
    return feat_pca

def mclust_R(adata, num_cluster, modelNames='EEE', used_obsm='emb_pca', random_seed=2020):
    """\
    Clustering using the mclust algorithm.
    The parameters are the same as those in the R package mclust.
    """
    
    np.random.seed(random_seed)
    import rpy2.robjects as robjects
    robjects.r.library("mclust")

    import rpy2.robjects.numpy2ri
    rpy2.robjects.numpy2ri.activate()
    r_random_seed = robjects.r['set.seed']
    r_random_seed(random_seed)
    rmclust = robjects.r['Mclust']
    
    emb = np.asarray(adata.obsm[used_obsm], dtype=np.float64)
    if emb.dtype == object or emb.ndim == 1:
        try:
            emb = np.vstack(emb).astype(np.float64)
        except:
            emb = np.array(emb.tolist(), dtype=np.float64)
    
    if emb.ndim == 1:
        emb = emb.reshape(-1, 1)
        
    emb = np.ascontiguousarray(emb)
    print(f"[mclust_R] used_obsm={used_obsm}, type={type(emb)}, shape={emb.shape}, dtype={emb.dtype}")

    # explicitly set dimnames to NULL to avoid mismatch
    r_matrix = robjects.r.matrix(
        robjects.FloatVector(emb.flatten()),
        nrow=emb.shape[0],
        ncol=emb.shape[1]
    )
    robjects.r['dimnames'](r_matrix).rx2(True)  # ensure call for logging
    robjects.r['dimnames'](r_matrix)[:] = robjects.rinterface.NULL
    dims = robjects.r['dim'](r_matrix)
    dimnames = robjects.r['dimnames'](r_matrix)
    dimnames_len = 0 if dimnames == robjects.rinterface.NULL else len(dimnames)
    print(f"[mclust_R] R dim={list(dims)}, dimnames_len={dimnames_len}")
    res = rmclust(r_matrix, num_cluster, modelNames)
    mclust_res = np.array(res[-2])

    adata.obs['mclust'] = mclust_res
    adata.obs['mclust'] = adata.obs['mclust'].astype('int')
    adata.obs['mclust'] = adata.obs['mclust'].astype('category')
    return adata

def clustering(adata, n_clusters=7, key='emb', add_key='SpatialGlue', method='mclust', start=0.1, end=3.0, increment=0.01, use_pca=False, n_comps=20):
    
    if use_pca:
       adata.obsm[key + '_pca'] = pca(adata, use_reps=key, n_comps=n_comps)
    
    if method == 'mclust':
       if use_pca: 
          adata = mclust_R(adata, used_obsm=key + '_pca', num_cluster=n_clusters)
       else:
          adata = mclust_R(adata, used_obsm=key, num_cluster=n_clusters)
       adata.obs[add_key] = adata.obs['mclust']
    elif method == 'leiden':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.leiden(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['leiden']
    elif method == 'louvain':
       if use_pca: 
          res = search_res(adata, n_clusters, use_rep=key + '_pca', method=method, start=start, end=end, increment=increment)
       else:
          res = search_res(adata, n_clusters, use_rep=key, method=method, start=start, end=end, increment=increment) 
       sc.tl.louvain(adata, random_state=0, resolution=res)
       adata.obs[add_key] = adata.obs['louvain']

def search_res(adata, n_clusters, method='leiden', use_rep='emb', start=0.1, end=3.0, increment=0.01):
    print('Searching resolution...')
    label = 0
    sc.pp.neighbors(adata, n_neighbors=50, use_rep=use_rep)
    for res in sorted(list(np.arange(start, end, increment)), reverse=True):
        if method == 'leiden':
           sc.tl.leiden(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['leiden']).leiden.unique())
           print('resolution={}, cluster number={}'.format(res, count_unique))
        elif method == 'louvain':
           sc.tl.louvain(adata, random_state=0, resolution=res)
           count_unique = len(pd.DataFrame(adata.obs['louvain']).louvain.unique()) 
           print('resolution={}, cluster number={}'.format(res, count_unique))
        if count_unique == n_clusters:
            label = 1
            break

    assert label==1, "Resolution is not found. Please try bigger range or smaller step!." 
       
    return res     



n_clusters = 10

path_fma = "hln_embedding_100.h5ad"
# path_praga = ""
output_path = "cluster_label.h5ad"

key_name = 'fmavae'

adata1 = sc.read_h5ad(path_fma)
print(adata1)
# adata2 = sc.read_h5ad(path_praga)


tool = 'leiden' # mclust, leiden, and louvain

clustering(adata1, key='X_aligned', add_key='fmavae', n_clusters=n_clusters, method=tool, use_pca=True)
# label = adata1.obs[key_name]
adata1.write_h5ad(output_path)