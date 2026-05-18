import torch
import pandas as pd
import scanpy as sc
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

adata = sc.read_h5ad('100epoch.h5ad')
# adata2 = sc.read_h5ad('./Data/adata_PRAGA_HLN.h5ad')
print(adata)

fig, ax_list = plt.subplots(1, 3, figsize=(11, 3))
resolution = 0.8


# 假设 embs 是你的 20 维 numpy 数组
adata1 = sc.AnnData(X=adata.obsm['X_aligned']) 
sc.pp.neighbors(adata1, use_rep='X')
sc.tl.umap(adata1)
sc.tl.leiden(adata1, resolution=resolution) # 自动聚类

sc.pl.umap(adata1, color=['leiden'], ax=ax_list[2], title="Integrate", show=False)

adata2 = sc.AnnData(X=adata.obsm['X_vae_atac']) 
sc.pp.neighbors(adata2, use_rep='X')
sc.tl.umap(adata2)
sc.tl.leiden(adata2, resolution=resolution) # 自动聚类
sc.pl.umap(adata2, color=['leiden'], ax=ax_list[1], title="ADT", show=False)

adata3 = sc.AnnData(X=adata.obsm['X_vae_rna']) 
sc.pp.neighbors(adata3, use_rep='X')
sc.tl.umap(adata3)
sc.tl.leiden(adata3, resolution=resolution) # 自动聚类
sc.pl.umap(adata3, color=['leiden'], ax=ax_list[0], title="RNA", show=False)

plt.tight_layout()
plt.show()