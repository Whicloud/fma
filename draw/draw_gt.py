import torch
import pandas as pd
import scanpy as sc
import numpy as np
import argparse
import time
import matplotlib.pyplot as plt
import matplotlib as mpl

adata1 = sc.read_h5ad('datasets/10x_ADT_preprocessed.h5ad')
adata2 = sc.read_h5ad('datasets/10x_RNA_preprocessed.h5ad')

print(adata1)
print(adata2)

fig, ax_list = plt.subplots(1, 2, figsize=(7, 3))



# 假设 embs 是你的 20 维 numpy 数组
adata_adt = sc.AnnData(X=adata1.obsm['feat']) 
sc.pp.neighbors(adata_adt, use_rep='X')
sc.tl.umap(adata_adt)
sc.tl.leiden(adata_adt) # 自动聚类
sc.pl.umap(adata_adt, color=['leiden'], ax=ax_list[1], title="ADT", show=False)

adata_rna = sc.AnnData(X=adata2.obsm['feat']) 
sc.pp.neighbors(adata_rna, use_rep='X')
sc.tl.umap(adata_rna)
sc.tl.leiden(adata_rna) # 自动聚类
sc.pl.umap(adata_rna, color=['leiden'], ax=ax_list[0], title="RNA", show=False)


plt.tight_layout()
plt.show()