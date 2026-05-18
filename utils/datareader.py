import torch
from torch.utils.data import Dataset
import scanpy as sc
import numpy as np

class H5ADMultiOmicsDataset(Dataset):
    def __init__(self, file_a_path, file_b_path, layer=None):
        """
        file_a_path: 模态 A (如 ATAC) 的 .h5ad 文件路径
        file_b_path: 模态 B (如 RNA) 的 .h5ad 文件路径
        layer: 如果数据存储在 adata.layers 中，请指定层名；否则默认读取 adata.X
        """
        # 读取数据
        adata_a = sc.read_h5ad(file_a_path)
        adata_b = sc.read_h5ad(file_b_path)

        # 核心：确保两个文件的细胞顺序完全一致
        # 假设 index 是细胞的唯一标识符 (Barcode)
        common_obs = adata_a.obs_names.intersection(adata_b.obs_names)
        if len(common_obs) == 0:
            raise ValueError("两个 h5ad 文件没有共同的细胞 Barcode，无法进行配对训练！")
        
        adata_a = adata_a[common_obs].copy()
        adata_b = adata_b[common_obs].copy()

        # 提取矩阵并转换为 FloatTensor
        # 如果是稀疏矩阵则转换为稠密矩阵
        self.data_a = self._to_tensor(adata_a, layer)
        self.data_b = self._to_tensor(adata_b, layer)
        
        self.input_dim_a = self.data_a.shape[1]
        self.input_dim_b = self.data_b.shape[1]

    def _to_tensor(self, adata, layer):
        data = adata.layers[layer] if layer else adata.X
        if hasattr(data, "toarray"): # 处理 scipy 稀疏矩阵
            data = data.toarray()
        return torch.FloatTensor(data)

    def __len__(self):
        return len(self.data_a)

    def __getitem__(self, idx):
        return self.data_a[idx], self.data_b[idx]