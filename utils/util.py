import torch
import scanpy as sc

def save_embeddings(vae_a, vae_b, aligner, path_a, path_b, device, output_path="aligned_embeddings.h5ad"):
    """
    提取模态 A 和 B 的 Embedding，并保存为 h5ad 文件
    """
    vae_a.eval(); vae_b.eval(); aligner.eval()
    
    # 重新读取 adata 以保留原始的 obs 信息（细胞标签等）[cite: 1]
    adata_a = sc.read_h5ad(path_a)
    adata_b = sc.read_h5ad(path_b)
    common_obs = adata_a.obs_names.intersection(adata_b.obs_names)
    adata_a = adata_a[common_obs].copy()
    
    # 转换数据为 Tensor
    def _get_input(adata):
        x = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        return torch.FloatTensor(x).to(device)

    with torch.no_grad():
        # 1. 提取各模态原始潜变量
        mu_a, _ = vae_a.encode(_get_input(adata_a))
        mu_b, _ = vae_b.encode(_get_input(adata_b))
        
        # 2. 通过 Flow Aligner 进行对齐变换 (A -> B 的演化起点)
        z_aligned = mu_a.clone()
        steps = 25
        for i in range(steps):
            t = torch.ones(z_aligned.shape[0], device=device) * (i / steps)
            v = aligner.get_velocity(z_aligned, t, z_cond=mu_a)
            z_aligned = z_aligned + v * (1.0 / steps)

    # 3. 构建新的 AnnData 对象[cite: 1]
    # 我们将对齐后的 Embedding 存入 obsm，方便后续 scanpy 分析
    combined_adata = sc.AnnData(X=z_aligned.cpu().numpy(), obs=adata_a.obs.copy())
    combined_adata.obsm["X_vae_atac"] = mu_a.cpu().numpy()
    combined_adata.obsm["X_vae_rna"] = mu_b.cpu().numpy()
    combined_adata.obsm["X_aligned"] = z_aligned.cpu().numpy()
    
    combined_adata.write_h5ad(output_path)
    print(f"Embeddings saved to {output_path}")