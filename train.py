import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from model import VanillaVAE, FlowAligner
from utils.datareader import H5ADMultiOmicsDataset
from utils.util import save_embeddings

def kl_loss_fn(mu, logvar):
    # 计算公式: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0)

def train(path_a, path_b, latent_dim=64, epochs=100, batch_size=32, lr=1e-4, device=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = H5ADMultiOmicsDataset(path_a, path_b)
    dim_a = dataset.input_dim_a
    dim_b = dataset.input_dim_b
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print(f"Data loaded. Modality A: {dim_a} dims, Modality B: {dim_b} dims. Cells: {len(dataset)}")

    # 初始化
    vae_a = VanillaVAE(dim_a, latent_dim).to(device)
    vae_b = VanillaVAE(dim_b, latent_dim).to(device)
    aligner = FlowAligner(latent_dim).to(device)
    
    params = list(vae_a.parameters()) + list(vae_b.parameters()) + list(aligner.parameters())
    optimizer = optim.Adam(params, lr=lr, weight_decay=1e-5)
    mse_loss = nn.MSELoss()
    beta = 0.1

    for epoch in range(epochs):
            vae_a.train()
            vae_b.train()
            aligner.train()
            
            epoch_vae_loss = 0
            epoch_fm_loss = 0
            
            for b_a, b_b in loader:
                b_a, b_b = b_a.to(device), b_b.to(device)
                optimizer.zero_grad()
                
                # --- VAE Forward: 提取并重构 ---
                recon_a, mu_a, logvar_a = vae_a(b_a)
                recon_b, mu_b, logvar_b = vae_b(b_b)
                
                # 重参数化获取潜变量[cite: 1]
                z_a = vae_a.reparameterize(mu_a, logvar_a)
                z_b = vae_b.reparameterize(mu_b, logvar_b)
                
                # --- Flow Matching: 学习轨迹 ---[cite: 1, 2]
                # 采样随机时间步 t ~ U(0, 1)
                t = torch.rand(z_a.shape[0], device=device)
                # t = torch.rand(batch_size, device=device)
                t_exp = t.view(-1, 1)
                
                # 构造概率路径 (Optimal Transport Path): x_t = (1-t)z_a + t*z_b[cite: 1]
                x_t = (1 - t_exp) * z_a + t_exp * z_b
                
                # 目标速度场 (Ground Truth Velocity)[cite: 1]
                target_v = z_b - z_a
                
                # 预测速度场 (以 z_a 为 condition 引导)[cite: 2]
                pred_v = aligner.get_velocity(x_t, t, z_cond=z_a)
                
                # --- Loss 计算 ---
                # 1. 重构损失: 确保潜空间保留原始组学信息
                loss_recon = mse_loss(recon_a, b_a) + mse_loss(recon_b, b_b) + (kl_loss_fn(mu_a, logvar_a) + beta * kl_loss_fn(mu_b, logvar_b))
                # 2. 对齐损失: 最小化预测速度与真实轨迹速度的差异[cite: 1]
                loss_fm = mse_loss(pred_v, target_v)

                # loss_kl = (kl_loss_fn(mu_a, logvar_a) + beta * kl_loss_fn(mu_b, logvar_b))
                
                total_loss = loss_recon + loss_fm
                
                total_loss.backward()
                optimizer.step()
                
                epoch_vae_loss += loss_recon.item()
                epoch_fm_loss += loss_fm.item()
                
            # 打印进度
            if (epoch + 1) % 10 == 0 or epoch == 0:
                avg_vae = epoch_vae_loss / len(loader)
                avg_fm = epoch_fm_loss / len(loader)
                print(f"Epoch [{epoch+1}/{epochs}] | VAE Loss: {avg_vae:.4f} | FM Loss: {avg_fm:.4f}")

        # 5. 保存检查点
    save_embeddings(vae_a, vae_b, aligner, path_a, path_b, device, "final_embeddings.h5ad")
    save_path = "sc_multiomics_flow_model.pth"
    torch.save({
        'vae_a_state_dict': vae_a.state_dict(),
        'vae_b_state_dict': vae_b.state_dict(),
        'aligner_state_dict': aligner.state_dict(),
        'config': {
            'dim_a': dim_a,
            'dim_b': dim_b,
            'latent_dim': latent_dim
        }
    }, save_path)
    print(f"Training complete. Model saved to {save_path}")
        
    return vae_a, vae_b, aligner

if __name__ == "__main__":
    # 使用示例 (请替换为你自己的 h5ad 路径)
    # 比如在 AutoDL 上：/root/autodl-tmp/data/atac.h5ad
    PATH_ATAC = "datasets/10x_ADT_preprocessed.h5ad"
    PATH_RNA = "datasets/10x_RNA_preprocessed.h5ad"
    
    if os.path.exists(PATH_ATAC) and os.path.exists(PATH_RNA):
        train(PATH_ATAC, PATH_RNA, epochs=50)
    else:
        print("Please provide valid paths to your .h5ad files.")