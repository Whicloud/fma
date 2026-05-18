import torch
from model import VanillaVAE, FlowAligner

@torch.no_grad()
def predict_modality_b(input_a, vae_a, vae_b, aligner, steps=25):
    """
    通过流匹配从模态 A 预测模态 B
    """
    device = next(aligner.parameters()).device
    vae_a.eval(); vae_b.eval(); aligner.eval()
    
    # 1. Encode A
    mu_a, _ = vae_a.encode(input_a.to(device))
    
    # 2. Flow ODE Solver (Euler)[cite: 1]
    z = mu_a.clone()
    dt = 1.0 / steps
    for i in range(steps):
        t = torch.ones(z.shape[0], device=device) * (i / steps)
        v = aligner.get_velocity(z, t, z_cond=mu_a)
        z = z + v * dt
    
    # 3. Decode to B
    output_b = vae_b.decoder(z)
    return output_b