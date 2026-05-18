import numpy as np
from train import train
from test import predict_modality_b
import torch

# 1. 准备你的数据 (Replace with your scanpy/adata data)
X_atac = np.random.rand(500, 2000) 
X_rna = np.random.rand(500, 1000)

# 2. 训练
vae_a, vae_b, aligner = train(2000, 1000, 32, X_atac, X_rna, epochs=10)

# 3. 推理示例
test_cell = torch.FloatTensor(X_atac[:1])
pred_rna = predict_modality_b(test_cell, vae_a, vae_b, aligner)
print("Prediction finished:", pred_rna.shape)