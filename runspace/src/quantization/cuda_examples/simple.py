import torch
from runspace.src.quantization.qfp8_tensor import QFP8Tensor

x = torch.arange(1, 20, dtype=torch.float32, device='cuda')

q = QFP8Tensor.from_float(x, e=4, m=3, b=15, mode='tensor')
x_recon = q.to_float()



print('original :', x.cpu().tolist())
print('decoded  :', x_recon.cpu().tolist())