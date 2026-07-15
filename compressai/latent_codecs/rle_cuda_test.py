import torch
import numpy as np
try:
    import rle_cuda
except ImportError:
    print("CUDA extension not found. Compiling from source...")


target = [0, 0, 0, 0, 5, 7, 9, 1] 
target = torch.from_numpy(np.array(target)).long().unsqueeze(0).cuda()
new_target, mask = rle_cuda.rle_encode(target, 1)
print(new_target)