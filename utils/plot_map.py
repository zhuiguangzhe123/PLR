import matplotlib.pyplot as plt
import numpy as np
import torch
import seaborn as sns
from matplotlib.gridspec import GridSpec
from torch import Tensor

path1 = "CompressAI/compress_output/visual_test/y/00reg"
path2 = "CompressAI/compress_output/visual_test/y/02reg"

order = np.load("CompressAI/compress_output/visual_test/y/target.npy")
# x, y = 15, 15
x, y = 16, 16
inverse_map = np.argsort(order[:,0])
path1_mean = np.flip(np.load(path1 + "/means_hat.npy")[inverse_map, :, 0],-1).reshape(32, 32, 8, 8)[x, y, :, :]
path2_mean = np.flip(np.load(path2 + "/means_hat.npy")[inverse_map, :, 0],-1).reshape(32, 32, 8, 8)[x, y, :, :]

path1_scale = np.flip(np.load(path1 + "/scales_hat.npy")[inverse_map, :, 0],-1).reshape(32, 32, 8, 8)[x, y, :, :]
path1_scale = np.clip(path1_scale, 0.11, 1000)
path2_scale = np.flip(np.load(path2 + "/scales_hat.npy")[inverse_map, :, 0],-1).reshape(32, 32, 8, 8)[x, y, :, :]
path2_scale = np.clip(path2_scale, 0.11, 1000)

path1_weight = np.flip(np.load(path1 + "/weights_hat.npy")[inverse_map, :, 0],-1).reshape(32, 32, 8, 8)[x, y, :, :]
path1_weight = torch.sigmoid(torch.from_numpy(path1_weight.copy())).numpy()
path2_weight = np.flip(np.load(path2 + "/weights_hat.npy")[inverse_map, :, 0],-1).reshape(32, 32, 8, 8)[x, y, :, :]
path2_weight = torch.sigmoid(torch.from_numpy(path2_weight.copy())).numpy()

path1_target = np.flip(np.load(path1 + "/target.npy")[inverse_map],-1).reshape(32, 32, 8, 8)[x, y, :, :]
path2_target = np.flip(np.load(path2 + "/target.npy")[inverse_map],-1).reshape(32, 32, 8, 8)[x, y, :, :]


def laplace_standardized_cumulative(inputs: Tensor, means: Tensor, scales: Tensor, half: Tensor):
    values = (half - torch.abs(inputs-means)) / scales
    exp = torch.exp(-torch.abs(values))
    return torch.where(values > 0, 2 - exp, exp) / 2
def gaussian_standardized_cumulative(inputs: Tensor, means: Tensor, scales: Tensor, half: Tensor):
    const = float(-(2**-0.5))
    values = (half - torch.abs(inputs-means)) / scales
    # Using the complementary error function maximizes numerical precision.
    return 0.5 * torch.erfc(const * values)
    

def bpp(inputs: Tensor, scales: Tensor, means: Tensor, weights: Tensor
    ) -> Tensor:
    half = float(0.5)
    # weights = torch.sigmoid(weights)
    scales = torch.clamp_min(scales, 0.11)
    upper_gau = gaussian_standardized_cumulative(inputs, means, scales, half)
    lower_gau = gaussian_standardized_cumulative(inputs, means, scales, -half)
    upper_lap = laplace_standardized_cumulative(inputs, means, scales, half)
    lower_lap = laplace_standardized_cumulative(inputs, means, scales, -half)
    upper = upper_gau * weights + upper_lap * (1 - weights)
    lower = lower_gau * weights + lower_lap * (1 - weights)
    likelihood = upper - lower
    return -torch.log(likelihood)

path1_bpp = bpp(torch.from_numpy(path1_target.copy()), torch.from_numpy(path1_scale.copy()), torch.from_numpy(path1_mean.copy()), torch.from_numpy(path1_weight.copy()))
path2_bpp = bpp(torch.from_numpy(path2_target.copy()), torch.from_numpy(path2_scale.copy()), torch.from_numpy(path2_mean.copy()), torch.from_numpy(path2_weight.copy()))

# path2_scale[1, 1] = 10.2

print("bpp of path1:", path1_bpp.mean().item())
print("bpp of path2:", path2_bpp.mean().item())

rows, cols = 2, 6
fig, axes = plt.subplots(rows, cols, figsize=(18, 6))
datas = [path1_target, path1_mean, path1_scale, path1_weight, (path1_target-path1_mean)/path1_scale, path1_bpp,
         path2_target, path2_mean, path2_scale, path2_weight, (path2_target-path2_mean)/path2_scale, path2_bpp]
datas1_max_min = [(path1_target.max(),path1_target.min()), (path1_mean.max(),path1_mean.min()), (path1_scale.max(),path1_scale.min()), (path1_weight.max(),path1_weight.min()), 
                 (((path1_target-path1_mean)/path1_scale).max(), ((path1_target-path1_mean)/path1_scale).min()),
                 (path1_target.max(),path1_target.min()), (path1_mean.max(),path1_mean.min()), (path1_scale.max(),path1_scale.min()), (path1_weight.max(),path1_weight.min()), 
                 (((path1_target-path1_mean)/path1_scale).max(), ((path1_target-path1_mean)/path1_scale).min())]

datas2_max_min = [(path2_target.max(),path2_target.min()), (path2_mean.max(),path2_mean.min()), (path2_scale.max(),path2_scale.min()), (path2_weight.max(),path2_weight.min()), 
                 (((path2_target-path2_mean)/path2_scale).max(), ((path2_target-path2_mean)/path2_scale).min()),
                 (path2_target.max(),path2_target.min()), (path2_mean.max(),path2_mean.min()), (path2_scale.max(),path2_scale.min()), (path2_weight.max(),path2_weight.min()), 
                 (((path2_target-path2_mean)/path2_scale).max(), ((path2_target-path2_mean)/path2_scale).min())]

tiles = ["DCT coeffs", "Mean", "Scale", "Weight", "Remaining redundancy", "Required bits", "DCT coeffs", "Mean", "Scale", "Weight", "Remaining redundancy", "Required bits"]
for i in range(rows):
    for j in range(cols):
        data = datas[i*cols+j]  # 示例数据
        tile = tiles[i*cols+j]  # 示例数据名称
        # data_max1, data_min1 = datas1_max_min[i*cols+j]  # 最大最小值
        # data_max2, data_min2 = datas2_max_min[i*cols+j]  # 最大最小值

        # vmax = max(data_max1, data_max2)
        # vmin = min(data_min1, data_min2)
        
        # im = axes[i, j].imshow(data, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")  # 画图
        im = axes[i, j].imshow(data, cmap="RdBu_r", aspect="auto")  # 画图
        axes[i, j].axis("off")  # 去掉坐标轴
        if i == 0:
            axes[i, j].set_title(tile, fontsize=12, weight='bold')

        # 单独加 colorbar
        plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()

