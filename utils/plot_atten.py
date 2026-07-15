import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import math

attentions = np.load('CompressAI/compress_output/visual_test/y/attention_1024_4_65_65.npy')
index = 10
head1 = attentions[index,0, 1:, 1:]
head2 = attentions[index,1, 1:, 1:]
head3 = attentions[index,2, 1:, 1:]
head4 = attentions[index,3, 1:, 1:]


###############################################################################
values =  np.load('CompressAI/compress_output/visual_test/y/value_1024_64.npy')
# flattened = torch.from_numpy(values[0]).float()

# # 计算所有位置的点积
# attention_scores = torch.outer(flattened, flattened)  # 64x64

# # 生成因果掩码（左下三角为1，右上为0）
# mask = torch.triu(torch.ones(64, 64), diagonal=1).bool()
# attention_scores.masked_fill_(mask, float('-inf'))

# # Softmax归一化
# causal_attention = F.softmax(attention_scores, dim=-1)

flattened = values[index]
causal_similarity = np.zeros((64, 64))

# 计算每个位置i与j<=i的余弦相似度
for i in range(64):
    for j in range(64):
        if j <= i:
            # 余弦相似度 = (A·B) / (||A|| * ||B||)
            # dot_product = np.dot(flattened[i], flattened[j])
            # norm_i = np.linalg.norm(flattened[i])
            # norm_j = np.linalg.norm(flattened[j])
            # causal_similarity[i, j] = - abs(flattened[i] - flattened[j])
            causal_similarity[i, j] = math.exp(-(flattened[i] - flattened[j])**2 / (2 * 1.0**2))
            # if norm_i > 0 and norm_j > 0:
            #     causal_similarity[i, j] = dot_product / (norm_i * norm_j)
            # else:
            #     causal_similarity[i, j] = 1.0  # 零向量处理
        else:
            causal_similarity[i, j] = 0  # 未来位置置0
###############################################################################

rows, cols = 3, 2
fig, axes = plt.subplots(rows, cols, figsize=(9, 6))

datas = [head1, head2, head3, head4, causal_similarity, causal_similarity]
tiles = ['head1', 'head2', 'head3', 'head4', 'similarity', 'similarity']


for i in range(rows):
    for j in range(cols):
        data = datas[i*cols+j]  # 示例数据
        tile = tiles[i*cols+j]  # 示例数据名称
        # data_max1, data_min1 = datas1_max_min[i*cols+j]  # 最大最小值
        # data_max2, data_min2 = datas2_max_min[i*cols+j]  # 最大最小值

        # vmax = max(data_max1, data_max2)
        # vmin = min(data_min1, data_min2)
        
        # im = axes[i, j].imshow(data, cmap="RdBu_r", vmin=vmin, vmax=vmax, aspect="auto")  # 画图
        im = axes[i, j].imshow(data, cmap="coolwarm", aspect="auto")  # 画图
        axes[i, j].axis("off")  # 去掉坐标轴
        axes[i, j].set_title(tile, fontsize=12, weight='bold')
        # if i == 0:
        #     axes[i, j].set_title(tile, fontsize=12, weight='bold')

        # 单独加 colorbar
        plt.colorbar(im, ax=axes[i, j], fraction=0.046, pad=0.04)

plt.tight_layout()
plt.show()
