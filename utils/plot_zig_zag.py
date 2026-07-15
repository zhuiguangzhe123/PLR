# import matplotlib.pyplot as plt
# import numpy as np

# # 生成 Zig-Zag 路径
# zigzag_path = []
# for s in range(15):  # s = u + v
#     if s % 2 == 0:
#         for i in range(s + 1):
#             j = s - i
#             if i < 8 and j < 8:
#                 zigzag_path.append((i, j))
#     else:
#         for i in range(s + 1):
#             j = s - i
#             if j < 8 and i < 8:
#                 zigzag_path.append((j, i))

# y_coords, x_coords = zip(*zigzag_path)

# # 绘制箭头路径图
# fig, ax = plt.subplots(figsize=(6, 6))
# ax.set_xlim(-0.5, 7.5)
# ax.set_ylim(7.5, -0.5)
# ax.set_xticks(np.arange(8))
# ax.set_yticks(np.arange(8))
# ax.grid(True)
# ax.set_title("JPEG DCT Zig-Zag Path (with Arrows)")

# # 绘制箭头
# for i in range(len(x_coords) - 1):
#     ax.annotate("",
#         xy=(x_coords[i + 1], y_coords[i + 1]),
#         xytext=(x_coords[i], y_coords[i]),
#         arrowprops=dict(arrowstyle="->", color='blue', lw=1),
#     )

# # 起点终点标记
# ax.text(x_coords[0], y_coords[0], "Start", color="green", ha="center", va="center", fontsize=10, fontweight="bold")
# ax.text(x_coords[-1], y_coords[-1], "End", color="red", ha="center", va="center", fontsize=10, fontweight="bold")

# plt.tight_layout()
# plt.savefig("jpeg_dct_zigzag_arrow_path.png")
# plt.show()


import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

def generate_zigzag_indices(n=8):
    """ 生成n x n矩阵的Zig-zag索引顺序 """
    indices = []
    for i in range(2*n-1):
        if i % 2 == 0:  # 偶数对角线：从下往上
            x = min(i, n-1)
            # y = max(0, i - n + 1)
            y = i - x
            while x >= 0 and y < n:
                indices.append((x, y))
                x -= 1
                y += 1
        else:           # 奇数对角线：从上往下
            y = min(i, n-1)
            x = i - y
            while x < n and y >= 0:
                indices.append((x, y))
                x += 1
                y -= 1
    return indices

# 生成 Zig-Zag 路径
zigzag_path = []
for s in range(15):  # s = u + v
    if s % 2 == 0:
        for i in range(s + 1):
            j = s - i
            if i < 8 and j < 8:
                zigzag_path.append((i, j))
    else:
        for i in range(s + 1):
            j = s - i
            if j < 8 and i < 8:
                zigzag_path.append((j, i))

# 中心坐标
zigzag_path = generate_zigzag_indices(n=8)

y_coords, x_coords = zip(*zigzag_path)
x_coords = np.array(x_coords) + 0.5
y_coords = np.array(y_coords) + 0.5

# 生成正常路径
ori_path = []
for i in range(8):  # s = u + v
    for j in range(8):
        ori_path.append((i, j))

# 中心坐标
y_coords_ori, x_coords_ori = zip(*ori_path)
x_coords_ori = np.array(x_coords_ori) + 0.5
y_coords_ori = np.array(y_coords_ori) + 0.5

# 绘图
fig, ax = plt.subplots(figsize=(7, 7))
ax.set_xlim(0, 8)
ax.set_ylim(8, 0)
# ax.set_xticks(np.arange(9))
# ax.set_yticks(np.arange(9))
ax.grid(True)
ax.set_title("JPEG DCT Zig-Zag Path (Centered with Arrows & Indices)")

zigzag_order = np.array([
    [0,  1,  5,  6,  14, 15, 27, 28, 28],
    [2,  4,  7,  13, 16, 26, 29, 42, 42],
    [3,  8,  12, 17, 25, 30, 41, 43, 43],
    [9,  11, 18, 24, 31, 40, 44, 53, 53],
    [10, 19, 23, 32, 39, 45, 52, 54, 54],
    [20, 22, 33, 38, 46, 51, 55, 60, 60],
    [21, 34, 37, 47, 50, 56, 59, 61, 61],
    [35, 36, 48, 49, 57, 58, 62, 63, 63],
    [35, 36, 48, 49, 57, 58, 62, 63, 63]
])
colors = ["#2b08a8", "#1e90ff", "#00bfff", "#87cefa", "#e0ffff", "#fffacd", "#ffd700", "#ffa500"]
cmap = LinearSegmentedColormap.from_list("custom_jet", colors)
# plt.figure(figsize=(8, 6), dpi=120)
# plt.imshow(zigzag_order, cmap=cmap, vmin=0, vmax=63)
ax.imshow(zigzag_order, cmap=cmap, vmin=0, vmax=63)
# 绘制每个编号

for idx, (x, y) in enumerate(zip(x_coords_ori, y_coords_ori)):
    # ax.text(x, y, str(idx), fontsize=20, ha="center", va="center",
    #         bbox=dict(facecolor='white', edgecolor='none', boxstyle='round,pad=0.1'))
    ax.text(x, y, str(idx), fontsize=20, ha="center", va="center")

# 绘制路径箭头
for i in range(len(x_coords) - 1):
    ax.annotate("",
        xy=(x_coords[i + 1], y_coords[i + 1]),
        xytext=(x_coords[i], y_coords[i]),
        arrowprops=dict(arrowstyle="simple", color='blue', lw=1),
    )


# plt.tight_layout()

plt.show()
plt.savefig("jpeg_dct_zigzag_vector_path.png")