

import matplotlib.pyplot as plt
import numpy as np

# 数据准备（来自表格）
quality = [55, 65, 75, 85, 95, 100]  # 注意：横坐标间隔不等
methods = {
    'JPEG': [1.644, 1.88, 2.226, 2.921, 5.175, 9.315],
    'Lepton': [1.362, 1.559, 1.843, 2.399, 4.104, 7.47],
    'JPEG XL': [1.43, 1.623, 1.9, 2.44, 4.084, 7.469],
    'Guo': [1.155, 1.335, 1.595, 2.07, 3.682, 6.956],
    'Eff-Net': [1.136, 1.306, 1.572, 2.064, 3.615, 6.803],
    'Ours': [1.062, 1.234, 1.482, 1.907, 3.523, 6.635]
}

# 创建画布
plt.figure(figsize=(10, 6), dpi=120)
# plt.style.use('seaborn-v0_8')

# 关键技巧：构建等间隔的虚拟x轴
x_ticks = [55, 65, 75, 85, 95, 100]  # 实际值
x_visual = np.arange(len(x_ticks))    # 等间隔虚拟坐标 [0,1,2,3,4,5]

# 绘制折线（在虚拟坐标上画线，但标签显示实际值）
markers = ['o', 's', '^', 'D', 'v', 'P']
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for i, (name, data) in enumerate(methods.items()):
    plt.plot(
        x_visual, data,
        label=name,
        marker=markers[i],
        color=colors[i],
        linewidth=2,
        markersize=8,
        linestyle='-'
    )

# 设置坐标轴和网格
plt.title('YCbCr 4:2:0', fontsize=14, pad=20)
plt.xlabel('Quality', fontsize=12)
plt.ylabel('Bits per pixel (BPP)', fontsize=12)

# 关键设置：在虚拟坐标位置显示实际值
plt.xticks(x_visual, x_ticks, fontsize=10)  # 等间隔显示不等距数据
plt.yticks(np.arange(0, 10, 1), fontsize=10)
plt.grid(True, linestyle='--', alpha=0.6, which='both')

# 统一设置网格间隔（虽然x值不等距，但网格线均匀）
ax = plt.gca()
ax.set_axisbelow(True)  # 网格线在数据下方
ax.xaxis.set_major_locator(plt.FixedLocator(x_visual))  # 强制对齐虚拟坐标

# 添加图例
plt.legend(
    loc='upper left',
    frameon=True,
    shadow=True,
    fontsize=10,
    title='Methods'
)

plt.tight_layout()
plt.show()






# import matplotlib.pyplot as plt
# import numpy as np

# # 数据准备（来自表格）
# quality = [55, 65, 75, 85, 95, 100]  # 注意：横坐标间隔不等
# methods = {
#     'JPEG': [2.045, 2.358, 2.828, 3.769, 6.71, 14.334],
#     'Lepton': [1.663, 1.918, 2.297, 3.04, 5.237, 11.38],
#     'JPEG XL': [1.731, 1.979, 2.348, 3.073, 5.229, 11.374],
#     'Guo': [1.675, 1.719, 2.134, 2.896, 4.867, 10.735],
#     'Eff-Net': [1.683, 1.703, 2.073, 2.725, 4.771, 10.577],
#     'Ours': [1.309, 1.536, 1.883, 2.551, 4.547, 10.04]
# }

# # 创建画布
# plt.figure(figsize=(10, 6), dpi=120)
# # plt.style.use('seaborn-v0_8')

# # 关键技巧：构建等间隔的虚拟x轴
# x_ticks = [55, 65, 75, 85, 95, 100]  # 实际值
# x_visual = np.arange(len(x_ticks))    # 等间隔虚拟坐标 [0,1,2,3,4,5]

# # 绘制折线（在虚拟坐标上画线，但标签显示实际值）
# markers = ['o', 's', '^', 'D', 'v', 'P']
# colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# for i, (name, data) in enumerate(methods.items()):
#     plt.plot(
#         x_visual, data,
#         label=name,
#         marker=markers[i],
#         color=colors[i],
#         linewidth=2,
#         markersize=8,
#         linestyle='-'
#     )

# # 设置坐标轴和网格
# plt.title('YCbCr 4:4:4', fontsize=14, pad=20)
# plt.xlabel('Quality', fontsize=12)
# plt.ylabel('Bits per pixel (BPP)', fontsize=12)

# # 关键设置：在虚拟坐标位置显示实际值
# plt.xticks(x_visual, x_ticks, fontsize=10)  # 等间隔显示不等距数据
# plt.yticks(np.arange(0, 16, 2), fontsize=10)
# plt.grid(True, linestyle='--', alpha=0.6, which='both')

# # 统一设置网格间隔（虽然x值不等距，但网格线均匀）
# ax = plt.gca()
# ax.set_axisbelow(True)  # 网格线在数据下方
# ax.xaxis.set_major_locator(plt.FixedLocator(x_visual))  # 强制对齐虚拟坐标

# # 添加图例
# plt.legend(
#     loc='upper left',
#     frameon=True,
#     shadow=True,
#     fontsize=10,
#     title='Methods'
# )

# plt.tight_layout()
# plt.show()