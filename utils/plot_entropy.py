# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm, laplace

# # ===== 熵计算函数 =====
# def mixture_entropy(x, pdf):
#     """
#     用数值积分近似连续分布的熵：
#     H = - ∫ p(x) log p(x) dx
#     """
#     dx = x[1] - x[0]
#     return -np.sum(pdf * np.log(pdf + 1e-12)) * dx

# # ===== 理论熵公式 =====
# def gaussian_entropy(sigma):
#     """高斯分布的熵（nats）"""
#     return 0.5 * np.log(2 * np.pi * np.e * sigma**2)

# def laplace_entropy(b):
#     """拉普拉斯分布的熵（nats）"""
#     return np.log(2 * b * np.e)

# # ===== x 范围 =====
# x = np.linspace(-10, 10, 2000)

# # 无 KL 约束：两个分布相差较大
# pg1 = norm.pdf(x, loc=0, scale=1)         # 高斯 μ=0, σ=1
# pl1 = laplace.pdf(x, loc=5, scale=1)      # 拉普拉斯 μ=5, b=1
# pm1 = 0.5 * pg1 + 0.5 * pl1
# H_m1 = mixture_entropy(x, pm1)

# # 有 KL 约束：两个分布更接近
# pg2 = norm.pdf(x, loc=0, scale=1)
# pl2 = laplace.pdf(x, loc=0.5, scale=1)    # 拉普拉斯 μ=0.5, b=1
# pm2 = 0.5 * pg2 + 0.5 * pl2
# H_m2 = mixture_entropy(x, pm2)

# # 单独分布的理论熵
# H_g = gaussian_entropy(1)
# H_l = laplace_entropy(1)

# # ===== 绘图 =====
# fig, axs = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

# # 无 KL 约束
# axs[0].plot(x, pg1, label="Gaussian")
# axs[0].plot(x, pl1, label="Laplacian")
# axs[0].plot(x, pm1, label="Mixture", color="green")
# axs[0].set_title(f"No KL constraint\nMixture Entropy ≈ {H_m1:.3f} nats")
# axs[0].legend()
# axs[0].grid(True, linestyle="--", alpha=0.5)

# # 有 KL 约束
# axs[1].plot(x, pg2, label="Gaussian")
# axs[1].plot(x, pl2, label="Laplacian")
# axs[1].plot(x, pm2, label="Mixture", color="green")
# axs[1].set_title(f"With KL constraint\nMixture Entropy ≈ {H_m2:.3f} nats")
# axs[1].legend()
# axs[1].grid(True, linestyle="--", alpha=0.5)

# plt.suptitle(
#     f"Gaussian entropy ≈ {H_g:.3f} nats | Laplace entropy ≈ {H_l:.3f} nats"
# )
# plt.show()



import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm, laplace

# ===== 熵计算函数 =====
def mixture_entropy(x, pdf):
    """连续分布熵的数值近似（nats）"""
    dx = x[1] - x[0]
    return -np.sum(pdf * np.log(pdf + 1e-12)) * dx

# ===== KL 散度计算函数 =====
def kl_divergence(p, q, x):
    """
    计算 D_KL(p || q) 的数值近似
    p, q: 概率密度函数值
    """
    dx = x[1] - x[0]
    return np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))) * dx

# ===== 理论熵公式 =====
def gaussian_entropy(sigma):
    return 0.5 * np.log(2 * np.pi * np.e * sigma**2)

def laplace_entropy(b):
    return np.log(2 * b * np.e)

# ===== x 范围 =====
x = np.linspace(-10, 10, 20000)

# 情况 1：无 KL 约束
pg1 = norm.pdf(x, loc=0, scale=1)
pl1 = laplace.pdf(x, loc=5, scale=1)
pm1 = 0.5 * pg1 + 0.5 * pl1
H_m1 = mixture_entropy(x, pm1)
KL_pg_pl1 = kl_divergence(pg1, pl1, x)
KL_pl_pg1 = kl_divergence(pl1, pg1, x)

# 情况 2：有 KL 约束（均值更接近）
pg2 = norm.pdf(x, loc=0, scale=1)
pl2 = laplace.pdf(x, loc=0.0, scale=1)
pm2 = 0.9 * pg2 + 0.1 * pl2
H_m2 = mixture_entropy(x, pm2)
KL_pg_pl2 = kl_divergence(pg2, pl2, x)
KL_pl_pg2 = kl_divergence(pl2, pg2, x)

# 单分布理论熵
H_g = gaussian_entropy(1)
H_l = laplace_entropy(1)

# ===== 绘图 =====
fig, axs = plt.subplots(1, 2, figsize=(13, 4), sharey=True)

# 无 KL 约束
axs[0].plot(x, pg1, label="Gaussian")
axs[0].plot(x, pl1, label="Laplacian")
axs[0].plot(x, pm1, label="Mixture", color="green")
axs[0].set_title(
    f"No KL constraint\n"
    f"Mixture H ≈ {H_m1:.3f} nats\n"
    f"KL(G||L) ≈ {KL_pg_pl1:.3f}, KL(L||G) ≈ {KL_pl_pg1:.3f}"
)
axs[0].legend()
axs[0].grid(True, linestyle="--", alpha=0.5)

# 有 KL 约束
axs[1].plot(x, pg2, label="Gaussian")
axs[1].plot(x, pl2, label="Laplacian")
axs[1].plot(x, pm2, label="Mixture", color="green")
axs[1].set_title(
    f"With KL constraint\n"
    f"Mixture H ≈ {H_m2:.3f} nats\n"
    f"KL(G||L) ≈ {KL_pg_pl2:.3f}, KL(L||G) ≈ {KL_pl_pg2:.3f}"
)
axs[1].legend()
axs[1].grid(True, linestyle="--", alpha=0.5)

plt.suptitle(
    f"Gaussian entropy ≈ {H_g:.3f} nats | Laplace entropy ≈ {H_l:.3f} nats"
)
plt.show()
