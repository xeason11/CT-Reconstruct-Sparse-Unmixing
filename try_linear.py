import numpy as np
import scipy.io

# 加载数据
data = scipy.io.loadmat('HongHe_square3_dataset.mat')
M = data['M']
Y = data['Y']

# 直接解混计算
A_hat = np.linalg.lstsq(M, Y, rcond=None)[0]

# 限制丰度值在 [0, 1] 范围内
Abundance_matrix = np.clip(A_hat, 0, 1)

# 重构解混后的图像
image_height = 560
image_width = 560
num_bands = 32
Y_hat = Abundance_matrix.reshape((4, image_height, image_width))

# 保存解混后的图像为.mat文件
scipy.io.savemat('Y_hat.mat', {'Y_hat': Y_hat})

非负约束解混示例（使用 nnls）：
import numpy as np
from scipy.optimize import nnls

def linear_unmixing_nnls(M, Y):
    """
    线性解混函数 (非负最小二乘法)
    :param M: 端元矩阵 (L x p)
              L: 波段数
              p: 端元数
    :param Y: 观察数据矩阵 (L x N)
              N: 像素点数
    :return: 丰度矩阵 (p x N)
    """
    p = M.shape[1]  # 端元数
    N = Y.shape[1]  # 像素点数

    A = np.zeros((p, N))  # 初始化丰度矩阵

    for i in range(N):
        A[:, i], _ = nnls(M, Y[:, i])

    return A
