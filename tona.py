import numpy as np
import scipy.io as sio
import pandas as pd
import cv2  # 使用cv2读取图像
from sunsal import sunsal

# 设置随机种子以获得可重复的结果
np.random.seed(23)

# 端元数
p = 4

# 设置信噪比（SNR）
SNR = 60

# 读取endmember光谱数据
A1 = pd.read_excel('square3_endmember.xlsx').values
A = A1[:, :]
L, m = A.shape  # L为波段数；m为材料数

# 读取图像数据
image_data = cv2.imread('square3.tif', -1)  # 读取为原始数据类型

# 获取图像的行数、列数和波段数
rows, cols, bands = image_data.shape

# 归一化处理
x = image_data / 1000.0  # 图像数据的值范围是 0 到 1000

# 将三维数组重新整形为二维数组
Y = x.reshape(rows * cols, bands)

# 混合矩阵
index = np.random.permutation(m)
M = A[:, index[:p]]


X_hat = sunsal(A, Y.T)

# 重新整形为原始图像大小
X_hat2 = X_hat.T.reshape(rows, cols, m)

# 带正则化约束的最小二乘法
X_hat_l11 = sunsal(A, Y.T, POSITIVITY='yes', VERBOSE='yes', ADDONE='no', 
                   lambda_=1e-4, AL_ITERS=2000, TOL=1e-6)

# 将结果保存为.mat格式
sio.savemat('result.mat', {'A': A, 'M': M, 'Y': Y, 'X_hat2': X_hat2, 'X_hat_l11': X_hat_l11})
