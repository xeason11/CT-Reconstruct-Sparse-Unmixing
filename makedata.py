import numpy as np
import pandas as pd
from scipy.io import savemat, loadmat

# 读取端元矩阵的Excel文件，并将其转换为numpy数组
filename = 'square3_endmember.xlsx'  # 替换为你的Excel文件名
M = pd.read_excel(filename).values

# 针对洪河数据三位数的，除以1000归一化到0-1之间
M = M / 1000.0

# 将数据保存为.mat文件
savemat('M.mat', {'M': M})

# 复制端元矩阵并保存为M1.mat
M1 = M.copy()
savemat('M1.mat', {'M1': M1})


data = loadmat('M1.mat')
M1 = data['M1']
a, b = -0.1, 0.1
random_noise = np.random.uniform(a, b, M1.shape)
M1 += random_noise
savemat('M1.mat', {'M1': M1})

# 加载四个.mat文件的内容
A = loadmat('A.mat')['A']
M = loadmat('M.mat')['M']
M1 = loadmat('M1.mat')['M1']
Y = loadmat('Y.mat')['Y']

# 转置Y
Y = Y.T

# 保存成一个新的.mat文件
savemat('HongHe_square3_dataset.mat', {'A': A, 'M': M, 'M1': M1, 'Y': Y})
