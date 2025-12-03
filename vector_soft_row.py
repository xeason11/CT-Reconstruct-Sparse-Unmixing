import numpy as np

def vector_soft_row(X, tau):
    # 向量软阈值操作
    # 逐列计算向量软阈值

    NU = np.sqrt(np.sum(X**2, axis=1))
    # 欧几里得范数

    A = np.maximum(0, NU - tau)
    # 软阈值化

    Y = np.tile((A / (A + tau))[:, np.newaxis], (1, X.shape[1])) * X
    # 将 A 扩展为与 X 相同的大小，并将其除以 A+tau 的值，然后与 X 相乘
    # 这一步实现了软阈值操作，即将 X 中每个元素与相应的软阈值比较，大于软阈值的保持不变，小于软阈值的设为 0

    return Y
