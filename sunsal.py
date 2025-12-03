import numpy as np

def sunsal(M, y, **kwargs):
    # 检查输入参数的个数是否正确
    if len(kwargs) % 2 != 0:
        raise ValueError('Optional parameters should always go by pairs')


    LM, p = M.shape  # 混合矩阵 M 通道数和端元数

    L, N = y.shape  # 观测数据 y 通道数和像素数
    if LM != L:
        raise ValueError('Mixing matrix M and data set y are inconsistent')


    AL_iters = 1000
    lambda_ = 0.0
    verbose = 'off'
    positivity = 'no'
    addone = 'no'
    tol = 1e-4
    x0 = 0


    for i in range(0, len(kwargs), 2):
        key = kwargs[list(kwargs.keys())[i]]
        value = kwargs[list(kwargs.keys())[i+1]]
        if key.upper() == 'AL_ITERS':
            AL_iters = int(value)
            if AL_iters <= 0:
                raise ValueError('AL_iters must be a positive integer')
        elif key.upper() == 'LAMBDA':
            lambda_ = value
            if np.any(lambda_ < 0):
                raise ValueError('lambda must be positive')
        elif key.upper() == 'POSITIVITY':
            positivity = value
        elif key.upper() == 'ADDONE':
            addone = value
        elif key.upper() == 'TOL':
            tol = value
        elif key.upper() == 'VERBOSE':
            verbose = value
        elif key.upper() == 'X0':
            x0 = value
            if x0.shape != (p, N):
                raise ValueError('initial X is inconsistent with M or Y')
        else:
            raise ValueError(f'Unrecognized option: {key}')

    #
    lambda_ = np.array(lambda_)
    if lambda_.size == 1:
        lambda_ = lambda_ * np.ones((p, N))
    elif lambda_.size != N:
        raise ValueError('Lambda size is inconsistent with the size of the data set')
    else:
        lambda_ = np.tile(lambda_.reshape(1, -1), (p, 1))


    norm_y = np.sqrt(np.mean(y**2))

    M = M / norm_y
    y = y / norm_y
    lambda_ = lambda_ / norm_y**2


    if np.all(lambda_ == 0) and positivity == 'no' and addone == 'no':
        z = np.linalg.pinv(M).dot(y)
        res_p = 0
        res_d = 0
        return z, res_p, res_d


    SMALL = 1e-12
    B = np.ones((1, p))
    a = np.ones((1, N))

    if addone == 'yes' and positivity == 'no':
        F = M.T @ M
        if np.linalg.cond(F) > 1 / SMALL:
            IF = np.linalg.inv(F)
            z = IF @ M.T @ y - IF @ B.T @ np.linalg.inv(B @ IF @ B.T) @ (B @ IF @ M.T @ y - a)
            res_p = 0
            res_d = 0
            return z, res_p, res_d

    mu_AL = 0.01
    mu = 10 * np.mean(lambda_) + mu_AL

    UF, SF, _ = np.linalg.svd(M.T @ M)
    sF = np.diag(SF)
    IF = UF @ np.diag(1 / (sF + mu)) @ UF.T

    Aux = IF @ B.T @ np.linalg.inv(B @ IF @ B.T)
    x_aux = Aux @ a
    IF1 = IF - Aux @ B @ IF
    yy = M.T @ y


    if x0 == 0:
        x = IF @ M.T @ y
    z = x
    d = np.zeros_like(z)


    tol1 = np.sqrt(N * p) * tol
    tol2 = np.sqrt(N * p) * tol
    i = 1
    res_p = np.inf
    res_d = np.inf
    maskz = np.ones_like(z)
    mu_changed = 0

    while (i <= AL_iters) and ((abs(res_p) > tol1) or (abs(res_d) > tol2)):
        if i % 10 == 1:
            z0 = z
        z = soft(x - d, lambda_ / mu)
        if positivity == 'yes':
            maskz = z >= 0
            z = z * maskz
        if addone == 'yes':
            x = IF1 @ (yy + mu * (z + d)) + x_aux
        else:
            x = IF @ (yy + mu * (z + d))
        d = d - (x - z)
        if i % 10 == 1:
            res_p = np.linalg.norm(x - z, 'fro')
            res_d = mu * np.linalg.norm(z - z0, 'fro')
            if verbose == 'yes':
                print(f'i = {i}, res_p = {res_p}, res_d = {res_d}')
            if res_p > 10 * res_d:
                mu = mu * 2
                d = d / 2
                mu_changed = 1
            elif res_d > 10 * res_p:
                mu = mu / 2
                d = d * 2
                mu_changed = 1
            if mu_changed:
                IF = UF @ np.diag(1 / (sF + mu)) @ UF.T
                Aux = IF @ B.T @ np.linalg.inv(B @ IF @ B.T)
                x_aux = Aux @ a
                IF1 = IF - Aux @ B @ IF
                mu_changed = 0
        i += 1

    return z, res_p, res_d

def soft(x, kappa):
    return np.sign(x) * np.maximum(np.abs(x) - kappa, 0)
