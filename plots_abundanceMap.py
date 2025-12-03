import scipy.io
import plots  # 程序自编绘图脚本
import torch

## 6*900*900的：square1

# data_a = scipy.io.loadmat(r'E:\Python_Projects\DeepTrans-HSU\data\HongHe_square2_dataset.mat')  # 原数据，包含Y、A、M、M1的那个mat
# target = torch.reshape((torch.from_numpy(data_a['A'].T)), (900, 900, 6)).cpu().numpy()  # reshape原丰度矩阵A为H*W*B，准备画图
#
# data_a1 = scipy.io.loadmat(
#     r'E:\Python_Projects\DeepTrans-HSU\trans_mod_HongHe_square2\HongHe_square2_abd_map.mat')  # 本程序解混的丰度矩阵，mat里名为A_est
#
# data_l=scipy.io.loadmat('data/try_linear2.mat')#线性解混的丰度矩阵
# data_m1 = scipy.io.loadmat(
#     r'E:\Python_Projects\DeepTrans-HSU\trans_mod_HongHe_square2\HongHe_square2_endmem.mat')  # 本程序解混的反射率值光谱曲线，mat里名为E_est
#
# plots.plot_abundance(target, data_a1['A_est'],data_l['Y_hat'], 6, './')
# plots.plot_endmembers(data_a['M'], data_m1['E_est'], 6, './')

# ## ##################4*454*454的square2

# data_a = scipy.io.loadmat(r'E:\Python_Projects\DeepTrans-HSU\data\HongHe_roisquare_dataset.mat')  # 原数据，包含Y、A、M、M1的那个mat
# target = torch.reshape((torch.from_numpy(data_a['A'].T)), (454, 454, 4)).cpu().numpy()  # reshape原丰度矩阵A为H*W*B，准备画图
#
# data_a1 = scipy.io.loadmat(
#     r'E:\Python_Projects\DeepTrans-HSU\trans_mod_HongHe_roisquare\HongHe_roisquare_abd_map.mat')  # 本程序解混的丰度矩阵，mat里名为A_est
#
# data_l=scipy.io.loadmat('data/try_linear.mat')#线性解混的丰度矩阵
# data_m1 = scipy.io.loadmat(
#     r'E:\Python_Projects\DeepTrans-HSU\trans_mod_HongHe_roisquare\HongHe_roisquare_endmem.mat')  # 本程序解混的反射率值光谱曲线，mat里名为E_est
#
# plots.plot_abundance(target, data_a1['A_est'],data_l['Y_hat'], 4, './')
# plots.plot_endmembers(data_a['M'], data_m1['E_est'], 4, './').


# ## 4*560*560的square3

data_a = scipy.io.loadmat(r'E:\Python_Projects\DeepTrans-HSU\data\HongHe_square3_dataset.mat')  # 原数据，包含Y、A、M、M1的那个mat
target = torch.reshape((torch.from_numpy(data_a['A'].T)), (560, 560, 4)).cpu().numpy()  # reshape原丰度矩阵A为H*W*B，准备画图

data_a1 = scipy.io.loadmat(
    r'E:\Python_Projects\DeepTrans-HSU\trans_mod_HongHe_square3\HongHe_square3_abd_map.mat')  # 本程序解混的丰度矩阵，mat里名为A_est

data_l=scipy.io.loadmat('data/try_linear3.mat')#线性解混的丰度矩阵
data_m1 = scipy.io.loadmat(
    r'E:\Python_Projects\DeepTrans-HSU\trans_mod_HongHe_square3\HongHe_square3_endmem.mat')  # 本程序解混的反射率值光谱曲线，mat里名为E_est

plots.plot_abundance(target, data_a1['A_est'],data_l['Y_hat'], 4, './')
plots.plot_endmembers(data_a['M'], data_m1['E_est'], 4, './')

