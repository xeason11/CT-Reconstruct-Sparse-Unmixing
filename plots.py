import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
import scipy.io
#size看着改
config = {
    "font.family":'serif',
    "font.size": 12,
    "mathtext.fontset":'stix',
    "font.serif": ['SimSun'],
}
rcParams.update(config)
# land_cover = ['水体', '林地', '沼泽湿地', '草地']#添加 x 轴标签 454，roi1的
# land_cover=['建设用地','草地','耕地','沼泽湿地','林地','水体']#添加 x 轴标签 900，square2的
# land_cover=['沼泽湿地','林地','草地','建设用地']#添加 x 轴标签 560，square3的
def plot_abundance(ground_truth, estimated, linear,em, save_dir):

    fig, a = plt.subplots(em, 3, figsize=(30, 50))
    for i in range(em):
        #先画稀疏解混ground_truth
        # plt.xlabel(land_cover[i])  # 添加 x 轴标签
        a[i][0].imshow(ground_truth[:, :, i].T, cmap='jet')#改了.T的转置，让出图符合原图像南北的排布
        a[i][0].set_xticks([])
        a[i][0].set_yticks([])

    for i in range(em):
        #线性解混丰度图
        # plt.xlabel(land_cover[i])  # 添加 x 轴标签
        a[i][1].imshow(linear[i], cmap='jet')
        a[i][1].set_xticks([])
        a[i][1].set_yticks([])

    for i in range(em):
        # Transformer解混丰度图
        a[i][2].imshow(estimated[:, :, i].T, cmap='jet')
        a[i][2].set_xticks([])
        a[i][2].set_yticks([])
    fig.tight_layout()

    fig.savefig(save_dir + "abundance.png",dpi=300)
    # 横向排列
# def plot_abundance(ground_truth, estimated, em, save_dir):
#
#     plt.figure(figsize=(12, 6), dpi=150)
#     for i in range(em):
#         plt.subplot(2, em, i + 1)
#         plt.imshow(ground_truth[:, :, i], cmap='jet')
#
#     for i in range(em):
#         plt.subplot(2, em, em + i + 1)
#         plt.imshow(estimated[:, :, i], cmap='jet')
#     plt.tight_layout()
#
#     plt.savefig(save_dir + "abundance.png")
#
#
# def plot_endmembers(target, pred, em, save_dir):
#
#     plt.figure(figsize=(12, 6), dpi=150)
#     for i in range(em):
#         plt.subplot(2, em // 2 if em % 2 == 0 else em, i + 1)
#         plt.plot(pred[:, i], label="Extracted")
#         plt.plot(target[:, i], label="GT")
#         plt.legend(loc="upper left")
#     plt.tight_layout()
#
#     plt.savefig(save_dir + "end_members.png")

# def plot_endmembers(target, pred, em, save_dir):
#
#     plt.figure(figsize=(12, 6), dpi=300)
#     for i in range(em):
#         plt.subplot(2, em // 2 if em % 2 == 0 else em, i + 1)
#         # plt.xlabel(land_cover[i])  # 添加 x 轴标签
#         plt.plot(pred[:, i], label="Transformer")
#         plt.plot(target[:, i], label="稀疏解混")
#         plt.legend(loc="upper left")
#     plt.tight_layout()
#
#     plt.savefig(save_dir + "end_members.png")

