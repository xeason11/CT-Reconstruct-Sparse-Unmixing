import torch.utils.data
import scipy.io as sio
import torchvision.transforms as transforms

class TrainData(torch.utils.data.Dataset):
    def __init__(self, img, target, transform=None, target_transform=None):
        self.img = img.float()
        self.target = target.float()
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        img, target = self.img[index], self.target[index]
        if self.transform:
            img = self.transform(img)
        if self.target_transform:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.img)

class Data:
    def __init__(self, dataset, device):
        super(Data, self).__init__()
        #数据必须是方形的
        data_path = "./data/" + dataset + "_dataset.mat"
        # P端元数 L波段数 col图像宽/高（方形的）
        if dataset == 'HongHe_roisquare':
            self.P, self.L, self.col = 4, 32, 454
        elif dataset == 'HongHe_square2':
            self.P, self.L, self.col = 6, 32, 900
        elif dataset == 'HongHe_square3':
            self.P, self.L, self.col = 4, 32, 560

        data = sio.loadmat(data_path)# 文件夹data里的samson_dataset.mat数据，此mat包含下面四个二维矩阵
        self.Y = torch.from_numpy(data['Y'].T).to(device)#高光谱数据矩阵，行：波段数 列：图像H*W。
        self.A = torch.from_numpy(data['A'].T).to(device)#丰度矩阵，行：端元数，列：图像H*W。Aij表示像素中第i个端元的丰度 解混之后输入
        self.M = torch.from_numpy(data['M'])#端元矩阵 行数：波段数 ，列数：端元种类数。从envi手动选择纯净像素，转置连接制作。文件夹里的sample.xlsx就是
        self.M1 = torch.from_numpy(data['M1'])#初始化权重，用于模型训练的初始端元光谱，或者用于端元估计的预训练权重。

    def get(self, typ):
        if typ == "hs_img":#高光谱数据矩阵
            return self.Y.float()
        elif typ == "abd_map":#丰度矩阵
            return self.A.float()
        elif typ == "end_mem":#端元矩阵
            return self.M
        elif typ == "init_weight":#初始化权重
            return self.M1

    def get_loader(self, batch_size=1):
        train_dataset = TrainData(img=self.Y, target=self.A, transform=transforms.Compose([]))
        train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False)
        return train_loader
