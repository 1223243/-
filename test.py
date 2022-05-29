from torchvision import models
from getdata import DogsVSCatsDataset as DVCD
from network import Net
import torch
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import random
import os
import getdata

dataset_dir = 'D:/python程序/DogsvsCats/data/test/'                    # 数据集路径
model_file = 'D:/pythonCNN程序/猫狗大战/model/model.pth'                # 模型保存路径
N = 10                                          #一下预测 N 张

def test():

    # setting model
    # model = Net()                                       # 实例化一个网络
    # ------------------GPU---------------------
    # model.cuda()                                        # 送入GPU，利用GPU计算
    # model = nn.DataParallel(model)
    model = models.vgg16(pretrained=True)

    model.load_state_dict(torch.load(model_file))       # 加载训练好的模型参数
    model.eval()                                        # 设定为评估模式，即计算过程中不要dropout

    # get data
    files = random.sample(os.listdir(dataset_dir), N)   # 随机获取N个测试图像
    imgs = []           # img
    imgs_data = []      # img data
    for file in files:
        img = Image.open(dataset_dir + file)            # 打开图像
        img_data = getdata.dataTransform(img)           # 转换成torch tensor数据

        imgs.append(img)                                # 图像list
        imgs_data.append(img_data)                      # tensor list
    imgs_data = torch.stack(imgs_data)                  # tensor list合成一个4D tensor

    # calculation
    out = model(imgs_data)                              # 对每个图像进行网络计算
    out = F.softmax(out, dim=1)                         # 输出概率化
    out = out.data.cpu().numpy()                        # 转成numpy数据

    # pring results         显示结果
    for idx in range(N):
        plt.figure()
        if out[idx, 0] > out[idx, 1]:
            plt.suptitle('cat:{:.1%},dog:{:.1%}'.format(out[idx, 0], out[idx, 1]))
        else:
            plt.suptitle('dog:{:.1%},cat:{:.1%}'.format(out[idx, 1], out[idx, 0]))
        plt.imshow(imgs[idx])
    plt.show()


if __name__ == '__main__':
    test()