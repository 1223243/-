import time
from getdata import DogsVSCatsDataset as DVCD
from torch.utils.data import DataLoader as DataLoader
from network import Net
import torch
from torch.autograd import Variable
import torch.nn as nn
from matplotlib import pyplot as plt
from torchvision import models

# loss下降曲线的绘制
def plot_curve(data):
    fig = plt.figure()
    plt.plot(range(len(data)), data, color='blue')
    plt.legend(['value'], loc='upper right')
    plt.xlabel('step')
    plt.ylabel('value')
    plt.show()

dataset_dir = 'D:/Python_code/DogsvsCats/data/'   # 数据集路径
model_cp = 'D:/Python_code/猫狗大战/model/'    # 网络参数保存位置
workers = 10                 # PyTorch读取数据线程数量
batch_size = 20              # batch_size大小
lr = 0.0001                  # 学习率
nepoch = 1                   # 训练次数

def train():
    since = time.time()
    datafile = DVCD('train', dataset_dir)  # 实例化一个数据集
    dataloader = DataLoader(datafile, batch_size=batch_size, shuffle=True, num_workers=workers, drop_last=True)

    print('Dataset loaded! length of train set is {0}'.format(len(datafile)))

    # model=Net()
    # ------------------GPU-----------------------------------------------
    # model=model.cude()
    model=models.vgg16(pretrained=True)
    # model = model.cude()
    for param in model.features.parameters():param.requires_grad=False   #冻结卷积层的权重和参数
    model.classifier[6].out_features=2       #将最后一层(第6层)的输出特征从1000->2

    model.train()          # 网络设定为训练模式，有两种模式可选，.train()和.eval()，训练模式和评估模式，区别就是训练模式采用了dropout策略，可以放置网络过拟合

    criterion=nn.CrossEntropyLoss()                          # 定义loss计算方法，cross entropy，交叉熵，可以理解为两者数值越接近其值越小
    # optimizer = torch.optim.Adam(model.parameters(), lr=lr)  # 实例化一个优化器，即调整网络参数，优化方式为adam方法
    optimizer=torch.optim.Adam(model.classifier.parameters(),lr=lr)     # 实例化一个优化器，即调整网络参数(只调整分类层)，优化方式为adam方法
    cnt=0               # 训练图片数量
    train_loss = []     # 将loss 保存起来
    for epoch in range(nepoch):
        # 读取数据集中数据进行训练，因为dataloader的batch_size设置为16，所以每次读取的数据量为16，即img包含了16个图像，label有16个
        for img,label in dataloader:
            img, label = Variable(img), Variable(label)
            #------------------GPU-----------------------------------------------
            # img,label=Variable(img).cuda(),Variable(label).cuda()
            out=model(img)                      # 计算网络输出值，就是输入网络一个图像数据，输出猫和狗的概率，调用了网络中的forward()方法
            loss=criterion(out,label.squeeze()) # 计算损失，也就是网络输出值和实际label的差异，显然差异越小说明网络拟合效果越好，此处需要注意的是第二个参数，必须是一个1维Tensor
            loss.backward()                     # 误差反向传播，采用求导的方式，计算网络中每个节点参数的梯度，显然梯度越大说明参数设置不合理，需要调整
            optimizer.step()                    # 优化采用设定的优化方法对网络中的各个参数进行调整
            optimizer.zero_grad()               # 清除优化器中的梯度以便下一次计算，因为优化器默认会保留，不清除的话，每次计算梯度都回累加
            cnt+=1                              # 训练图片数量+1
            train_loss.append(loss.item())      # 存储loss值
            print('Epoch:{0},Frame:{1}, train_loss {2}'.format(epoch, cnt * batch_size, loss / batch_size))#loss / batch_size=每一个batch的平均loss
            if cnt==100:
                break

        torch.save(model.state_dict(),'{0}/model.pth'.format(model_cp)) # 训练所有数据后，保存网络的参数

    # 用图像打印出loss 的改变 需要用到utils里面的工具
    plot_curve(train_loss)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

if __name__=='__main__':
    train()
















