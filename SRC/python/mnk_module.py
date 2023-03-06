import numpy as np
import torch
data = np.loadtxt("/share/home/wanghongyu/learn/test/cublas/test/2473_on_g04.out", delimiter=',')
device = torch.device("cuda:0")
step = 1
data_train1 = np.array([1, 10, 220, 60]).reshape(1, -1)
data_test1 = np.array([1, 10, 220, 60]).reshape(1, -1)
len = 1
len1 = 0
len2 = 0
len_test1 = 1
len_train1 = 1
for d in data:
    if (step%2):
        d1 = d
    else:
        d2 = d
    if (step%2 == 0):
        if (d1[4] < d2[4]):
            # print("openblas speed cublas")
            d3 = np.delete(d1, -1)
            # print(d3)
            if (len % 10):
                temp = np.array(data_train1, dtype=np.float32)
                data_train1 = np.append(temp, d3.reshape(1, -1), axis=0)
                len_train1 += 1
            else:
                temp = np.array(data_test1, dtype=np.float32)
                data_test1 = np.append(temp, d3.reshape(1, -1), axis=0)
                len_test1 +=1
            # print(data_train1[1])
            len1 += 1
        else:
            # print("cublas speed openblas")
            d4 = np.delete(d2, -1)
            # print(d4)
            if (len % 10):
                temp = np.array(data_train1, dtype=np.float32)
                data_train1 = np.append(temp, d4.reshape(1, -1), axis=0)
                len_train1 += 1
            else:
                temp = np.array(data_test1, dtype=np.float32)
                data_test1 = np.append(temp, d4.reshape(1, -1), axis=0)
                len_test1 += 1
            # print(d2)
            len2 += 1
        len += 1
    step+=1
# print("len1 = {}, len2 = {}".format(len1, len2))
# Dataset
from torch.nn import Module
from torch.utils.data import Dataset
import torch

class MKNData(Dataset):
    def __init__(self, data_train1, len):
        self.len = len
        self.data = data_train1

    def __getitem__(self, idx):
        # idx = np.random.rand(1) * 2 * self.len % self.len
        data11 = self.data[int(idx)]
        target = torch.from_numpy((np.array([data11[0] - 1])).astype(np.float32))
        target = torch.squeeze(target)
        target = target.long()
        data = torch.from_numpy((np.delete(data11, 0)).astype(np.float32))
        return target, data
    
    def __len__(self):
        return self.len

data_MKN = MKNData(data_train1, len_train1)
data_MKN_test = MKNData(data_test1, len_test1)
#DataLoader
from torch.utils.data import DataLoader
data_loader = DataLoader(data_MKN, batch_size = 1, shuffle=True)
data_loader_test = DataLoader(data_MKN_test, batch_size = 1, shuffle=True)
# for data in data_loader:
#     print(data)
    
## Module()
from torch.nn import Linear, Sequential,ReLU
class MNK_Moudle(Module):
    def __init__(self):
        super().__init__()
        self.Seq = Sequential(
            Linear(3, 64),
            torch.nn.Sigmoid(),
            Linear(64, 10),
            torch.nn.Sigmoid(),
            Linear(10, 2),
            torch.nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.Seq(x)
        return x

mnk_module = MNK_Moudle()
#test Module
# y = torch.Tensor([1, 2, 3])
# out = mnk_module(y)
# print(out)

#loss opt
from torch.optim import SGD
loss = torch.nn.CrossEntropyLoss()
opt = SGD(params = mnk_module.parameters(), lr = 0.01)

times = 2
for epoch in range(times):
    for data in data_loader:
        target, mnk = data
        #forward
        mnk = mnk_module(mnk)
        # if target == 1:
            # print(11111)
        #backward
        opt.zero_grad()
        loss_result = loss(mnk, target)
        # print("loss = ", loss_result)
        loss_result.backward()
        opt.step()

    #test
    
    with torch.no_grad():
        acc1 = 0
        len_acc = 0
        for data in data_loader_test:
            target, mnk = data
            # if target == 1:
            #     print(11111)
            out = mnk_module(mnk)
            out = torch.argmax(out)
            if (out.item() == target.item()):
                acc1 += 1
            # else:
                # print(len_acc)
            # print(out , target.item())
            len_acc += 1
            # print(len_acc)
    print(acc1)
    print("epoch {} accurary: {}".format(epoch, acc1 / len_acc))

torch.save(mnk_module.state_dict(), "mnk.module.pth")
