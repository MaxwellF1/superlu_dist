import sys
import torch
import time
from torch.nn import Linear, Sequential,ReLU, Module
t1 = time.time()
#定义出相同的模型
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

#创建模型例子
mnk = MNK_Moudle()
#加载模型参数
mnk.load_state_dict(torch.load("/share/home/wanghongyu/learn/pexsi/superlu_dist/SRC/python/mnk.module.pth"))

mnk.eval()
a1 = int(sys.argv[1])
a2 = int(sys.argv[2])
a3 = int(sys.argv[3])
device = torch.device("cuda:0")


x = torch.tensor([a1, a2, a3], dtype=torch.float32)
print(torch.argmax(mnk(x)).item())
print(time.time() - t1)
