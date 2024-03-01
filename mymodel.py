import torch   
import torch.nn as nn

class net(nn.Module):
    def __init__(self, classes_num):
        super(net, self).__init__()
        self.fc1  = nn.Linear(512, 512)
        self.bn1  = nn.BatchNorm1d(512)
        self.act1 = nn.ReLU()
        self.fc2  = nn.Linear(512, 512)
        self.bn2  = nn.BatchNorm1d(512)
        self.act2 = nn.ReLU()
        self.fc3  = nn.Linear(512, classes_num)
    def forward (self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.fc3(x)
        return x