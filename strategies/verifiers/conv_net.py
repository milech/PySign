# -----------------------------------------------------------
# Handwritten signature verification system using DTW and CNN
#
#
# (C) 2022 Michal Lech, Gdynia, Poland
# Released under GNU General Public License v3.0 (GPL-3.0)
# email: mlech.ksm@gmail.com
# -----------------------------------------------------------

import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.__conv1 = nn.Conv2d(3, 6, 5)
        self.__pool = nn.MaxPool2d(2, 2)
        self.__conv2 = nn.Conv2d(6, 16, 5)
        self.__fc1 = nn.Linear(16*5*5, 120)
        self.__fc2 = nn.Linear(120, 84)
        self.__fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.__pool(F.relu(self.__conv1(x)))
        x = self.__pool(F.relu(self.__conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.__fc1(x))
        x = F.relu(self.__fc2(x))
        x = self.__fc3(x)
        return x
