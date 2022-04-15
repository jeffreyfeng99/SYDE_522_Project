import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: let's just start with two layers. If we want to try deeper, we could try residual connections?
class DNNet(nn.Module):
    def __init__(self):
        super(DNNet, self).__init__()
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output