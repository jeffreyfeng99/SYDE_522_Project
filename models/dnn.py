import torch
import torch.nn as nn
import torch.nn.functional as F


# TODO: let's just start with two layers. If we want to try deeper, we could try residual connections?
class DNNet(nn.Module):
    def __init__(self, n_features, n_classes):
        super(DNNet, self).__init__()

        self.fc1 = nn.Linear(n_features, 128) # TODO change in_features by target_problem/df
        self.fc2 = nn.Linear(128, 64) # TODO change n_classes by target_problem
        self.fc3 = nn.Linear(64, 32)
        self.fc4 = nn.Linear(32, n_classes)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        x = F.relu(x)
        x = self.fc4(x)
        output = F.softmax(x, dim=1)
        return output