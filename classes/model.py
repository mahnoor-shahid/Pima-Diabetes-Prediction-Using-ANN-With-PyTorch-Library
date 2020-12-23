import torch.nn as nn
import torch.nn.functional as F

class ANN_model(nn.Module):

    def __init__(self, input_features=7, hidden1=20, hidden2=20, output_features=2):
        """
        In this constructor nn.Linear modules are instantiated
        """
        super().__init__()
        self.f_connected1 = nn.Linear(input_features, hidden1)
        self.f_connected2 = nn.Linear(hidden1, hidden2)
        self.out= nn.Linear(hidden2, output_features)
    def forward(self, x):
        x = F.relu(self.f_connected1(x))
        x = F.relu(self.f_connected2(x))
        x = self.out(x)
        return x