from torch import nn
from torchsummary import summary

class ScorePredictionModel(nn.Module):
    def __init__(self):
        super(ScorePredictionModel, self).__init__()
        self.layer_1 = nn.Linear(2, 64, bias=False)
        self.layer_2 = nn.Linear(64, 1, bias=False)
        self.activation = nn.Softmax(dim=1)

    def forward(self, X):
        X = self.layer_1(X)
        X = self.layer_2(X)
        X = self.activation(X)
        return X


# model = ScorePredictionModel()
# summary(model, (len(X), 2))