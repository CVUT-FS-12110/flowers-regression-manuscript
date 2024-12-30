import torch
import torch.nn as nn

from torchvision.models import resnet50, ResNet50_Weights

class MultiViewNetwork(nn.Module):

    def __init__(self):
        super(MultiViewNetwork, self).__init__()
        print("Standard model")

        self.resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.fc_in_features = self.resnet.fc.in_features
        self.resnet = torch.nn.Sequential(*(list(self.resnet.children())[:-1]))

        embed_dim = 1

        num_features = 256
        self.fc1 = nn.Sequential(
            nn.Linear(self.fc_in_features, num_features * 2),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(num_features * 2, num_features),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(num_features, embed_dim),
            nn.ReLU(),
        )

        self.fc1.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain=0.1)
            m.bias.data.fill_(0.01)

    def forward_once(self, x):
        output = self.resnet(x)
        output = output.view(output.size()[0], -1)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        num1 = self.fc1(output1)
        num2 = self.fc1(output2)

        output = num1 + num2

        return output

