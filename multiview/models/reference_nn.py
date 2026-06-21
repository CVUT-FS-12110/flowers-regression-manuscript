import torch.nn as nn

from torchvision.models import vgg16, VGG16_Weights


class ReferenceMultiViewNetwork(nn.Module):

    def __init__(self):
        super(ReferenceMultiViewNetwork, self).__init__()

        # Load VGG16 with pretrained weights
        self.vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        # Retain only convolutional layers (Conv5 block outputs 512 channels)
        self.vgg = nn.Sequential(*(list(self.vgg.features.children())))

        # Global Average Pooling (GAP) for feature reduction
        self.global_pooling = nn.AdaptiveAvgPool2d((1, 1))

        factor = 1
        self.fc = nn.Sequential(
            nn.Linear(512 * factor, out_features=256 * factor),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=256 * factor, out_features=128 * factor),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(in_features=128 * factor, out_features=1),
            nn.ReLU(),
        )
        self.fc.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_normal_(m.weight)
            m.bias.data.fill_(0.0)

    def forward_once(self, x):
        x = self.vgg(x)
        x = self.global_pooling(x)
        x = x.view(x.size(0), -1)
        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)

        num1 = self.fc(output1)
        num2 = self.fc(output2)

        return num1 + num2
