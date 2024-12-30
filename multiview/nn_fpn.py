
from torchvision.models import resnet50, ResNet50_Weights

import torch
import torch.nn as nn
import torch.nn.functional as F



class MultiViewNetwork(nn.Module):

    def __init__(self):
        super(MultiViewNetwork, self).__init__()
        print("FPN model")

        # Load ResNet backbone
        base_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
        self.resnet_layers = nn.ModuleDict({
            'conv1': base_model.conv1,
            'bn1': base_model.bn1,
            'relu': base_model.relu,
            'maxpool': base_model.maxpool,
            'layer1': base_model.layer1,  # ResNet Block 1
            'layer2': base_model.layer2,  # ResNet Block 2
            'layer3': base_model.layer3,  # ResNet Block 3
            'layer4': base_model.layer4,  # ResNet Block 4
        })

        # Define lateral and top-down FPN layers
        self.fpn_c5_to_p4 = nn.Conv2d(2048, 256, kernel_size=1)  # Reduce channels
        self.fpn_c4_to_p4 = nn.Conv2d(1024, 256, kernel_size=1)
        self.fpn_c3_to_p3 = nn.Conv2d(512, 256, kernel_size=1)
        self.fpn_c2_to_p2 = nn.Conv2d(256, 256, kernel_size=1)

        self.fpn_p4_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.fpn_p3_upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.fpn_p2_upsample = nn.Upsample(scale_factor=2, mode="nearest")

        # Fully connected layers after global pooling
        self.fc = nn.Sequential(
            nn.Linear(2816, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )


    def forward_once(self, x):
        # Extract features at different scales from ResNet
        c1 = self.resnet_layers['conv1'](x)
        c1 = self.resnet_layers['bn1'](c1)
        c1 = self.resnet_layers['relu'](c1)
        c1 = self.resnet_layers['maxpool'](c1)

        c2 = self.resnet_layers['layer1'](c1)  # [B, 256, H/4, W/4]
        c3 = self.resnet_layers['layer2'](c2)  # [B, 512, H/8, W/8]
        c4 = self.resnet_layers['layer3'](c3)  # [B, 1024, H/16, W/16]
        c5 = self.resnet_layers['layer4'](c4)  # [B, 2048, H/32, W/32]

        # Compute FPN outputs
        p4 = self.fpn_c5_to_p4(c5)
        p4 = self.fpn_p4_upsample(p4) + self.fpn_c4_to_p4(c4)

        p3 = self.fpn_c3_to_p3(c3)
        p3 = self.fpn_p3_upsample(p4) + p3

        p2 = self.fpn_c2_to_p2(c2)
        p2 = self.fpn_p2_upsample(p3) + p2

        # Apply Global Average Pooling to each scale
        p2_gap = F.adaptive_avg_pool2d(p2, (1, 1)).squeeze(-1).squeeze(-1)  # [B, 256]
        p3_gap = F.adaptive_avg_pool2d(p3, (1, 1)).squeeze(-1).squeeze(-1)  # [B, 256]
        p4_gap = F.adaptive_avg_pool2d(p4, (1, 1)).squeeze(-1).squeeze(-1)  # [B, 256]
        c5_gap = F.adaptive_avg_pool2d(c5, (1, 1)).squeeze(-1).squeeze(-1)  # [B, 256]

        # Concatenate features from all scales
        combined_features = torch.cat([p2_gap, p3_gap, p4_gap, c5_gap], dim=1)  # [B, 256*4]

        # Pass through fully connected layers
        output = self.fc(combined_features)

        return output


    def forward(self, input1, input2):
        num1 = self.forward_once(input1)
        num2 = self.forward_once(input2)
        output = num1 + num2
        return output

