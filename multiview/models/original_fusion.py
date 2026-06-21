import torch
import torch.nn as nn


class OriginalFeatureFusionNetwork(nn.Module):
    """Fusion ablations using the original repo's feature extractors."""

    def __init__(self, use_fpn=True, fusion="concat"):
        super().__init__()
        self.use_fpn = use_fpn
        self.fusion = fusion

        if use_fpn:
            from models.nn_fpn import MultiViewNetwork
        else:
            from models.nn import MultiViewNetwork

        self.base = MultiViewNetwork()
        in_features = self._feature_dim()

        if fusion == "concat":
            self.head = self._make_head(in_features * 2)
        elif fusion == "attention":
            heads = self._attention_heads(in_features)
            self.attention = nn.MultiheadAttention(
                embed_dim=in_features,
                num_heads=heads,
                batch_first=True,
            )
            self.head = self._make_head(in_features)
        else:
            raise ValueError(f"Unsupported fusion strategy '{fusion}'")

    def _feature_dim(self):
        if self.use_fpn:
            return self.base.fc[0].in_features
        return self.base.fc_in_features

    def _make_head(self, in_features):
        return nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 1),
        )

    def _attention_heads(self, in_features):
        for heads in (8, 4, 2, 1):
            if in_features % heads == 0:
                return heads
        return 1

    def forward(self, input1, input2):
        features1 = self.base.extract_features(input1)
        features2 = self.base.extract_features(input2)

        if self.fusion == "concat":
            return self.head(torch.cat([features1, features2], dim=1))

        tokens = torch.stack([features1, features2], dim=1)
        attended, _ = self.attention(tokens, tokens, tokens)
        return self.head(attended.mean(dim=1))
