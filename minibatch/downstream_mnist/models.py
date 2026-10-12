"""MNIST-from-scratch models. ResNet depths include 1 stem conv and 1 FC layer.
ResNet16: 1 + 2*(2+2+3) + 1 = 16 layers; ResNet32: 1 + 2*(5+5+5) + 1 = 32.
Neither uses pretrained weights; 1-channel 28x28 input, 10-class output.
"""
import torch
from torch import nn


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                                          nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        shortcut = self.shortcut(x)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return self.relu(x + shortcut)


class SmallResNet(nn.Module):
    def __init__(self, stage_blocks: tuple[int, int, int]):
        super().__init__()
        self.in_channels = 16
        self.stem = nn.Sequential(nn.Conv2d(1, 16, 3, 1, 1, bias=False),
                                  nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.layer1 = self._layer(16, stage_blocks[0], 1)
        self.layer2 = self._layer(32, stage_blocks[1], 2)
        self.layer3 = self._layer(64, stage_blocks[2], 2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, 10)

    def _layer(self, channels: int, count: int, stride: int):
        blocks = [BasicBlock(self.in_channels, channels, stride)]
        self.in_channels = channels
        blocks += [BasicBlock(channels, channels) for _ in range(count - 1)]
        return nn.Sequential(*blocks)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return self.fc(self.pool(x).flatten(1))


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
                                    nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 10))

    def forward(self, x):
        return self.layers(x)

class MNISTTransformer(nn.Module):
    def __init__(
        self,
        image_size=28,
        patch_size=4,
        in_channels=1,
        num_classes=10,
        embed_dim=128,
        depth=4,
        num_heads=4,
        mlp_ratio=4.0,
        dropout=0.1,
    ):
        super().__init__()

        assert image_size % patch_size == 0

        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size

        # 1x28x28 -> embed_dim x 7 x 7
        self.patch_embed = nn.Conv2d(
            in_channels,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
        )

        self.cls_token = nn.Parameter(
            torch.zeros(1, 1, embed_dim)
        )

        self.pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.num_patches + 1,
                embed_dim,
            )
        )

        self.pos_drop = nn.Dropout(dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=int(embed_dim * mlp_ratio),
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=depth,
        )

        self.norm = nn.LayerNorm(embed_dim)

        self.head = nn.Linear(
            embed_dim,
            num_classes,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(
            self.cls_token,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.pos_embed,
            std=0.02,
        )

        nn.init.trunc_normal_(
            self.head.weight,
            std=0.02,
        )

        nn.init.zeros_(self.head.bias)

    def forward(self, x):
        # x: [B, 1, 28, 28]

        x = self.patch_embed(x)
        # [B, embed_dim, 7, 7]

        x = x.flatten(2)
        # [B, embed_dim, 49]

        x = x.transpose(1, 2)
        # [B, 49, embed_dim]

        cls = self.cls_token.expand(
            x.size(0),
            -1,
            -1,
        )

        x = torch.cat(
            [cls, x],
            dim=1,
        )

        x = x + self.pos_embed

        x = self.pos_drop(x)

        x = self.encoder(x)

        x = self.norm(x[:, 0])

        x = self.head(x)

        return x
        
def create_model(name: str) -> nn.Module:
    if name == "resnet16":
        return SmallResNet((2, 2, 3))
    if name == "resnet32":
        return SmallResNet((5, 5, 5))
    if name == "mlp":
        return MLP()
    if name == "transformer":
        return MNISTTransformer()
    raise ValueError(f"Unknown model {name}; expected resnet16,resnet32,mlp")
