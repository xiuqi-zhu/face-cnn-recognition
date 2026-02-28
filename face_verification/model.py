"""
ConvNeXt-based backbone and head for face classification and verification.
Outputs logits and normalized features (for verification via cosine similarity).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNorm(nn.Module):
    """Layer norm supporting channels_last and channels_first."""

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


class ConvNeXtBlock(nn.Module):
    """ConvNeXt block: depthwise conv + layer norm + MLP with residual."""

    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + x
        return x


class ConvNeXtTiny_Balanced(nn.Module):
    """ConvNeXt-Tiny style backbone (balanced channel widths)."""

    def __init__(self, in_chans=3, drop_path_rate=0.1):
        super().__init__()
        dims = [88, 176, 352, 704]
        depths = [3, 3, 9, 3]

        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first"),
        )
        self.downsample_layers.append(stem)

        for i in range(3):
            down = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(down)

        self.stages = nn.ModuleList()
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        for i in range(4):
            blocks = [
                ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j])
                for j in range(depths[i])
            ]
            self.stages.append(nn.Sequential(*blocks))
            cur += depths[i]
        self.out_dim = dims[-1]

    def forward(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return x


class Network(nn.Module):
    """
    Single model for both tasks: classification logits and embedding for verification.
    forward() returns dict with 'out' (logits) and 'feats' (embedding before classifier).
    """

    def __init__(self, num_classes):
        super().__init__()
        self.backbone = ConvNeXtTiny_Balanced(in_chans=3)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.dropout = nn.Dropout(p=0.2)
        self.cls_layer = nn.Linear(704, num_classes)

    def forward(self, x):
        feats = self.backbone(x)
        pooled = self.pool(feats)
        flattened = self.flatten(pooled)
        flattened = self.dropout(flattened)
        out = self.cls_layer(flattened)
        return {"feats": flattened, "out": out}
