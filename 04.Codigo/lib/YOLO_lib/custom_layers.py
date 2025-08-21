import torch
import torch.nn as nn
from ultralytics.nn.modules import C2f

class R_ELAN(nn.Module):
    def __init__(self, c1, c2=None, n=1, shortcut=False, *args, **kwargs):
        super().__init__()
        c2 = c2 or c1
        self.block = C2f(c1, c2, n=n, shortcut=shortcut)
    def forward(self, x):
        return self.block(x)

class C2PSA(nn.Module):
    def __init__(self, c1, c2=None, n=1, shortcut=False, *args, **kwargs):
        super().__init__()
        c2 = c2 or c1
        self.core = C2f(c1, c2, n=n, shortcut=shortcut)
        self.spatial = nn.Sequential(nn.Conv2d(c2, 1, 1), nn.Sigmoid())
    def forward(self, x):
        y = self.core(x)
        return y * self.spatial(y)

class A2(nn.Module):
    def __init__(self, c1, c2=None, r=4, *args, **kwargs):
        super().__init__()
        c2 = c2 or c1
        m = max(c2 // r, 1)
        self.attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c2, m, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(m, c2, 1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.attn(x)