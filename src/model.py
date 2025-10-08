import torch.nn as nn
import torch

class Conv(nn.Module):
    # Standard convolution
    def __init__(
            self, 
            in_channels, out_channels, 
            kernel_size=3, stride=1, padding=1, 
            activation=True,
            eps=0.001, momentum=0.03
        ):  
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum, affine=True, track_running_stats=True)
        self.act = nn.SiLU(inplace=True) if activation else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))
    
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
            self, 
            in_channels, out_channels, 
            shortcut=True
        ):
        super().__init__()
        self.cv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.cv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = shortcut 
        
    def forward(self, x):
        x_origin = x
        x = self.cv1(x)
        x = self.cv2(x)
        if self.shortcut:
            x += x_origin
        return x

class C2f(nn.Module):
    def __init__(
            self, 
            in_channels, out_channels, 
            n_bottlenecks=1, 
            shortcut=True
        ):
        super().__init__()
        self.mid_channels = out_channels // 2
        self.n_bottlenecks = n_bottlenecks

        self.cv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bottlenecks = nn.ModuleList(
            [Bottleneck(self.mid_channels, self.mid_channels, shortcut) for _ in range(n_bottlenecks)]
        )
        self.cv2 = Conv((n_bottlenecks + 2) * out_channels // 2, out_channels, kernel_size=1, stride=1, padding=0)
                    
    def forward(self, x):
        x = self.cv1(x)
        x1, x2 = x.chunk(2, dim=1)

        for bottleneck in self.bottlenecks:
            x1 = bottleneck(x1)
            x = torch.cat([x, x1], dim=1)
        x = self.cv2(x)
        return x

class SPPF(nn.Module):
    def __init__(
            self,
            in_channels, out_channels,
            kernel_size=5
        ):
        super().__init__()
        self.mid_channels = in_channels // 2
        self.cv1 = Conv(in_channels, self.mid_channels, kernel_size=1, stride=1, padding=0)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2, ceil_mode=False, dilation=1)
        self.cv2 = Conv(self.mid_channels * 4, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.cv1(x)
        y = x
        for _ in range(3):
            y = self.m(y)
            x = torch.cat([x, y], dim=1)
        x = self.cv2(x)
        return x

class DetectionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            Conv(3, 16, kernel_size=3, stride=2, padding=1),
            Conv(16, 32, kernel_size=3, stride=2, padding=1),
            C2f(32, 64, n_bottlenecks=2),
            SPPF(64, 128, kernel_size=5),
            C2f(128, 256, n_bottlenecks=3),
            SPPF(256, 512, kernel_size=5)
        )

    def forward(self, x):
        return self.model(x)
    
if __name__ == "__main__":
    c2f = C2f(64, 128, n_bottlenecks=2)
    print(f"{sum(p.numel() for p in c2f.parameters())/1e6} parameters")

    x = torch.randn(1, 64, 32, 32)
    y = c2f(x)
    print(y.shape)

    sppf = SPPF(128, 256, kernel_size=5)
    print(f"{sum(p.numel() for p in sppf.parameters())/1e6} parameters")

    y1 = sppf(y)
    print(y1.shape)