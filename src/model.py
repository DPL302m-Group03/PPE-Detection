import torch.nn as nn
import torch

class Conv(nn.Module):
    # Standard convolution
    def __init__(
            self, 
            in_channels, out_channels, 
            kernel_size=3, stride=1, padding=1, 
            groups=1, 
            activation=True,
            eps=0.001, momentum=0.03
        ):  
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, out_channels,
            kernel_size, stride, padding,
            groups=groups, bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=eps, momentum=momentum)
        self.act = nn.SiLU() if activation else nn.Identity()

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
        self.conv1 = Conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = shortcut 
        
    def forward(self, x):
        x_origin = x
        x = self.conv1(x)
        x = self.conv2(x)
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

        self.conv1 = Conv(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.bottlenecks = nn.ModuleList(
            [Bottleneck(self.mid_channels, self.mid_channels) for _ in range(n_bottlenecks)]
        )
        self.conv2 = Conv((n_bottlenecks + 2) * out_channels // 2, out_channels, kernel_size=1, stride=1, padding=0)
                    
    def forward(self, x):
        x = self.conv1(x)
        x1, x2 = x.chunk(self.mid_channels, dim=1)

        for bottleneck in self.bottlenecks:
            x1 = bottleneck(x1)
            x = torch.cat([x, x1], dim=1)
        
        x = self.conv2(x)
        return x