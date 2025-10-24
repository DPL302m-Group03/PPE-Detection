import torch.nn as nn
import torch
from processor import yolo_type

class Conv(nn.Module):
    # Standard convolution
    def __init__(
            self, 
            in_channels, out_channels, 
            kernel_size=3, stride=2, padding=1, 
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
        self.m = nn.ModuleList(
            [Bottleneck(self.mid_channels, self.mid_channels, shortcut) for _ in range(n_bottlenecks)]
        )
        self.cv2 = Conv((n_bottlenecks + 2) * out_channels // 2, out_channels, kernel_size=1, stride=1, padding=0)
                    
    def forward(self, x):
        x = self.cv1(x)
        x1, x2 = x.chunk(2, dim=1)

        for bottleneck in self.m:
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

class Concat(nn.Module):
    def __init__(self, dimension=1):
        super().__init__()
        self.dimension = dimension

    def forward(self, x):
        return torch.cat(x, dim=self.dimension)

class DFL(nn.Module):
    def __init__(self, bins=16):
        super().__init__()
        self.bins = bins
        self.conv = nn.Conv2d(bins, 1, kernel_size=1, bias=False)
        
        bin_w = torch.arange(bins, dtype=torch.float32).reshape(1, bins, 1, 1)
        self.conv.weight = torch.nn.Parameter(bin_w, requires_grad=False)

    def forward(self, x):
        b, B, c = x.shape
        x = x.view(b, 4, self.bins, c).transpose(1, 2)
        x = x.softmax(1)
        x = self.conv(x)
        x = x.view(b, 4, c)
        return x

class Detect(nn.Module):
    def __init__(self, type='s', bins=16, n_classes=80):
        super().__init__()
        self.bins = bins
        self.n_classes = n_classes
        self.coordinates = 4 * bins 
        self.no = self.coordinates + n_classes 

        self.stride = torch.zeros(3)

        d, w, r = yolo_type(type)  # 'n', 's', 'm', 'l', 'x'
        # Box
        self.cv2 = nn.ModuleList([
            nn.Sequential(
                Conv(int(256*w), self.coordinates, kernel_size=3, stride=1, padding=1),
                Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)
            ),

            nn.Sequential(
                Conv(int(512*w), self.coordinates, kernel_size=3, stride=1, padding=1),
                Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)
            ),

            nn.Sequential(
                Conv(int(512*w*r), self.coordinates, kernel_size=3, stride=1, padding=1),
                Conv(self.coordinates, self.coordinates, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.coordinates, self.coordinates, kernel_size=1, stride=1)
            )
        ])    

        # Class
        self.cv3 = nn.ModuleList([
            nn.Sequential(
                Conv(int(256*w), self.n_classes, kernel_size=3, stride=1, padding=1),
                Conv(self.n_classes, self.n_classes, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1, stride=1)
            ),

            nn.Sequential(
                Conv(int(512*w), self.n_classes, kernel_size=3, stride=1, padding=1),
                Conv(self.n_classes, self.n_classes, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1, stride=1)
            ),

            nn.Sequential(
                Conv(int(512*w*r), self.n_classes, kernel_size=3, stride=1, padding=1),
                Conv(self.n_classes, self.n_classes, kernel_size=3, stride=1, padding=1),
                nn.Conv2d(self.n_classes, self.n_classes, kernel_size=1, stride=1)
            )
        ]) 

        self.dfl = DFL(bins=16)

    def forward(self, x):
        for i in range(len(self.cv2)):
            box = self.cv2[i](x[i])
            cls = self.cv3[i](x[i])
            x = torch.cat([box, cls], dim=1)

        if self.training:
            return x
        
        anchors, strides = (i.transpose(0, 1) for i in self.make_anchors(x, self.stride))     
        
class DetectionModel(nn.Module):
    def __init__(self, path=None, type='s'):
        super().__init__()
        if path:
            type = path.split('_')[0][-1]
            state_dict = torch.load(path)
            self.load_state_dict(state_dict, strict=False)  # strict=False nếu không khớp 100% tên layer
            print("Loaded state_dict into custom model!")
        
        d, w, r = yolo_type(type)  # 'n', 's', 'm', 'l', 'x'
        self.model = nn.Sequential(
            Conv(3, int(64*w), kernel_size=3, stride=2, padding=1),                # 0
            Conv(int(64*w), int(128*w), kernel_size=3, stride=2, padding=1),       # 1
            C2f(int(128*w), int(128*w), n_bottlenecks=max(int(3*d), 1)),           # 2
            Conv(int(128*w), int(256*w), kernel_size=3, stride=2, padding=1),      # 3
            C2f(int(256*w), int(256*w), n_bottlenecks=max(int(6*d), 1)),           # 4
            Conv(int(256*w), int(512*w), kernel_size=3, stride=2, padding=1),      # 5
            C2f(int(512*w), int(512*w), n_bottlenecks=max(int(6*d), 1)),           # 6
            Conv(int(512*w), int(512*w*r), kernel_size=3, stride=2, padding=1),    # 7
            C2f(int(512*w*r), int(512*w*r), n_bottlenecks=max(int(3*d), 1)),       # 8
            SPPF(int(512*w*r), int(512*w*r), kernel_size=5),                       # 9
            nn.Upsample(scale_factor=2, mode='nearest'),                           # 10
            Concat(dimension=1),                                                   # 11
            C2f(int(512*w*(1 + r)), int(512*w), n_bottlenecks=max(int(3*d), 1)),   # 12  
            nn.Upsample(scale_factor=2, mode='nearest'),                           # 13
            Concat(dimension=1),                                                   # 14
            C2f(int(768*w), int(256*w), n_bottlenecks=max(int(3*d), 1)),           # 15
            Conv(int(256*w), int(256*w), kernel_size=3, stride=2, padding=1),      # 16
            Concat(dimension=1),                                                   # 17
            C2f(int(768*w), int(512*w), n_bottlenecks=max(int(3*d), 1)),           # 18
            Conv(int(512*w), int(512*w), kernel_size=3, stride=2, padding=1),      # 19
            Concat(dimension=1),                                                   # 20
            C2f(int(512*w*(1 + r)), int(512*w*r), n_bottlenecks=max(int(3*d), 1)), # 21
            Detect(type, bins=16)                                                               # 22
        )

    def forward(self, x):
        return self.model(x)
    
class YOLO(nn.Module):
    def __init__(self, path=None):
        super().__init__()
        self.model = DetectionModel(path)

    def forward(self, x):
        return super().forward(x)
# ...existing code...

if __name__ == "__main__":
    # DetectionModel = YOLO(path='weights\yolov8s_state_dict.pt')
    # print("Loaded state_dict into custom model!")
    # print(DetectionModel.model)

    # # In tổng số parameters
    # total_params = sum(p.numel() for p in DetectionModel.model.parameters())
    # print(f"Tổng số parameters: {total_params:,}")

    model = DetectionModel()
    print(model.model)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Tổng số parameters: {total_params:,}")