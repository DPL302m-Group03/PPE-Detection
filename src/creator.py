import torch

def make_anchors(X, strides, offset=0.5):
    assert X is not None
    anchor_tensor, stride_tensor = [], []
    dtype, device = X[0].dtype, X[0].device
    
    for x, stride in zip(X, strides):
        print(x.shape)
        _, _, h, w = x.shape # LƯU Ý: Dòng này đã được thêm vào dựa trên logic code
        
        sx = torch.arange(end=w, device=device, dtype=dtype) + offset
        sy = torch.arange(end=h, device=device, dtype=dtype) + offset
        
        sy, sx = torch.meshgrid(sy, sx, indexing='ij') # LƯU Ý: Cần thêm indexing='ij' cho Pytorch >= 1.10
        
        print(torch.stack((sx, sy), -1).view(-1, 2).shape)
        print(torch.stack((sx, sy), -1).view(-1, 2))
        print(torch.full((h * w, 1), stride, dtype=dtype, device=device).shape)
        print(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        print('---')
        anchor_tensor.append(torch.stack((sx, sy), -1).view(-1, 2))
        stride_tensor.append(torch.full((h * w, 1), stride, dtype=dtype, device=device))
        
    return torch.concat(anchor_tensor, dim=0), torch.concat(stride_tensor, dim=0)