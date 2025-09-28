import torch

x = torch.arange(4*3*3).reshape(4, 3, 3)  # shape (2,6)
print(x)
print(x.shape)
# chia theo dim=1, mỗi khối có 2 phần tử
x1, x2 = x.chunk(2, dim=0)
print(x1)
print(x2)
print(x1.shape)
print(x2.shape)
outputs = [x1, x2]
print([x1, x2])

x = torch.cat(outputs, dim=0)
print(x)
print(x.shape)

# a = [1, 2, 3]
# a.insert(0, 9)
# print(a)