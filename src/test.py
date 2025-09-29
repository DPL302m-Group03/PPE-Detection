import torch

x = torch.arange(2*2*2).reshape(2, 3, 3)  # shape (2,6)
# x1 = torch.arange(3*4).reshape(2, 2, 3) 
# # assert(x.shape == x1.shape)
# print(x)
# print(x1)
# print(torch.cat([x, x1], dim=0))
# chia theo dim=1, mỗi khối có 2 phần tử
x1, x2 = x.chunk(2, dim=0)
print(x1)
print(x2)
print(x1.shape)
print(x2.shape)
outputs = [x1, x2]
print([x1, x2])

# x = torch.cat(outputs, dim=0)
# print(x)
# print(x.shape)

# a = [1, 2, 3]
# a.insert(0, 9)
# print(a)