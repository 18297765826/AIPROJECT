import torch


torch.manual_seed(0)
#构造数据
data = torch.randn(2, 3, 4, 5)
# 构造四维张量数据
print(data.shape)
print(data)
print("===============================")


print(data.mean(dim=[0, 2, 3]))  # 计算每个通道的均值
print(data.std(dim=[0, 2, 3]))   # 计算每个通道的标准差
print("===============================")
print("===============================")
print(data[:, 0, :, :])  # 打印第一通道的数据
print(data[:, 1, :, :])  # 打印第二通道的数据
print(data[:, 2, :, :])  # 打印第三通道的数据
print("===============================")
# 构造批归一化层
batch_norm = torch.nn.BatchNorm2d(num_features=3)
# 对数据进行批归一化
normalized_data = batch_norm(data)
print(normalized_data.shape)
print(normalized_data)

