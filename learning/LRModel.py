# 生成数据
X = [0, 1, 2, 3, 4]
Y = [2*x + 1 for x in X]

print(X)
print(Y)

#初始化参数
w= 0.0
b= 0.0

def forward(x):
    return w * x + b


for x in X:
    print(f"预测值: {forward(x):.2f}, 真实值: {Y[X.index(x)]}")

# 定义损失函数（单个样本）
def loss_fn(y_pred, y_true):
    return (y_pred - y_true) ** 2

# 测试 loss
for x, y in zip(X, Y):
    y_pred = forward(x)
    loss = loss_fn(y_pred, y)
    print(f"x={x}, y={y}, y_pred={y_pred}, loss={loss}")

# 学习率（控制步子大小）
lr = 0.01

# 训练一轮（遍历所有数据）
for epoch in range(10):  # 先跑10轮看看
    total_loss = 0

    for x, y in zip(X, Y):
        # 1. 前向
        y_pred = forward(x)

        # 2. 计算loss
        loss = loss_fn(y_pred, y)
        total_loss += loss

        # 3. 计算梯度
        dw = 2 * (y_pred - y) * x
        db = 2 * (y_pred - y)

        # # 4. 更新参数（重点！！！）
        # global w, b
        w = w - lr * dw
        b = b - lr * db

    print(f"epoch={epoch}, loss={total_loss}")
print(f"训练完成后参数: w={w:.2f}, b={b:.2f}")