#%%
import numpy as np 
import matplotlib.pyplot as plt 
# %%
# 激活函数

# 1.阶跃函数
def step_function(x):
    y = x > 0
    return y.astype(int)

# 第三个参数表示产生数据个数
x = np.linspace(-5.0, 5.0, 100)
y_step = step_function(x)


# 2.sigmoid
def sigmoid(x):
    return 1 / (1+np.exp(-x))

y_sigmoid = sigmoid(x)


# 3.ReLU
def relu(x):
    return np.maximum(0, x)

y_relu = relu(x)


# 绘制图形
plt.figure()
plt.plot(x, y_step, '--', label='u')
plt.plot(x, y_sigmoid, 'k', label='Simoid function')
plt.plot(x, y_relu, label='ReLU function')

plt.ylim([-0.1, 1.1])
plt.legend()
plt.show()

# %%
# softmax函数
def softmax(a):
    return np.exp(a) / np.sum(np.exp(a))

a1 = [1, 2, 3] 
y1 = softmax(a1)
print(np.sum(y1))