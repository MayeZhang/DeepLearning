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
# %%
# MNIST数据集
import sys, os
sys.path.append(os.pardir)
# from dataset.mnist import * 
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
normalize=False, one_hot_label=False)

print(x_train.shape)

img = x_train[2]
label = t_train[2]

print(label)
print(img.shape)

img = img.reshape(28, 28)

print(img.shape)
img_show(img)

# %%
# 导入网络参数
import pickle

# 参数是以字典的形式存储的
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network

# 预测函数
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    #第一层
    z1 = np.dot(x, W1) + b1
    a1 = sigmoid(z1)

    #第二层
    z2 = np.dot(a1, W2) + b2
    a2 = sigmoid(z2)

    #第三层
    z3 = np.dot(a2, W3) + b3
    y = softmax(z3)

    return y

# %%
# 批处理
print(x_train.shape, x_test.shape)
# (60000, 784), (10000, 784)

batch_size = 100 # 批数量
accuracy_cnt = 0 # 正确预测的数量
network = init_network()

for i in range(0, len(x_test), batch_size):
    y = predict(network, x_test[i:i+batch_size, :])
    p = np.argmax(y, axis=1)
    accuracy_cnt += np.sum(p == t_test[i:i+batch_size])
    

print('准确率为:{0}%'.format(accuracy_cnt/(x_test.shape[0])*100))


# %%

# %%

# %%
