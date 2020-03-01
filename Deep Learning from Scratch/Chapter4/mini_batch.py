#%%
import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist
# %%
# mini-batch
""" 当训练集很庞大时可以随机抽取一些进行训练 """
(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
normalize=False, one_hot_label=True)
print('训练集的维度为:{0}'.format(x_train.shape))
print(t_train[0])
#print('训练集第一个元素的标签为:{1}'.format(t_train[0]))
# %%
# 随机抽取100笔数据
train_size = x_train.shape[0]
batch_size = 100

# random.choice随机选取
a = np.random.choice(train_size, batch_size)

x_batch = x_train[a]
t_batch = t_train[a]

# %%
# 看一下取得第3个元素
from PIL import Image

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()


img3_data = x_batch[2]
img3_label = t_batch[2]

img3_data = img3_data.reshape(28, 28)
img_show(img3_data)

# 看一下是不是对应的
print(img3_label)

# %%
