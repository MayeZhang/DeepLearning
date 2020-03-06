#%%
import numpy as np

# %%
# 交叉熵
# 当标签数据以one-hot形式存储时
def cross_entroy_error1(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.shape[0])
        y = y.reshape(1, y.shape[0])
    batch_size = y.shape[0]
    # 要防止y太小溢出
    return -np.sum(t * np.log(y + 1e-7)) / batch_size

# %%
# 当标签数据不是以one-hot形式存储时
# 1.自己的挫code
def cross_entroy_error2(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.shape[0])
        y = y.reshape(1, y.shape[0])
    
    error = 0
    batch_size = y.shape[0]
    for i in range(batch_size):
        error += -np.sum(np.log(y[i, t[i]] + 1e-7))

    return error / batch_size

# %%
# 当标签数据不是以one-hot形式存储时
# 2.书上的code
def cross_entroy_error3(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.shape[0])
        y = y.reshape(1, y.shape[0])
    
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size


# %%
y = np.array([1, 2, 3, 4])
t1 = np.array([0, 0 ,1, 0])
t2 = np.array([2])
t3 = t2

error1 = cross_entroy_error1(y, t1)
error2 = cross_entroy_error2(y, t2)
error3 = cross_entroy_error3(y, t3)


error1, error2, error3
# %%

# %%
