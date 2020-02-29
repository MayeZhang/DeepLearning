#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
# 定义类
class Man:
    def __init__(self, name):
        self.name = name
        print("Initialized!")

    def hello(self):
        print('Hello' + self.name + '!')

    def goodbye(self):
        print('Good-bye' + self.name + '!')

m = Man("Maye Zhang")

m.hello()
m.goodbye()

# %%
# 定义函数
def nihao():
    print('hi,nihao')

nihao()

# %%
x = np.linspace(0, 2*np.pi, 50)

#for i in x:
#    print(i)

y1 = np.sin(x)
plt.plot(x, y1, '--', label='sin')

y2 = np.cos(x)
plt.plot(x, y2, label='cos')

plt.xlabel('$x$')
plt.ylabel('$y$')

plt.title('sin & cos')
plt.legend()
plt.show()
# %%
# 转化为一维数组
a = np.random.randn(2, 3)
print(a)

a1 = a.flatten()
print(a1)
# %%


