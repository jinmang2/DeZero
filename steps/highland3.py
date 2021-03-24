if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import math
import numpy as np
from dezero import Variable, Function
from dezero.utils import plot_dot_graph


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        return gy * np.cos(x)


def sin(x):
    return Sin()(x)


x = Variable(np.array(np.pi / 4))
y = sin(x)
y.backward()
print('--- original sin ---')
print(y.data) # 0.7071067811865476
print(x.grad) # 0.7071067811865476


def my_sin(x, threshold=0.0001):
    y = 0
    for i in range(100000):
        c = (-1) ** i / math.factorial(2 * i + 1)
        t = c * x ** (2 * i + 1)
        y = y + t
        if abs(t.data) < threshold:
            break
    return y


x.grad = None
y = my_sin(x, threshold=1e-150)
y.backward()
print('--- approximate sin ---')
print(y.data) # 0.7071064695751781 | 0.7071067811865475
print(x.grad) # 0.7071032148228457 | 0.7071067811865475

x.name = 'x'
y.name = 'y'
plot_dot_graph(y, verbose=False, to_file='my_sin_1e-150.png')
