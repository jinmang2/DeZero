import numpy as np
from overrides import overrides


class Variable:
    def __init__(self, data):
        self.data = data


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(y)
        return output

    def forward(self, x):
        raise NotImplementedError()


class Square(Function):
    @overrides
    def forward(self, x):
        return x ** 2


if __name__ == '__main__':
    x = Variable(np.array(10))
    f = Square()
    y = f(x)
    print(type(y)) # <class '__main__.Variable'>
    print(y.data) # 100
