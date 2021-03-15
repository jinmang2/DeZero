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


class Exp(Function):
    @overrides
    def forward(self, x):
        return np.exp(x)


if __name__ == '__main__':
    A = Square()
    B = Exp()
    C = Square()

    x = Variable(np.array(0.5))
    a = A(x)
    b = B(a)
    y = C(b)
    print(y.data) # 1.648721270700128
