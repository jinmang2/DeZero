import numpy as np
from overrides import overrides


class Variable:
    def __init__(self, data):
        # (3) Only handling ndarray
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError('{} is not supported'.format(type(data)))
            
        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        # (2) Simplify backward method
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop() # 1. Get a function
            x, y = f.input, f.output # 2. Get the function's input/output
            x.grad = f.backward(y.grad) # 3. Call the function's backward

            if x.creator is not None:
                funcs.append(x.creator)


# (3) Only handling ndarray
def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)
        output = Variable(as_array(y)) # (3) Only handling ndarray
        output.set_creator(self)
        self.input = input
        self.output = output
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    @overrides
    def forward(self, x):
        return x ** 2

    @overrides
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx


class Exp(Function):
    @overrides
    def forward(self, x):
        return np.exp(x)

    @overrides
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx


# ======================
# (1) Using python function
def square(x):
    return Square()(x)


def exp(x):
    return Exp()(x)
# ======================


def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)
    return (y1.data - y0.data) / (2 * eps)


if __name__ == '__main__':
    # (1) Using Python function
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.grad = np.array(1.0)
    y.backward()
    print(x.grad) # 3.297442541400256

    # (2) Simplify backward method
    x = Variable(np.array(0.5))
    y = square(exp(square(x)))
    y.backward()
    print(x.grad) # 3.297442541400256

    # (3) Only handling ndarray
    x = Variable(np.array(1.0))  # OK
    x = Variable(None)  # OK
    x = Variable(1.0)  # NG
