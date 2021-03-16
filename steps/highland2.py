import numpy as np
from overrides import overrides


class Variable:
    def __init__(self, data):
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None

    def set_creator(self, func):
        self.creator = func

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [self.creator]
        while funcs:
            f = funcs.pop()
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                x.grad = gx # 여기가 실수!

                if x.creator is not None:
                    funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Square(Function):
    def forward(self, x):
        y = x ** 2
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        gx = 2 * x * gy
        return gx


def square(x):
    f = Square()
    return f(x)


class Add(Function):
    @overrides
    def forward(self, x0, x1):
        y = x0 + x1
        return y

    @overrides
    def backward(self, gy):
        return gy, gy


def add(x0, x1):
    return Add()(x0, x1)


if __name__ == '__main__':
    x = Variable(np.array(3.0))
    y = add(x, x)
    print('y', y.data)

    y.backward()
    print('x.grad', x.grad)
    """
    >>> y 6.0
    >>> x.grad 1.0
    """
