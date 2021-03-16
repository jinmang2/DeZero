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
            x, y = f.input, f.output
            x.grad = f.backward(y.grad)

            if x.creator is not None:
                funcs.append(x.creator)


def as_array(x):
    if np.isscalar(x):
        return np.array(x)
    return x


class Function:
    # (1) Easy to use function
    def __call__(self, *inputs):
        xs = [x.data for x in inputs]
        # (2) Easy to use function
        ys = self.forward(*xs)
        if not isinstance(ys, tuple):
            ys = (ys,)
        outputs = [Variable(as_array(y)) for y in ys]

        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = outputs
        # (1) Easy to use function
        return outputs if len(outputs) > 1 else outputs[0]

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()


class Add(Function):
    @overrides
    # (2) Easy to implement function
    def forward(self, x0, x1):
        y = x0 + x1
        return y


# (3) Implement Add function
def add(x0, x1):
    return Add()(x0, x1)


if __name__ == '__main__':
    """
    기존의 test code
    xs = [Variable(np.array(2)), Variable(np.array(3))]
    f = Add()
    ys = f(xs)
    y = ys[0]
    """
    x0 = Variable(np.array(2))
    x1 = Variable(np.array(3))
    y = add(x0, x1)
    print(y.data) # 5
