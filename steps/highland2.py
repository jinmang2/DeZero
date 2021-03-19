import weakref
import numpy as np
from overrides import overrides


class Variable:
    def __init__(self, data):
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1

    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = []
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop()
            gys = [output().grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator)


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

        self.generation = max([x.generation for x in inputs])
        for output in outputs:
            output.set_creator(self)
        self.inputs = inputs
        self.outputs = [weakref.ref(output) for output in outputs]
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
    # (4) Weakref Test
    a = np.array([1, 2, 3])
    b = weakref.ref(a)
    print(b) # <weakref at 0x00000204A92B06D8; to 'numpy.ndarray' at 0x00000204A92B0670>
    print(b()) # [1 2 3]

    a = None
    print(b) # <weakref at 0x000001C268550688; dead>

    # (5) Operation check
    for i in range(10):
        # 덮어 씌우면서 이전의 계산 그래프를 참조하지 않게 됨
        x = Variable(np.random.randn(10000))
        y = square(square(square(x)))
