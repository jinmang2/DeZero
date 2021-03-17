import numpy as np
from overrides import overrides


class Variable:
    def __init__(self, data):
        if data is not None and not isinstance(data, np.ndarray):
            raise TypeError('{} is not supported'.format(type(data)))

        self.data = data
        self.grad = None
        self.creator = None
        self.generation = 0 # (1) Instance variable which records the number of generation

    def set_creator(self, func):
        self.creator = func
        self.generation = func.generation + 1 # (1) Record the generation (parents + 1) (1)

    def cleargrad(self):
        self.grad = None

    def backward(self):
        if self.grad is None:
            self.grad = np.ones_like(self.data)

        funcs = [] # (3) Variable.backward method
        seen_set = set()

        def add_func(f):
            if f not in seen_set:
                funcs.append(f)
                seen_set.add(f)
                funcs.sort(key=lambda x: x.generation)

        add_func(self.creator)

        while funcs:
            f = funcs.pop() # Difficult to handle complex computational graphs
            gys = [output.grad for output in f.outputs]
            gxs = f.backward(*gys)
            if not isinstance(gxs, tuple):
                gxs = (gxs,)

            for x, gx in zip(f.inputs, gxs):
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx

                if x.creator is not None:
                    add_func(x.creator) # Before: funcs.append(x.creator)


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

        self.generation = max([x.generation for x in inputs]) # (1)
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
    # (2) Dummy DeZero Test
    generations = [2, 0, 1, 4, 2]
    funcs = []
    for g in generations:
        f = Function() # Dummy function class
        f.generation = g
        funcs.append(f)
    print([f.generation for f in funcs]) # [2, 0, 1, 4, 2]
    funcs.sort(key=lambda x: x.generation) # Sort list
    print([f.generation for f in funcs]) # [0, 1, 2, 2, 4]
    f = funcs.pop() # Pop the largest value
    print(f.generation) # 4

    # (4) Operation check
    x = Variable(np.array(2.0))
    a = square(x)
    y = add(square(a), square(a))
    y.backward()

    print(y.data) # 36.0
    print(x.grad) # 64.0
