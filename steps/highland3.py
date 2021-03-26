if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable, Function
# import dezero's simple_core explicitly
import dezero
from dezero.utils import plot_dot_graph


def f(x):
    return x ** 4 - 2 * x ** 2


if __name__ == '__main__':
    # (1) Calculate 2nd derivative
    x = Variable(np.array(2.0))
    y = f(x)
    y.backward(create_graph=True)
    print(x.grad) # variable(24.0)
    gx = x.grad
    x.cleargrad()
    gx.backward()
    print(x.grad) # variable(44.0)

    # (2) Optimization using Newton's method
    x = Variable(np.array(2.0))
    iters = 10

    for i in range(iters):
        print(i, x)

        y = f(x)
        x.cleargrad()
        y.backward(create_graph=True)

        gx = x.grad
        x.cleargrad()
        gx.backward()
        gx2 = x.grad

        x.data -= gx.data / gx2.data

    """
    0 variable(2.0)
    1 variable(1.4545454545454546)
    2 variable(1.1510467893775467)
    3 variable(1.0253259289766978)
    4 variable(1.0009084519430513)
    5 variable(1.0000012353089454)
    6 variable(1.000000000002289)
    7 variable(1.0)
    8 variable(1.0)
    9 variable(1.0)
    """
