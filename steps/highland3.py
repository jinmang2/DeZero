if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable, Function
# import dezero's simple_core explicitly
import matplotlib.pyplot as plt
import dezero
import dezero.functions as F
from dezero.utils import plot_dot_graph


if __name__ == '__main__':
    x = Variable(np.array(2.0))
    y = x ** 2
    y.backward(create_graph=True)
    gx = x.grad
    x.cleargrad()

    z = gx ** 3 + y
    z.backward()
    print(x.grad)
