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
    x = Variable(np.array(1.0))
    y = F.tanh(x)
    x.name = 'x'
    y.name = 'y'
    y.backward(create_graph=True)

    iters = 7

    for i in range(iters):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    # Figure computational graph
    gx = x.grad
    gx.name = 'gx' + str(iters+1)
    plot_dot_graph(gx, verbose=False, to_file=f'tanh_iters{iters+1}.png')
