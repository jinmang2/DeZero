if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable
# import dezero's simple_core explicitly
import dezero


def rosenbrock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2+ (x0 - 1) ** 2
    return y


x0 = Variable(np.array(0.0))
x1 = Variable(np.array(2.0))

# Calculate Rosenbrock function's derivative
y = rosenbrock(x0, x1)
y.backward()
print(x0.grad, x1.grad) # -2.0 400.0

x0.cleargrad()
x1.cleargrad()

# Implement Gradient Descent
lr = 0.001
iters = 10000

print(x0, x1) # variable(0.0) variable(2.0)
for i in range(iters):

    y = rosenbrock(x0, x1)

    x0.cleargrad()
    x1.cleargrad()
    y.backward()

    x0.data -= lr * x0.grad
    x1.data -= lr * x1.grad
print(x0, x1)
# iters =   1000: variable(0.6837118569138317) variable(0.4659526837427042)
# iters = 10000: variable(0.9944984367782456) variable(0.9890050527419593)
