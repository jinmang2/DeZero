if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
from dezero import Variable, Function
# import dezero's simple_core explicitly
import dezero


class Sin(Function):
    def forward(self, x):
        y = np.sin(x)
        return y

    def backward(self, gy):
        x = self.inputs[0].data
        """
        현재의 DeZero는 아래 gradient 계산에 대해 어떠한 계산 그래프도 만들지 않음
        """
        gx = gy * np.cos(x)
        """
        아래 gx는 원래부터가 y=sin(x)의 미분! 이를 x로 또 미분한다면?
        >>> gx.backward() # 2nd derivative
        """
        return gx


def sin(x):
    return Sin()(x)


if __name__ == '__main__':
    x = Variable(np.array(1.0))
    y = sin(x)
    y.backward(retain_grad=True)
