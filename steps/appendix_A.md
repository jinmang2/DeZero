# In-place Operation (APPENDIX A)

## A.1 Check problem
- If we write the code `x.grad += gx`, why is the problem with the following code?
```python
class Variable:
    ...
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
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad = x.grad + gx # Here!

                if x.creator is not None:
                    funcs.append(x.creator)
```

## A.2 Copy and Overwrite
- Copy vs Overwrite?
- Overwrite, in other words `in-place operation`
```python
>>> import numpy as np
>>> x = np.array(1)
>>> id(x)
4370746224

>>> x += x # Overwrite
>>> id(x)
4370746224

>>> x = x + x # Copy(Directly created)
>>> id(x)
4377585368
```

## A.3 DeZero's backpropagation?
- Change `Variable.backward` to '' in-place operation'
```python
class Variable:
    ...
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
                if x.grad is None:
                    x.grad = gx
                else:
                    x.grad += gx # Here!

                if x.creator is not None:
                    funcs.append(x.creator)
```

Execute following codes
```python
x = Variable(np.array(3))
y = add(x, x)
y.backward()

print(f"y.grad: {y.grad} ({id(y.grad)})")
print(f"y.grad: {x.grad} ({id(x.grad)})")
```
```
y.grad: 2 (4427494384)
x.grad: 2 (4427494384)
```
