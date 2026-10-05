"""Scalar autograd engine and a small neural network library on top of it.

Grown out of micrograd.ipynb (after Andrej Karpathy's micrograd). Everything is
plain Python and the math module: every weight is a Value, every operation
records how to push a gradient back to its inputs, and backward() walks that
graph in reverse.

Three things go beyond the notebook, all of them needed for MNIST:

- relu, log and a cross-entropy loss built from exp and log
- Value.weighted_sum, one graph node for sum(w_i * x_i) + b. Built from single
  * and + nodes a 784-128-10 network creates about 200,000 Values per image,
  which costs 0.6 s per image. As one node per neuron it takes 10 ms and
  computes exactly the same gradients.
- an iterative topological sort, because the recursive one runs into Python's
  recursion limit on graphs this deep
"""

import math
import random


class Value:
    """A scalar that remembers how it was computed."""

    __slots__ = ("data", "grad", "_backward", "_prev", "_op", "label")

    def __init__(self, data, _children=(), _op='', label=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = _children
        self._op = _op
        self.label = label

    def __repr__(self):
        return f"Value(data={self.data})"

    # ------------------------------------------------------------------
    # Arithmetic
    # ------------------------------------------------------------------
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad

        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, other):
        assert isinstance(other, (int, float))
        out = Value(self.data**other, (self, ), f'**{other}')

        def _backward():
            self.grad += (other * self.data**(other-1)) * out.grad

        out._backward = _backward
        return out

    def __neg__(self): # -self
        return self * -1

    def __sub__(self, other): # self - other
        return self + (-other)

    def __truediv__(self, other): # self / other
        return self * other**-1

    def __radd__(self, other): # other + self
        return self + other

    def __rsub__(self, other): # other - self
        return other + (-self)

    def __rmul__(self, other): # other * self
        return self * other

    def __rtruediv__(self, other): # other / self
        return other * self**-1

    # ------------------------------------------------------------------
    # Functions and activations
    # ------------------------------------------------------------------
    def exp(self):
        x = self.data
        out = Value(math.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad

        out._backward = _backward
        return out

    def log(self):
        x = self.data
        out = Value(math.log(x), (self, ), 'log')

        def _backward():
            self.grad += (1 / x) * out.grad

        out._backward = _backward
        return out

    def tanh(self):
        t = math.tanh(self.data)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1 - t**2) * out.grad

        out._backward = _backward
        return out

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self, ), 'relu')

        def _backward():
            # Slope 1 where the input was positive, 0 everywhere else.
            if out.data > 0:
                self.grad += out.grad

        out._backward = _backward
        return out

    @staticmethod
    def weighted_sum(weights, inputs, bias):
        """sum(w_i * x_i) + bias as a single node instead of 2n small ones.

        inputs are either all Values (activations of the previous layer) or all
        plain numbers (pixels). Numbers need no gradient, so they do not become
        part of the graph.
        """
        if isinstance(inputs[0], Value):
            xs = [x.data for x in inputs]
            children = (*weights, *inputs, bias)
        else:
            xs = inputs
            inputs = None
            children = (*weights, bias)

        total = bias.data
        for w, x in zip(weights, xs):
            total += w.data * x
        out = Value(total, children, 'wsum')

        def _backward():
            grad = out.grad
            if grad == 0.0:
                return  # e.g. behind a ReLU that is switched off: nothing to add
            # d(out)/d(w_i) = x_i, d(out)/d(x_i) = w_i, d(out)/d(bias) = 1
            for w, x in zip(weights, xs):
                w.grad += x * grad
            if inputs is not None:
                for x, w in zip(inputs, weights):
                    x.grad += w.data * grad
            bias.grad += grad

        out._backward = _backward
        return out

    # ------------------------------------------------------------------
    # Backpropagation
    # ------------------------------------------------------------------
    def backward(self):
        """Fill .grad of every Value this one depends on with d(self)/d(value)."""
        # Topological order by depth-first search with an explicit stack. A node
        # is pushed twice: first to visit its children, then (done=True) to be
        # appended once all of them are in the list.
        topo = []
        visited = set()
        stack = [(self, False)]

        while stack:
            v, done = stack.pop()
            if done:
                topo.append(v)
            elif v not in visited:
                visited.add(v)
                stack.append((v, True))
                for c in v._prev:
                    if c not in visited:
                        stack.append((c, False))

        self.grad = 1.0
        for node in reversed(topo):
            node._backward()


# ----------------------------------------------------------------------
# Neural network building blocks
# ----------------------------------------------------------------------
class Neuron:

    def __init__(self, nin, activation='relu'):
        # Same initialisation as torch.nn.Linear: uniform in +-1/sqrt(fan_in).
        bound = 1 / math.sqrt(nin)
        self.w = [Value(random.uniform(-bound, bound)) for _ in range(nin)]
        self.b = Value(random.uniform(-bound, bound))
        self.activation = activation

    def __call__(self, x):
        # w * x + b
        act = Value.weighted_sum(self.w, x, self.b)
        if self.activation == 'relu':
            return act.relu()
        if self.activation == 'tanh':
            return act.tanh()
        return act

    def predict(self, x):
        """Forward pass on plain floats, without building a graph."""
        act = self.b.data
        for wi, xi in zip(self.w, x):
            act += wi.data * xi
        if self.activation == 'relu':
            return act if act > 0 else 0.0
        if self.activation == 'tanh':
            return math.tanh(act)
        return act

    def parameters(self):
        return self.w + [self.b]


class Layer:

    def __init__(self, nin, nout, activation='relu'):
        self.neurons = [Neuron(nin, activation) for _ in range(nout)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def predict(self, x):
        return [n.predict(x) for n in self.neurons]

    def parameters(self):
        return [p for neuron in self.neurons for p in neuron.parameters()]


class MLP:
    """Fully connected network. Hidden layers use the activation, the last one is linear."""

    def __init__(self, nin, nouts, activation='relu'):
        sz = [nin] + nouts
        self.layers = [
            Layer(sz[i], sz[i+1], activation if i < len(nouts) - 1 else None)
            for i in range(len(nouts))
        ]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x):
        """Logits as plain floats. The counterpart of torch.no_grad() for evaluation."""
        for layer in self.layers:
            x = layer.predict(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]


# ----------------------------------------------------------------------
# Loss and optimizer
# ----------------------------------------------------------------------
def cross_entropy(logits, target):
    """Softmax followed by negative log-likelihood, for one sample.

    loss = -log(exp(z_target) / sum_j exp(z_j)) = log(sum_j exp(z_j)) - z_target

    The largest logit is subtracted first. Softmax does not change when every
    logit is shifted by the same constant, but exp() can no longer overflow.
    """
    shift = max(logit.data for logit in logits)
    shifted = [logit - shift for logit in logits]
    exps = [z.exp() for z in shifted]
    return sum(exps[1:], exps[0]).log() - shifted[target]


class SGD:
    """Plain stochastic gradient descent: p <- p - lr * dL/dp."""

    def __init__(self, parameters, lr):
        self.parameters = parameters
        self.lr = lr

    def zero_grad(self):
        for p in self.parameters:
            p.grad = 0.0

    def step(self):
        for p in self.parameters:
            p.data -= self.lr * p.grad
