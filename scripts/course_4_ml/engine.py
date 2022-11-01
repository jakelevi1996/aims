import math

class Value:
    """ stores a single scalar value and its gradient """

    def __init__(self, data, _op=''):
        self.data = data
        self.grad = 0
        # internal variables used for autograd graph construction
        self._backward = lambda: None # this will be called on the backward pass
        self._children = set()
        self._op = _op # the op that produced this node, for graphviz / debugging / etc

    def add_child(self, other):
        self._children.add(other)

    def _to_value(self, other):
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._to_value(other)
        out_data = self.data + other.data
        out = Value(out_data, '+')
        out.add_child(self)
        out.add_child(other)

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = self._to_value(other)
        out_data = self.data * other.data
        out = Value(out_data, '*')
        out.add_child(self)
        out.add_child(other)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def __pow__(self, other):
        other = self._to_value(other)
        out_data = self.data ** other.data
        out = Value(out_data, '^')
        out.add_child(self)
        out.add_child(other)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def relu(self):
        out_data = max(self.data, 0)
        out = Value(out_data, 'relu')
        out.add_child(self)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def sigmoid(self):
        out_data = 1.0 / (1.0 + math.exp(self.data))
        out = Value(out_data, 'sigmoid')
        out.add_child(self)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def cos(self):
        out_data = math.cos(self.data)
        out = Value(out_data, 'cos')
        out.add_child(self)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def sin(self):
        out_data = math.sin(self.data)
        out = Value(out_data, 'sin')
        out.add_child(self)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def backward(self):

        # topological order all of the children in the graph
        topo = []
        visited = set()
        def build_topo(v):
            # this needs to fill topo with the nodes in a topological ordering of the graph
            # so we can visit them and call their _backward() in the correct order

            # for p in self._prev:

            raise NotImplementedError
        build_topo(self)

        # go one variable at a time and apply the chain rule to get its gradient
        self.grad = 1
        for v in reversed(topo):
            v._backward()

    def __neg__(self): # -self
        out_data = -self.data
        out = Value(out_data, '-')
        out.add_child(self)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def __radd__(self, other): # other + self
        other = self._to_value(other)
        out_data = self.data + other.data
        out = Value(out_data, '+')
        out.add_child(self)
        out.add_child(other)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def __sub__(self, other): # self - other
        other = self._to_value(other)
        out_data = self.data - other.data
        out = Value(out_data, '-')
        out.add_child(self)
        out.add_child(other)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def __rsub__(self, other): # other - self
        other = self._to_value(other)
        out_data = other.data - self.data
        out = Value(out_data, '-')
        out.add_child(self)
        out.add_child(other)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def __rmul__(self, other): # other * self
        other = self._to_value(other)
        out_data = other.data * self.data
        out = Value(out_data, '*')
        out.add_child(self)
        out.add_child(other)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def __truediv__(self, other): # self / other
        other = self._to_value(other)
        out_data = self.data / other.data
        out = Value(out_data, '/')
        out.add_child(self)
        out.add_child(other)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def __rtruediv__(self, other): # other / self
        other = self._to_value(other)
        out_data = other.data / self.data
        out = Value(out_data, '/')
        out.add_child(self)
        out.add_child(other)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
