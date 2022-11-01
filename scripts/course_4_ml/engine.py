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

    def add_children(self, *children):
        for child in children:
            self._children.add(child)

    def _to_value(self, other):
        return other if isinstance(other, Value) else Value(other)

    def __add__(self, other):
        other = self._to_value(other)
        out_data = self.data + other.data
        out = Value(out_data, '+')
        out.add_children(self, other)

        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward

        return out

    def __mul__(self, other):
        other = self._to_value(other)
        out_data = self.data * other.data
        out = Value(out_data, '*')
        out.add_children(self, other)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def __pow__(self, other):
        other = self._to_value(other)
        out_data = self.data ** other.data
        out = Value(out_data, '^')
        out.add_children(self, other)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def relu(self):
        out_data = max(self.data, 0)
        out = Value(out_data, 'relu')
        out.add_children(self)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def sigmoid(self):
        out_data = 1.0 / (1.0 + math.exp(self.data))
        out = Value(out_data, 'sigmoid')
        out.add_children(self)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def cos(self):
        out_data = math.cos(self.data)
        out = Value(out_data, 'cos')
        out.add_children(self)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def sin(self):
        out_data = math.sin(self.data)
        out = Value(out_data, 'sin')
        out.add_children(self)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def get_all_children(self, visited=None, all_children=None):
        """
        Return a list containing the current node and all it's children,
        topologically ordered so that a node only appears after its children in
        the list, by performing a depth-first search. A node is only added to
        the list once all its children have been added to the list.
        """
        if visited is None:
            visited = set()
        if all_children is None:
            all_children = list()

        for child in self._children:
            child.get_all_children(visited, all_children)

        if self not in visited:
            all_children.append(self)
            visited.add(self)

        return all_children

    def backward(self):
        all_children = self.get_all_children()
        # Reset gradients
        for child in self.get_all_children():
            child.grad = 0

        self.grad = 1
        # go one variable at a time and apply the chain rule to get its
        # gradient. The list returned by self.get_all_children() is ordered
        # such that the current node is last, and every node only appears after
        # its children
        for v in reversed(self.get_all_children()):
            v._backward()

    def __neg__(self): # -self
        out_data = -self.data
        out = Value(out_data, '-')
        out.add_children(self)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def __radd__(self, other): # other + self
        other = self._to_value(other)
        out_data = self.data + other.data
        out = Value(out_data, '+')
        out.add_children(self, other)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def __sub__(self, other): # self - other
        other = self._to_value(other)
        out_data = self.data - other.data
        out = Value(out_data, '-')
        out.add_children(self, other)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def __rsub__(self, other): # other - self
        other = self._to_value(other)
        out_data = other.data - self.data
        out = Value(out_data, '-')
        out.add_children(self, other)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def __rmul__(self, other): # other * self
        other = self._to_value(other)
        out_data = other.data * self.data
        out = Value(out_data, '*')
        out.add_children(self, other)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def __truediv__(self, other): # self / other
        other = self._to_value(other)
        out_data = self.data / other.data
        out = Value(out_data, '/')
        out.add_children(self, other)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def __rtruediv__(self, other): # other / self
        other = self._to_value(other)
        out_data = other.data / self.data
        out = Value(out_data, '/')
        out.add_children(self, other)

        def _backward():
            raise NotImplementedError
        out._backward = _backward

        return out

    def __repr__(self):
        return f"Value(data={self.data}, grad={self.grad})"
