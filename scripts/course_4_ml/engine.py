import math
import operation

class Value:
    def __init__(self, data):
        self.data = data
        self.grad = 0
        self._children = set()
        self._operation = None

    def add_children(self, *children):
        for child in children:
            self._children.add(child)

    def set_operation(self, operation):
        self._operation = operation

    def apply_local_gradient(self):
        if self._operation is not None:
            self._operation.apply_gradient()

    def _to_value(self, other):
        return other if isinstance(other, Value) else Value(other)

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
        for child in self.get_all_children():
            child.grad = 0

        self.grad = 1
        for v in reversed(self.get_all_children()):
            v.apply_local_gradient()

    def __add__(self, other):
        return operation.Sum(self, self._to_value(other)).output_value

    def __mul__(self, other):
        return operation.Product(self, self._to_value(other)).output_value

    def __pow__(self, other):
        return operation.Power(self, self._to_value(other)).output_value

    def relu(self):
        return operation.Relu(self).output_value

    def sigmoid(self):
        return operation.Sigmoid(self).output_value

    def sin(self):
        return operation.Sin(self).output_value

    def cos(self):
        return operation.Cos(self).output_value

    def __neg__(self):
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
        return "Value(data=%s, grad=%s)" % (self.data, self.grad)
