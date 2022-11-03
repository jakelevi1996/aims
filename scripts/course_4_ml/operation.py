import math
import engine

class _Operation:
    def __init__(self, *values):
        self._input_values = values
        self.output_value = engine.Value(self._get_output_data(*values))
        self.output_value.add_children(*values)
        self.output_value.set_operation(self)

    def _get_output_data(self, *values):
        raise NotImplementedError()

    def apply_gradient(self):
        raise NotImplementedError()

    def __repr__(self):
        s = "%s(%s)" % (type(self).__name__, ", ".join(self._input_values))
        return s

class Sum(_Operation):
    def _get_output_data(self, *values):
        return sum(v.data for v in values)

    def apply_gradient(self):
        for v in self._input_values:
            v.grad += self.output_value.grad

class Product(_Operation):
    def _get_output_data(self, v1, v2):
        return v1.data * v2.data

    def apply_gradient(self):
        v1, v2 = self._input_values
        v1.grad += self.output_value.grad * v2.data
        v2.grad += self.output_value.grad * v1.data

class Power(_Operation):
    def _get_output_data(self, v1, v2):
        return pow(v1.data, v2.data)

    def apply_gradient(self):
        v1, v2 = self._input_values
        output_partial = self.output_value.grad * self.output_value.data
        v1.grad += v2.data / v1.data * output_partial
        v2.grad += math.log(abs(v1.data)) * output_partial
