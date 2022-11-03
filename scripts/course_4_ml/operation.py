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

    def apply_gradient(self, *values):
        for v in self._input_values:
            v.grad += self.output_value.grad
