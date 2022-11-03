import numpy as np
import engine
import __init__
import plotting

a = engine.Value(3)
b = engine.Value(2) + engine.Value(2)
c = a + b
print(a, b, c)
c.backward()
print(c.grad, a.grad, b.grad)
d = engine.Value(5)
e = c + d
print(e.get_all_children())
print(e.backward())
print(e.grad, d.grad, c.grad, a.grad, b.grad)
f = b + e
print(f.get_all_children())
print(c.get_all_children())
print(e.get_all_children())
print(e.get_all_children())
print(f.backward())
print(f.grad, e.grad, d.grad, c.grad, a.grad, b.grad)

a = engine.Value(3)
b = engine.Value(5)
c = engine.Value(10)
d = (a * b * c) + b
d.backward()
print(a, b, c, d)

a = engine.Value(3)
b = a*a
b.backward()
print(a, b)

a = engine.Value(3)
b = a ** 2
b.backward()
print(a, b)

for a in [engine.Value(3), engine.Value(-3)]:
    b = a.relu() * 2.3
    b.backward()
    print(a, b)

x = [engine.Value(xi) for xi in np.linspace(-5, 5)]
y = [xi.relu() for xi in x]
for yi in y:
    yi.backward()

x_data = [xi.data for xi in x]
y_data = [yi.data for yi in y]
dydx_data = [xi.grad for xi in x]

plotting.plot(
    plotting.Line(x_data, y_data, c="b", label="Relu"),
    plotting.Line(x_data, dydx_data, c="r", label="Relu gradient"),
    plot_name="Relu",
)

x = [engine.Value(xi) for xi in np.linspace(-5, 5)]
y = [xi.sigmoid() for xi in x]
for yi in y:
    yi.backward()

x_data = [xi.data for xi in x]
y_data = [yi.data for yi in y]
dydx_data = [xi.grad for xi in x]

plotting.plot(
    plotting.Line(x_data, y_data, c="b", label="Sigmoid"),
    plotting.Line(x_data, dydx_data, c="r", label="Sigmoid gradient"),
    plot_name="Sigmoid",
)
