import engine

a = engine.Value(3)
b = engine.Value(4)
c = a + b
print(a, b, c)
c.backward()
print(c.grad, a.grad, b.grad)
d = engine.Value(5)
e = c + d
print(e.get_all_children())
print(e.backward())
print(e.grad, d.grad, c.grad, a.grad, b.grad)
