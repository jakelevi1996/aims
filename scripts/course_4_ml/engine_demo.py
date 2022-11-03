import engine

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
