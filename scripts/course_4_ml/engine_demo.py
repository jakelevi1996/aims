import engine

a = engine.Value(3)
b = engine.Value(4)
c = a + b
print(a, b, c)
print(c.backward())
d = engine.value(5)
e = c + d
print(e.backward())
