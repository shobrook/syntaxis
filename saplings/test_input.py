import module

def compose(g, f):
    def h(x):
        return g(f(x))
    return h

def func1(t):
  return t.foo

def func2(t):
  return t.bar

convert = compose(func1, func2)
convert(module.attr)

# def foo():
#     x = 10
#     y = 20
#     def bar():
#         print(x)
#         print(y)
#         print(ayy)
#
#     return bar
