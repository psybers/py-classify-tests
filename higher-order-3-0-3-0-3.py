def compose(f, g):
    return lambda x: f(g(x))

compose1 = lambda f,g: lambda x: f(g(x))
