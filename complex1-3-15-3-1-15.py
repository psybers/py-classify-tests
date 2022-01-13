class MyNumbers:
  x = 1

  def m(self):
    def m3():
      return 1
    x = m3()
    return x

  def __iter__(self):
    self.a = 1
    return self

  def __next__(self):
    x = self.a
    self.a += 1
    return x

x = MyNumbers()
