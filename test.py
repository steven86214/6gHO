t = [1,2,3]
g = []
a = 0
for tt in reversed(t):
    a = tt + 0.1*a
    g.insert(0,a)
print(g)