import numpy as np

N = 50

a = np.linspace(-3.14, 3.14, N)

x = np.cos(a) / 1.5
y = np.sin(a) / 1.5

print N*2, 2, 1

for i in range(len(a)):
    print x[i], y[i], 1

x /= 3;
y /= 3;

for i in range(len(a)):
    print x[i], y[i], 0
