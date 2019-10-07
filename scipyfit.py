from scipy.optimize import leastsq
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


def f(xy, p):
    x = xy[0]
    y = xy[1]
    return p[0]*x*x + p[1]*y*y + p[2]*x + p[3]*y + p[4]


def nihe(p, y, x):
    # print(f(x,p))
    return y - f(x,p)


p0 = [1, 1, 1, 1, 1]
# tend:mlp:lgb
# x = np.array([(0, 0.3), (0, 0.5), (0, 0.7), (0.9, 0.1), (0.5, 0.5), (0.3, 0.7)])
x = np.array([[0,0,0,0.9,0.5,0.3],[0.3,0.5,0.7,0.1,0.5,0.7]])
D = np.array([0.5505, 0.5556, 0.5507, 0.5508, 0.5659, 0.5572])
p = leastsq(nihe, p0, args=(D, x))[0]

X = np.arange(0, 1, 0.01)
Y = np.arange(0, 1, 0.01)

z = []
for x in X:
    for y in Y:
        if x + y <= 1:
            z.append(f([x,y],p))
            if f([x,y],p) > 0.5663:
                print(x,y,f([x,y],p))
print(max(z))

