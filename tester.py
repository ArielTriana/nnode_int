from nnode_int import NNode_int
from plotter import plot_error, plot_result
from matplotlib import pyplot as plt
import math
from numpy import linspace

r = linspace(0, 2, 1000)
nn = NNode_int(5, 20, 0, 1, 0, 6)
nn.f = lambda x , y : 13*math.sin(2*x) -3*y
nn.fit(1000, max_learning_rate=0.001)
#plot_error(nn.loss)
y1 = [nn.TS(i[1]) for i in enumerate(r)]

plot_error(nn.loss)
plot_result(y1, r, lambda x, y : 8*math.exp(-3*x) - 2*math.cos(2*x) + 3*math.sin(2*x))

