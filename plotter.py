from matplotlib import pyplot as plt
from numpy import linspace

def plot_error(loss):
    x = [i for i in range(len(loss))]
    plt.plot(x, loss)

    plt.title("Fit process")
    plt.xlabel("Iterations")
    plt.ylabel("Error")
    plt.show()

def plot_result(y1, r,f=None):
    if f != None:
        plt.plot(r, [f(m[1], 0) for m in enumerate(r)] , "black")
    plt.plot(r, y1, "red")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("ODE Interpolation")
    plt.show()
