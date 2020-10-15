from autograd.numpy.random import normal, uniform
from autograd.numpy import exp
from autograd import grad

class NNode_int():

    def __init__(self, n_neurons, n_int, x_min, x_max, X0, Y0):
        self.n_neurons = n_neurons
        self.p = normal(size=3*(self.n_neurons))
        self.x_min = x_min
        self.x_max = x_max
        self.X0 = X0
        self.Y0 = Y0
        self.n_int = n_int
        self.loss = []

        self.f = None
    
    #sigmoid function
    def SGM (self, t):
        #to normalizate the data
        x = 2 * t / (self.x_max - self.x_min) - 1
        return 1 /(1 + exp(-x))

    #derivate of sigmoid function
    def dSGMdx (self, t):
        #to normalizate the data
        x = 2 * t/(self.x_max - self.x_min) - 1
        S=self.SGM(x)
        return S*(1-S)*(2/(self.x_max - self.x_min))

    # predictor function
    def N(self, x):
        r = 0
        for i in range (self.n_neurons):
            r += self.p[i+2*self.n_neurons]*self.SGM(self.p[i]*x + self.p[i+self.n_neurons])
        return r

    #derivate of predictor function
    def dNdx (self, x):
        r=0
        for i in range (self.n_neurons):
            r += self.p[i+2*self.n_neurons]*self.p[i]*self.dSGMdx(self.p[i]*x + self.p[i+self.n_neurons])
        return r

    # trial solutions   
    def TS(self, x):
        return self.Y0 + (x-self.X0)*self.N(x)

    # derivate of trial solutions
    def dTSdx (self, x):
        return self.N(x) + (x - self.X0)*self.dNdx(x)

    def E (self, p):
        p_temp = self.p
        self.p = p
        er = 0
        xi = uniform(low=self.x_min, high=self.x_max, size=(self.n_int,))
        for i in xi:
            er += (self.dTSdx(i) - self.f(i,self.TS(i)))**2
        self.p = p_temp
        return er

    # fit
    def fit(self, iterations, max_learning_rate = 0.001, tol = 1e-3):
        grad_E = grad(self.E)

        p1 = normal(size=3*(self.n_neurons))

        best_value = self.E(p1)
        best_p     = p1
        current_value = 1e100

        for i in range(iterations):
            if best_value > tol:
                p1 -= grad_E(p1) * uniform(0, max_learning_rate)
                current_value = self.E(p1)
                self.loss.append(current_value)
                if current_value < best_value:
                    best_p = p1
                    best_value = current_value
            else:
                break
        self.p = best_p