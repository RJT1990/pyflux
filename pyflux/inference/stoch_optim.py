import numpy as np

class RMSProp(object):
    """
    *** RMSProp *** 

    Computes adaptive learning rates for each parameter. Has an EWMA of squared gradients.
    """
    def __init__(self, starting_parameters, starting_variance, learning_rate, ewma):
        self.parameters = starting_parameters
        self.variance = starting_variance
        self.learning_rate = learning_rate
        self.ewma = ewma
        self.epsilon = np.power(10.0,-8)    
        self.t = 1

    def update(self, gradient):
        self.variance = self.ewma*self.variance + (1-self.ewma)*np.power(gradient,2)
        if self.t > 5:
            self.parameters += (self.learning_rate+(self.learning_rate*15.0*(0.990**self.t)))*(gradient/np.sqrt(self.variance+self.epsilon))  
        self.t += 1
        return self.parameters

class ADAM(object):
    """
    *** Adaptive Moment Estimation (ADAM) *** 

    Computes adaptive learning rates for each parameter. Has an EWMA of past gradients and squared gradients.
    """
    def __init__(self, starting_parameters, starting_variance, learning_rate, ewma_1, ewma_2):
        self.parameters = starting_parameters
        self.f_gradient = 0.0
        self.variance = starting_variance
        self.learning_rate = learning_rate
        self.ewma_1 = ewma_1
        self.ewma_2 = ewma_2

        self.epsilon = np.power(10.0,-8)        
        self.t = 1

    def update(self, gradient):
        self.f_gradient = self.ewma_1*self.f_gradient + (1-self.ewma_1)*gradient
        f_gradient_hat = self.f_gradient / (1-np.power(self.ewma_1,self.t))
        self.variance = self.ewma_2*self.variance + (1-self.ewma_2)*np.power(gradient,2)
        variance_hat = self.variance / (1-np.power(self.ewma_2,self.t))
        if self.t > 5:
            self.parameters += (self.learning_rate+(self.learning_rate*15.0*(0.990**self.t)))*(f_gradient_hat/(np.sqrt(variance_hat)+self.epsilon))  
        self.t += 1
        return self.parameters