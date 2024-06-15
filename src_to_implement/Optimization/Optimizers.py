import numpy as np



class Sgd: 
    def __init__(self, learning_rate):
        self.learning_rate= learning_rate

    def calculate_update (self, weight_tensor, gradient_tensor):
        uploeaded_weight= weight_tensor - np.dot(self.learning_rate, gradient_tensor)
        return uploeaded_weight


class SgdWithMomentum:
    def __init__(self,learning_rate:float, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.velocity = None

    def calculate_update(self, weight_tensor, gradient_tensor):

        if self.velocity is None :
            self.velocity = np.zeros_like(weight_tensor)

        self.velocity = (self.momentum_rate * self.velocity) - (self.learning_rate * gradient_tensor)

        return weight_tensor + self.velocity
    

class Adam:
    def __init__(self,learning_rate:float,mu:float,rho:float):
        self.learning_rate = learning_rate
        self.β1 = mu
        self.β2 = rho
        self.v = None
        self.r = None
        self.iteration = 1
        self.epsilon = np.finfo(float).eps

    def calculate_update(self,weight_tensor, gradient_tensor):
        if self.v is None :
            self.v = np.dot((1 - self.β1),gradient_tensor)
        else:
            self.v = np.dot(self.β1,self.v) + np.dot((1 - self.β1),gradient_tensor)

        if self.r is None :
            self.r = np.dot((1 - self.β2),(np.power(gradient_tensor,2)))
        else:
            self.r = np.dot(self.β2,self.r) + np.dot((1 - self.β2),(np.square(gradient_tensor)))
        
        v_hat = np.divide(self.v , 1-(np.power(self.β1,self.iteration)))
        r_hat = np.divide(self.r , 1-np.power(self.β2,self.iteration))
        self.iteration += 1
        return weight_tensor - np.dot(self.learning_rate , np.divide(v_hat,(np.sqrt(r_hat)+self.epsilon)))