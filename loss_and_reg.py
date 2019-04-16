import cvxpy as cp
import numpy as np
from numpy import ones, maximum, minimum, sign, floor, ceil


# Abstract Loss class
class Loss(object):
    def __init__(self, mask=None):return
    def loss(self, A, U): raise NotImplementedError("Override me!")
    def encode(self, A): return A # default
    def decode(self, A): return A # default
    def __str__(self): return "GLRM Loss: override me!"
    def __call__(self, A, U): return self.loss(A, U)

# class QuadraticLoss(Loss):
#     def loss(self, A, U): return cp.sum_squares(A - U)
#     def __str__(self): return "quadratic loss"
    
class QuadraticLoss(Loss):
    def loss(self, A, U): return cp.sum_squares(A - U)
    def scaled_loss(self, A, U, sigma_arr,mask):
        return cp.sum((cp.square(A-U)/sigma_arr)[mask])
        
    def __str__(self): return "quadratic loss"

class HuberLoss(Loss):
    a = 1.0 # XXX does the value of 'a' propagate if we update it?
    def loss(self, A, U): return cp.sum(cp.huber(cp.Constant(A) - U, self.a))
    def __str__(self): return "huber loss"

class HingeLoss(Loss):
    def loss(self, A, U): return cp.sum(cp.pos(ones(A.shape)-cp.multiply(cp.Constant(A), U)))
    def scaled_loss(self, A, U, sigma_arr, mask):
        return cp.sum((cp.pos(np.ones(A.shape)-cp.multiply(A, U))/sigma_arr)[mask])
    def decode(self, A): return sign(A) # return back to Boolean
    def __str__(self): return "hinge loss"

class OrdinalLoss(Loss):
    def __init__(self, Amin,Amax):
        """
        input
        -----
        Amax : Integer
            maximum value in A
        Amin: Integer
            minimum value in A
        """
        self.Amax, self.Amin = Amax,Amin
    def loss(self, A, U):
        return cp.sum(sum(cp.multiply(1*(b >= A),\
                cp.pos(U-b*ones(A.shape))) + cp.multiply(1*(b < A), \
                cp.pos(-U + (b+1)*ones(A.shape))) for b in range(int(self.Amin), int(self.Amax))))
    def decode(self, A): return maximum(minimum(A.round(), self.Amax), self.Amin)
    def __str__(self): return "ordinal loss"

    
#===================== Regularizers===============================
    
class Reg(object):
    # shape indicates how quickly it grows: 0 [flat], 1 [linear], 2 [quadratic+]
    def reg(self, X): raise NotImplementedError("Override me!")
    def __init__(self, nu=1): self.nu = nu # XXX think of a better way to handle nu?
    def __str__(self): return "GLRM Reg: override me!"
    def __call__(self, X): return self.reg(X)

class ZeroReg(Reg):
    def reg(self, X): return 0
    def __str__(self): return "zero reg"

class LinearReg(Reg):
    def reg(self, X): return self.nu*cp.norm(X)
    def __str__(self): return "linear reg"

class QuadraticReg(Reg):
    def reg(self, X): return self.nu*cp.sum_squares(X)
    def __str__(self): return "quadratic reg"