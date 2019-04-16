import cvxpy as cp
import numpy as np
from numpy import ones, maximum, minimum, sign, floor, ceil


   
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