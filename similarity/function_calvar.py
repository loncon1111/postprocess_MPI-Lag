#coding=utf-8
#import libraries
import numpy as np
import pandas as pd

# class
class CalVar:
    """
    Calculate meteorological variables
    """
    #def __init__(self):
    def TH(t, p):
        # physical parameters:
        rdcp = 0.286; tzero = 273.15
        # Calculation - distinction between temperature in K and in C
        if t > 100.:
            pt = (t + tzero) ( ( (1000./p) ** rdcp))
        else:
            pt = t * ( (1000./p) ** rdcp)
        return pt
    
    def RHO(t, p):
        # physical parameters:
        rd = 287.05; tzero = 273.15
        # Calculation
        if t > 100.:
            tk = t + tzero
        else:
            tk = t

        rho = 100. * p/ (tk * rd)
        
        
        
