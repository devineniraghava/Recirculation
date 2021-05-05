# -*- coding: utf-8 -*-
"""
Created on Tue May  4 13:19:39 2021

@author: Devineni
"""

from uncertainties import ufloat

v0 = ufloat(89.78,7.92)
V = ufloat(133.88,1.14)
tau0 = ufloat(1.8, 0.13)

K0=(tau0 * v0)/V

v1 = ufloat(27.2,2.57)
tau1 = ufloat(3.62, 0.27)


K1=(tau1 * v1)/V


print(K0, K1)



#%%


