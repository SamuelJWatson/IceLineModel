# Glacial ice line model with mass balance flip flop

import numpy as np
import scipy as scipy
import scipy.stats as stats
import scipy.linalg as linalg
import scipy.integrate as integrate
from pprint import pprint
import sys
import os
import copy
import string
import glob
import xarray as xr

import warnings

def IL(t,n0,pars):
    w = n0[0,]
    ns = n0[1,]
    nn = n0[2,]
    en = n0[3,]
    
    R = pars[0,]    # Surface layer heat capacity (W/m^2K)
    Q = pars[1,]    # Annual average insolation (W/m^2)
    beta = pars[2,] # Obliquity (radians)
    a1 = pars[3,]   # Albedo between ice line latitudes
    a2 = pars[4,]   # Albedo polewards of ice line latitudes
    A = pars[5,]    # Greenhouse gas parameter (W/m^2)
    B = pars[6,]    # Outgoing radiation (W/m^2K)
    C = pars[7,]    # Efficiency of heat transport (W/m^2K)
    T_cS = pars[8,]  # critical temperature south line (^oC)
    T_cNpos = pars[9,]  # critical temperature north line when retreating (^oC)
    T_cNneg = pars[10,]  # critical temperature north line when advancing (^oC)
    p = pars[11,]    # Ice line response to temp change (1/(K yr))
    eps = pars[12,]  # mass balance response to albedo change
    a = pars[13,]    # accumulation rate
    b = pars[14,]    # critical ablation rate
    bpos = pars[15,] # interglacial ablation rate
    bneg = pars[16,] # glacial ablation rate

    L = Q/(B+C)
    a0 = 0.5*(a1+a2)
    
    def s(y,beta):
        insol = 1 - (5/32)*(3*np.cos(beta)**2 - 1)*(3*y**2 - 1)
        return insol
    
    [sint,err] = integrate.quad(lambda y: s(y,beta),ns,nn)
    
    F = (1/B)*(Q*(1-a0) - A + 0.5*C*L*(a1-a2)*(1-sint)) 
    
    G_ns = -L*(1-a0)*(s(ns,beta)-1) + T_cS
    
    if en < (1+(a/b))*nn - a/b : # interglacial period of retreat
        H_nn = -L*(1-a0)*(s(nn,beta)-1) + T_cNpos
        bpm = bpos
    elif en > (1+(a/b))*nn - a/b : # glacial period of advance
        H_nn = -L*(1-a0)*(s(nn,beta)-1) + T_cNneg
        bpm = bneg
    
    dwdt = -(B/R)*(w - F)
    dnsdt = -p*(w - G_ns)
    dnndt = p*(w - H_nn)
    dendt = eps*(bpm*(nn-en) - a*(1-nn))
    
    ddt = np.array([dwdt,dnsdt,dnndt,dendt])
    
    return ddt