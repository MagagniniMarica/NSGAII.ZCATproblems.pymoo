# -*- coding: utf-8 -*-
"""
Created on Sat Jul 27 19:59:33 2024

@author: Maric
"""

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
#from pymoo.algorithms.moo.nsga3 import NSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling

from pymoo.config import Config
Config.warnings['not_compiled'] = False

from pymoo.termination import get_termination

from pymoo.optimize import minimize

import matplotlib.pyplot as plt

#from pymoo.util.ref_dirs import get_reference_directions
#from pymoo.visualization.scatter import Scatter

import timeit

###############################################################################
#  Function to set G
###############################################################################
def setG(flag_CP,ZCAT):
    if flag_CP == True:
        if ZCAT==1:
            return 4
        if ZCAT==2:
            return 5
        if ZCAT==3:
            return 2
        if ZCAT==4:
            return 7
        if ZCAT==5:
            return 9
        if ZCAT==6:
            return 4
        if ZCAT==7:
            return 5
        if ZCAT==8:
            return 2
        if ZCAT==9:
            return 7
        if ZCAT==10:
            return 9
        if ZCAT==11:
            return 3
        if ZCAT==12:
            return 10
        if ZCAT==13:
            return 1
        if ZCAT==14:
            return 6
        if ZCAT==15:
            return 8
        if ZCAT==16:
            return 10
        if ZCAT==17:
            return 1
        if ZCAT==18:
            return 8
        if ZCAT==19:
            return 6
        if ZCAT==20:
            return 3
    else:
        return 0
###############################################################################
# Definition of F functions
###############################################################################
#NOTA: j=1,..,M è l'indice delle funzioni obiettivo
"""y deve essere un np.array senno non funziona"""
def Fzcat1(yI, j,M):
    if(j==1):
        return np.prod(np.sin(yI*np.pi/2))
    if(j==M):
        return 1- np.sin(yI[0]*np.pi/2)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:                                               
        return np.prod(np.sin(yI[:M-j]*np.pi/2)) * np.cos(yI[M-j]*(np.pi/2))

def Fzcat2(yI, j,M):
    if(j==1):
        return np.prod(1-np.cos(yI*np.pi/2))
    if(j==M):
        return 1- np.sin(yI[0]*np.pi/2)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return (np.prod(1- np.cos(yI[:M-j]*np.pi/2))) * (1-np.sin(yI[M-j]*np.pi/2))

def Fzcat3(yI, j,M):
    if(j==1):
        return (1/(M-1))*np.sum(yI)
    if(j==M):
        return 1- yI[0]
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return (1/(M-j+1))*(np.sum(yI[:M-j])+(1-yI[M-j]))  

def Fzcat4(yI, j,M):
    if(j==M):
        return 1-(1/(M-1))*np.sum(yI)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return yI[j-1]

def Fzcat5(yI, j,M):
    if(j==M):
        """
        res = np.zeros(yI[0].shape)
        for i in range(M-1):
            res = res + (1-yI[i])
        """
        return (np.exp((1/(M-1))*np.sum(1-yI))**8 -1) / (np.exp(1)**8 - 1)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return yI[j-1]

def Fzcat6(yI, j,M):
    if(j==M):
        ro = 0.05
        k = 40
        mu = (1/(M-1))*sum(yI)
        
        a = 1 / (1 + np.exp(2*k*mu-k))
        b = 1/(1+np.exp(k))
        c = 1/(1+np.exp(-k))
        return (a - ro*mu -b +ro) / (c - b + ro)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return yI[j-1]

def Fzcat7(yI, j,M):
    if(j==M):
        return (1/(2*(M-1)*(0.5**5)))*np.sum((0.5-yI)**5) +1/2
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return yI[j-1]

def Fzcat8(yI, j,M):
    if(j==1):
        """
        res = np.ones(yI[0].shape)
        for i in range(M-1):
            res = res*(1- np.sin(yI[i]*np.pi/2))
        """
        return 1- np.prod(1- np.sin(yI*np.pi/2))
    if(j==M):
        return np.cos(yI[0]*np.pi/2)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        """
        res = np.ones(yI[0].shape)
        for i in range(M-j):
            res = res*(1- np.sin(yI[i]*np.pi/2))
        """
        return 1 - np.prod(1- np.sin(yI*np.pi/2))*(1-np.cos(yI[M-j]*np.pi/2))

def Fzcat9(yI, j,M):
    if(j==1):
        return (1/(M-1))*np.sum(np.sin(yI*np.pi/2))
    if(j==M):
        return np.cos(yI[0]*np.pi/2)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return ( 1/(M-j+1) ) * ( np.sum(np.sin(yI[:M-j]*np.pi/2)) + np.cos(yI[M-j]*np.pi/2) )

def Fzcat10(yI, j,M):
    if(j==M):
        ro = 0.02
        """
        res = np.zeros(yI[0].shape)
        for i in range(M-1):
            res = res + (1-yI[i])
        """
        return (1/ro - ((1/(M-1))*np.sum(1-yI) + ro)**(-1) ) / (1/ro - 1/(1+ro))
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return yI[j-1]

def Fzcat11(yI,j,M):
    if(j==1):
        return (1/(M-1))*np.sum(yI)
    if(j==M):
        K=4 #4 dal codice, 5 da tabella 4 nel paper
        return (np.cos((2*K-1)*yI[0]*np.pi) + 2*yI[0] + 4*K*(1-yI[0]) -1)/(4*K)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return (1/(M-j+1))*(np.sum(yI[:M-j])+(1-yI[M-j]))  
    
def Fzcat12(yI,j,M):
    if(j==1):
        """
        res = np.ones(yI[0].shape)
        for i in range(M-1):
            res = res * (1-yI[i])
        """
        return 1- np.prod(1-yI)
    if(j==M):
        K=3 # dal codice
        return (np.cos((2*K-1)*yI[0]*np.pi) + 2*yI[0] + 4*K*(1-yI[0]) -1)/(4*K)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        """
        res = np.ones(yI[0].shape)
        for i in range(M-j):
            res = res * (1-yI[i])
        """
        return 1- np.prod(1-yI)*yI[M-j]

def Fzcat13(yI,j,M):
    if(j==1):
        return 1- (1/(M-1))*np.sum(np.sin(yI*np.pi/2))
    if(j==M):
        K=3 # dal codice
        return 1- ((np.cos((2*K-1)*yI[0]*np.pi) + 2*yI[0] + 4*K*(1-yI[0]) -1)/(4*K))
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return 1- (1/(M-j+1)) * (np.sum(np.sin(yI*np.pi/2)) + np.cos(yI[M-j]*np.pi/2))

def Fzcat14(yI,j,M):
    if(j==1):
        return np.sin(yI[0]*np.pi/2)**2
    if(M>2 and j==M-1 ):
        return 0.5*(1+np.sin(6*yI[0]*np.pi/2 - np.pi/2))
    if(j==M):
        return np.cos(yI[0]*np.pi/2)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return np.sin(yI[0]*np.pi/2)**(2 + (j-1)/(M-2))

def Fzcat15(yI,j,M):
    if(j==M):
        K=3 # dal codice
        return  ((np.cos((2*K-1)*yI[0]*np.pi) + 2*yI[0] + 4*K*(1-yI[0]) -1)/(4*K))
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return yI[0]**(1+(j-1)/(4*M))

def Fzcat16(yI,j,M):
    if(j==1):
        return np.sin(yI[0]*np.pi/2)
    if(M>2 and j==M-1 ):
        return 0.5*(1+np.sin(10*yI[0]*np.pi/2 - np.pi/2))
    if(j==M):
        K=5 # dal codice
        return ((np.cos((2*K-1)*yI[0]*np.pi) + 2*yI[0] + 4*K*(1-yI[0]) -1)/(4*K))
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return np.sin(yI[0]*np.pi/2)**(2 + (j-1)/(M-2))
    
def Fzcat17(yI,j,M):
    if j == M:
        if (yI <= 0.5).all():
            return (np.exp(1-yI[0])**8 -1)/(np.exp(1)**8 -1)
        else:
            return (np.exp((1/(M-1))*np.sum(1-yI))**8 -1) /  (np.exp(1)**8 -1)
    else:
        if (yI <= 0.5).all():
            return yI[0]
        else:
            return yI[j-1]

def Fzcat18(yI,j,M):
    if j == M:
        if (yI <= 0.4).all() or (yI >= 0.6).all():
            return ((0.5-yI[0])**5 + 0.5**5) / (2*(0.5**5))
        else:
            return np.sum((0.5-yI)**5) / (2*(M-1)*(0.5**5)) + 1/2
    else:
        if (yI <= 0.4).all() or (yI >= 0.6).all():
            return yI[0]
        else:
            return yI[j-1]        

def Fzcat19(yI,j,M):
    if (yI[0] in [0,0.2]) or  (yI[0] in [0.4,0.6]):
        m = 1
    else:
        m = M-1
    if(j==M):
        if m ==1:
            return 1-yI[0]- (np.cos(10*np.pi*yI[0] + np.pi/2)/(10*np.pi))
        else:
            return 1- (1/(M-1))* np.sum(yI) - (np.cos( (10*np.pi/(M-1)) * np.sum(yI) + (np.pi/2) )/(10*np.pi))
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        if m ==1:
            return yI[0]
        else:
            return yI[j-1]

def Fzcat20(yI,j,M):
    if (yI[0] in [0.1,0.4]) or  (yI[0] in [0.6,0.9]):
        m = 1
    else:
        m = M-1
    if(j==M):
        if m ==1:
            return ((0.5- yI[0])**5 + 0.5**5) / (2 * (0.5**5))
        else:
            return np.sum((0.5 - yI)**5)/(2*(M-1)*(0.5**5)) + 0.5
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        if m==1:
            return yI[0]
        else:
            return yI[j-1]

###############################################################################
# Definition of g functions 
###############################################################################
""""m = M-1, G= quale g schelgo {0,..,10}, l = m+1,..,n, yI variabili prima parte"""
def g(yI,G, m,l,n):                                                                                     #NOTA: j=m+1,..,n
    
    if (l<m+1 or l >n):
        raise ValueError("Error: variable index out of bound in fuction g")
        
    theta = (2*np.pi*(l-(m+1)))/n 
    
    if G==0:
        return 0.2210
    elif G==1:
        return (1/(2*m))*np.sum(np.sin(1.5*np.pi*yI + theta)) +1/2
    elif G==2:
        return (1/(2*m))*np.sum(np.multiply((yI**2), np.sin(4.5*np.pi*yI + theta))) +1/2
    elif G==3:
        return (1/(m))*np.sum(np.cos(np.pi*yI + theta)**2)
    elif G==4:
        return (1/(2*m))*np.sum(yI*np.cos((4*np.pi/m)*np.sum(yI + theta))) + 1/2
    elif G==5:
        return (1/(2*m))*np.sum(np.sin(2*np.pi*yI -1 + theta)**3) + 1/2
    elif G==6:
        num = (-10 * np.exp(np.sqrt((1/m)*np.sum(yI**2))*(-2/5)) + 10 + np.exp(1) - 
               np.exp((1/m)*np.sum(np.cos(11*np.pi*yI+theta)**3)))
        denum = (-10*np.exp(-2/5) - np.exp(-1) +10 + np.exp(1))
        return  num/denum 
    elif G==7:
        num =(1/m) * np.sum( yI) + np.exp( np.sin( (7*np.pi/m)*np.sum(yI)-np.pi/2 +theta)  )  - np.exp(-1)
        return num/ (1- np.exp(-1) + np.exp(1))
    elif G==8:
        return (1/m)*np.sum(abs(np.sin(2.5*np.pi*(yI-0.5) + theta)))
    elif G==9:
        return (1/(2*m))*np.sum(yI) - (1/(2*m))*np.sum(abs(np.sin(2.5*np.pi*yI - np.pi/2 + theta))) + 1/2
    elif G==10:
        return (1/(2*(m**3))) * (np.sum(np.sin((4*yI - 2)*np.pi + theta))**3) + 1/2
    else:
        raise ValueError("Error: G index is not valid")
     
###############################################################################
# Definition of Bias functions 
###############################################################################
"""Se la flag del bias è true, prende il vettore zII = yII-g(yI) e gli applica un bias"""
def Bias(flag_bias,zII):
    if (flag_bias == True):
        z_bias = np.zeros(len(zII))
        for z in range(len(zII)):
            z_bias[z] = abs(zII[z])**0.05
        return z_bias
    else:
        return zII
        
"""Se la flag è zero non gli applica nessun bias"""

###############################################################################
# Definition of Z_level functions 
###############################################################################
#j = 1,...,M, w  = Bias(flag_bias,zII)
def Z_level(w,j, flag_DL,m,n,M):
    J = []
    for l in range(m+1, n+1):
        #print(l)
        if (l-m-j)%M ==0:
            J.append(l-m-1) # -m-1 perche dopo gli indici mi servono che partono da zero
    
    if flag_DL ==1:
        return (10/len(J))*sum(s**2 for s in w if (list(w).index(s)) in J)
    if flag_DL ==2:
        return 10 * max(abs(s) for s in w if (list(w).index(s)) in J)
    if flag_DL ==3:
        return (10/len(J)) * sum( ((1/3)*(s**2- np.cos(9*np.pi*s)+1)) for s in w if (list(w).index(s)) in J)
    if flag_DL ==4:
        e1 = max(abs(s)**0.5 for s in w if (list(w).index(s)) in J)
        e2 = (1/len(J)) * sum( ((1/2)*( np.cos(9*np.pi*s)+1)) for s in w if (list(w).index(s)) in J)
        return (10/(2* np.exp(1) - 2.0)) *  (np.exp(e1)-np.exp(e2) -1 +np.exp(1))
    if flag_DL == 5:
        return -0.7* Z_level(w, j, 3, m, n, M) + (10/len(J))*sum(abs(s)**0.002 for s in w if (list(w).index(s)) in J)
    if flag_DL == 6:
        return -0.7* Z_level(w, j, 4, m, n, M) + 10*((1/len(J))*sum(abs(s)for s in w if (list(w).index(s)) in J))**0.002 
 
        
###############################################################################
# Definition of alpha function
###############################################################################
"""m bisogna inserirla perché in zcat19 e zcat20 cambia, tra laltro lo devo inserire
calcolato con il setm"""
def alpha(yI, j, ZCAT,M):                             #NOTA: j=1,..,M
    if(ZCAT==1):
        return (j**2) * Fzcat1(yI,j,M)
    if(ZCAT==2):
        return (j**2) * Fzcat2(yI,j,M)
    if(ZCAT==3):
        return (j**2) * Fzcat3(yI, j,M)
    if(ZCAT==4):
        return (j**2) * Fzcat4(yI, j,M)
    if(ZCAT==5):
        return (j**2) * Fzcat5(yI, j,M)
    if(ZCAT==6):
        return (j**2) * Fzcat6(yI, j,M)
    if(ZCAT== 7):
        return (j**2) * Fzcat7(yI, j,M)
    if(ZCAT== 8):
        return (j**2) * Fzcat8(yI, j,M)
    if(ZCAT== 9):
        return (j**2) * Fzcat9(yI, j,M)
    if(ZCAT== 10):
        return (j**2) * Fzcat10(yI, j,M)
    if(ZCAT== 11):
        return (j**2) * Fzcat11(yI, j,M)
    if(ZCAT== 12):
        return (j**2) * Fzcat12(yI, j,M)
    if(ZCAT== 13):
        return (j**2) * Fzcat13(yI, j,M)
    if(ZCAT== 14):
        return (j**2) * Fzcat14(yI, j,M)
    if(ZCAT== 15):
        return (j**2) * Fzcat15(yI, j,M)
    if(ZCAT== 16):
        return (j**2) * Fzcat16(yI, j,M)
    if(ZCAT== 17):
        return (j**2) * Fzcat17(yI, j,M)
    if(ZCAT== 18):
        return (j**2) * Fzcat18(yI, j,M)
    if(ZCAT== 19):
        return (j**2) * Fzcat19(yI, j,M)
    if(ZCAT== 20):
        return (j**2) * Fzcat20(yI, j,M)
    else:
        raise ValueError("Error: G index is not valid")

###############################################################################
# Definition of beta function
###############################################################################
 #j = 1,...,M, w  = Bias(flag_bias,zII)
def beta(flag_imbalanced,w, j,flag_DL,m,n,M):  #NOTA: j=1,..,M
    if flag_imbalanced == True:
        if(j%2 ==0):
            return (j**2) *  Z_level(w,j, 4,m,n,M)
        else: 
            return (j**2) *  Z_level(w,j, 1,m,n,M)
    else:                                                                      
        return (j**2) *  Z_level(w,j, flag_DL,m,n,M)     

###############################################################################
 # Definition of f objective function 
###############################################################################
#j = 1,...,M, that is the jth objective function, w  = Bias(flag_bias,zII)
"""ricorda che la m va messa calcolata dal setm"""
def f(yI, j, ZCAT, flag_imbalanced,w,flag_DL,m,n,M):  #NOTA: j=1,..,M
    return alpha(yI, j, ZCAT,M) + beta(flag_imbalanced, w, j, flag_DL, m, n, M)

###############################################################################
###############################################################################
#
# Parameters definition 
#

M = 3                           # Number of objective functions
n = 10*M                        # Number of decision variables
m = M-1                         # Number of yI variables
flag_CP = False                 # Complicated pareto set flag
flag_DL =  5                    # Level of difficulty {1,..,6}
flag_bias = False               # Bias function
flag_imbalanced = False         # Imbalanced problems flag
ZCAT = 20                       # F function flag {1,..,20}
#G = setG(flag_CP,ZCAT)         # g function index {0,1,..,10} 



###############################################################################
# Solving the problem with NSGAII
###############################################################################
N = 200                                 # Population size

#
#IMPORT THE ALGORITHM
#
algorithm = NSGA2(
    pop_size=N,
    n_offsprings=50,
    sampling=FloatRandomSampling(),
    crossover=SBX(prob=0.9, eta=15),
    mutation=PM(prob=1/n,eta=20),
    eliminate_duplicates=True
)

#
# TERMINATION CRITERIUM
#
#termination = get_termination("n_gen", 40)
termination = get_termination("n_eval", 2000*N)

###############################################################################
# Definition of the ZCAT problem in pymoo
###############################################################################
import csv
with open(f"indicators_M{str(M)}_ZCAT{str(ZCAT)}_levels.csv", 'w', newline='') as file:
    w = csv.writer(file,delimiter=';',)
    w.writerow(['GD', 'GD+', 'IGD', 'IGD+', 'HV', 'maxHV', 'HR','RunTime'])
    for flag_DL in range(5,6):
        G = setG(flag_CP,ZCAT)          # g function index {0,1,..,10} 
        class MyProblem(ElementwiseProblem):

            def __init__(self):
                super().__init__(n_var=n,       #n
                                 n_obj=M,       #M 
                                 n_ieq_constr=0,
                                 xl=np.zeros(n),
                                 xu=np.ones(n))
            


            def _evaluate(self, x, out, *args, **kwargs):
                zII = np.zeros(n-m)
                for l in range(len(zII)):
                    #print(l)
                    zII[l] = x[l+m] -g(x[:m], G, m, l+m+1, n)
                
                if M == 2: 
                    f1 = f(x[:m], 1, ZCAT, flag_imbalanced,Bias(flag_bias,zII),flag_DL,m,n,M)
                    f2 = f(x[:m], 2, ZCAT, flag_imbalanced,Bias(flag_bias,zII),flag_DL,m,n,M)
                    out["F"] = [f1, f2]
                elif M == 3:
                    f1 = f(x[:m], 1, ZCAT, flag_imbalanced,Bias(flag_bias,zII),flag_DL,m,n,M)
                    f2 = f(x[:m], 2, ZCAT, flag_imbalanced,Bias(flag_bias,zII),flag_DL,m,n,M)
                    f3 = f(x[:m], 3, ZCAT, flag_imbalanced,Bias(flag_bias,zII),flag_DL,m,n,M)
                    out["F"] = [f1, f2,f3]
                


        problem = MyProblem()
        #
        #Get start time
        #
        start = timeit.default_timer()
        #
        # OPTIMIZE THE PROBLEM
        #
        res = minimize(problem,
                       algorithm,
                       termination,
                       seed=1,
                       save_history=True,
                       verbose=True)
        
        X = res.X
        F = res.F

        stop = timeit.default_timer()
        runtime = stop-start
        
        
        #
        # Indicators
        # 
            
        from ParetoFrontSet import Pareto_Front_Set
        if M ==2:
            n_punti = 225
        elif M == 3:
            n_punti = 15

        fig_front, ax_front, pf = Pareto_Front_Set(n_punti, ZCAT, M,G, n,m, 'F')

        from pymoo.indicators.gd import GD
        ind = GD(pf)
        GD = ind(F)
        
        from pymoo.indicators.gd_plus import GDPlus
        ind = GDPlus(pf)
        GDp = ind(F)
        
        from pymoo.indicators.igd import IGD
        ind = IGD(pf)
        IGD = ind(F)
        
        from pymoo.indicators.igd_plus import IGDPlus
        ind = IGDPlus(pf)
        IGDp = ind(F)

        from pymoo.indicators.hv import HV
        if M ==2:
            ref_point = np.array([50,50])
        elif M == 3:
            ref_point = np.array([50,50,50])
        ind = HV(ref_point=ref_point)
        HV= ind(F)
        HVmax = ind(pf)
        HR = HV/HVmax
        
        w.writerow([str(GD).replace('.',','),str(GDp).replace('.',','), 
                         str(IGD).replace('.',','), str(IGDp).replace('.',','),
                         str(HV).replace('.',','), str(HVmax).replace('.',','),
                         str(HR).replace('.',','), str(runtime).replace('.',',')])
        
        #
        # PLOTs
        #
        Xcopy = np.zeros(X.shape)

        for i in range(len(X)):
            for j in range(np.size(X,1)):
                Xcopy[i,j] = (j+1)*X[i,j] - (j+1)/2


        #code name plots
        code = str(flag_DL)
        if (flag_CP == True):
            code = 'CPS'
        elif (flag_bias == True):
            code = 'BIAS'
        elif (flag_imbalanced == True):
                code = 'IMB'
                
        
        if M ==2:
            #
            # Design space (Pareto set)
            #
            plt.figure(figsize=(7, 6))
            plt.scatter(Xcopy[:, 0], Xcopy[:, 1], s=30, facecolors='none', edgecolors='r')
            plt.xlabel("$x_1$", fontsize=15)
            plt.ylabel("$x_2$", fontsize=15)
            plt.title("Design Space")
            plt.savefig('sZCAT'+ str(ZCAT) + '_M2_' + code + '.png')

            fig_set, ax_set, pf = Pareto_Front_Set(n_punti, ZCAT, M,G, n,m, 'S')
            ax_set.scatter(Xcopy[:, 0], Xcopy[:, 1], s=30, facecolors='none', edgecolors='r')
            plt.title("Design Space")
            plt.savefig('sZCAT'+ str(ZCAT) + '_M2_' + code + '+.png')

            #
            # Objective space (Pareto front)
            #
            plt.figure(figsize=(7, 6))
            plt.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
            plt.xlabel("$f_1(\mathbf{x})$", fontsize=15)
            plt.ylabel("$f_2(\mathbf{x})$", fontsize=15)
            plt.title("Objective Space")
            plt.savefig('fZCAT'+ str(ZCAT) + '_M2_' + code + '.png')

            fig_front, ax_front, a = Pareto_Front_Set(n_punti, ZCAT, M,G, n,m, 'F')
            ax_front.scatter(F[:, 0], F[:, 1], s=30, facecolors='none', edgecolors='blue')
            plt.title("Objective Space")
            plt.savefig('fZCAT'+ str(ZCAT) + '_M2_' + code + '+.png')            
        
        if M==3:
            #
            # Design space (Pareto set)
            #
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            ax.scatter(Xcopy[:, 0], Xcopy[:, 1], Xcopy[:, 2], facecolors='r', s=30, marker='o', edgecolors='k')
            ax.set_xlabel("$x_1$", fontsize=15)
            ax.set_ylabel("$x_2$", fontsize=15)
            ax.set_zlabel("$x_3$", fontsize=15)
            plt.title("Design Space")
            plt.savefig('sZCAT'+ str(ZCAT) + '_M3_L' + code + '.png')
            
            fig_set, ax_set, a = Pareto_Front_Set(n_punti, ZCAT, M,G, n,m, 'S')
            ax_set.scatter(Xcopy[:, 0], Xcopy[:, 1], Xcopy[:, 2], facecolors='r', s=30, marker='o', edgecolors='k')
            plt.title("Design Space")
            plt.savefig('sZCAT'+ str(ZCAT) + '_M3_L' + code + '+.png')
            
            #
            # Objective space (Pareto front)
            #
            fig = plt.figure()
            ax = fig.add_subplot(projection='3d')
            xl, xu = problem.bounds()
            ax.scatter(F[:, 0], F[:, 1],F[:, 2], facecolors='b', s=30, marker='o', edgecolors='b')
            ax.set_xlabel("$f_1(\mathbf{x})$", fontsize=15)
            ax.set_ylabel("$f_2(\mathbf{x})$", fontsize=15)
            ax.set_zlabel("$f_3(\mathbf{x})$", fontsize=15)
            plt.title("Objective Space")
            plt.savefig('fZCAT'+ str(ZCAT) + '_M3_L' + code + '.png')
            
            
            fig_front, ax_front, a = Pareto_Front_Set(n_punti, ZCAT, M,G, n,m, 'F')
            ax_front.scatter(F[:, 0], F[:, 1],F[:, 2], facecolors='b', s=30, marker='o', edgecolors='b')
            plt.title("Objective Space")
            plt.savefig('fZCAT'+ str(ZCAT) + '_M3_L' + code + '+.png')
       
        
        #
        # Save the results to a csv
        #
        with open(f"sZCAT{str(ZCAT)}_M{str(M)}_{code}.csv", 'w', newline='') as h:
            writer = csv.writer(h, delimiter=';')
            for row in X:
                writer.writerow(row) 
                
        with open(f"fZCAT{str(ZCAT)}_M{str(M)}_{code}.csv", 'w', newline='') as h:
            writer = csv.writer(h, delimiter=';')
            for row in F:
                writer.writerow(row)
        

    print('All records registered.')


"""
AUMENTA I PUNTI PER FARE L'HVmax CHE ALMENO VIENE PIù PRECISO
fig_front, ax_front, pf2 = Pareto_Front_Set(10000, ZCAT, M,G, n,m, 'F')
from pymoo.indicators.hv import HV
if M ==2:
    ref_point = np.array([50,50])
ind = HV(ref_point=ref_point)
HVmax = ind(pf2)
"""

###############################################################################
# IMPORT PS SOLUTIONS AND COMPUTE PF SOLUTIONS
###############################################################################
from ParetoFrontSet import Pareto_Front_Set
import csv
M = 3                           # Number of objective functions
n = 10*M                        # Number of decision variables
m = M-1                         # Number of yI variables
ZCAT = 19                       # F function flag {1,..,20}
flag_CP = False                 # Complicated pareto set flag
flag_bias = False               # Bias function
flag_imbalanced = False         # Imbalanced problems flag


test = ['L1','L2','L3','L4','L5','L6','CPS','BIAS','IMB'] 
level = [1,2,3,4,5,6,1,1,1]
cont = 0
for t in test:
    with open(f'sZCAT{str(ZCAT)}_M{str(M)}_{t}.csv', newline='') as csvfile:
        data = list(csv.reader(csvfile, delimiter=';'))
    X = np.array(data, dtype=float)
    
    if M == 3:
        F = np.zeros((200,3))
    else:
        F = np.zeros((200,2))
    
    
    flag_DL =  level[cont]                    # Level of difficulty {1,..,6}
    if (t == 'CPS'):
        flag_CP = True
    elif (t == 'BIAS'):
        flag_bias = True   
    elif (t == 'IMB'):
        flag_imbalanced = True
    
    G = setG(flag_CP,ZCAT)         # g function index {0,1,..,10} 
    
    
    for riga in range(0,200):
        x = X[riga,:]
        zII = np.zeros(n-m)
        for l in range(len(zII)):
            #print(l)
            zII[l] = x[l+m] -g(x[:m], G, m, l+m+1, n)

        if M == 2: 
            F[riga,0] = f(x[:m], 1, ZCAT, flag_imbalanced,Bias(flag_bias,zII),flag_DL,m,n,M)
            F[riga,1]= f(x[:m], 2, ZCAT, flag_imbalanced,Bias(flag_bias,zII),flag_DL,m,n,M)
        elif M == 3:
            F[riga,0] = f(x[:m], 1, ZCAT, flag_imbalanced,Bias(flag_bias,zII),flag_DL,m,n,M)
            F[riga,1] = f(x[:m], 2, ZCAT, flag_imbalanced,Bias(flag_bias,zII),flag_DL,m,n,M)
            F[riga,2] = f(x[:m], 3, ZCAT, flag_imbalanced,Bias(flag_bias,zII),flag_DL,m,n,M)

    
    
    if M ==2:
        n_punti = 225
    elif M == 3:
        n_punti = 30
            
    fig_front, ax_front, pf = Pareto_Front_Set(n_punti, ZCAT, M,G, n,m, 'F')


    from pymoo.indicators.hv import HV
    if M ==2:
        ref_point = np.array([1.1,4.4])
    elif M == 3:
        ref_point = np.array([50,50,50])
    ind = HV(ref_point=ref_point)
    HV= ind(F)
    HVmax = ind(pf)
    HR = HV/HVmax
    
    print(f"HV_{t} = ", HV)
    print(f"HVmax_{t} = ", HVmax)
    print(f"HR_{t} = ", HR)
    print('------------------------------------------------------------------')
    
    cont = cont + 1 
    flag_CP = False                 # Complicated pareto set flag
    flag_bias = False               # Bias function
    flag_imbalanced = False         # Imbalanced problems flag


