# -*- coding: utf-8 -*-
"""
@author: Marica Magagnini
"""

import numpy as np
import csv
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
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

from zcat import setG, f, Bias

###############################################################################
# Parameters definition 
###############################################################################

M = 3                           # Number of objective functions
n = 10*M                        # Number of decision variables
m = M-1                         # Number of yI variables
flag_CP = False                 # Complicated pareto set flag
flag_DL =  5                    # Level of difficulty {1,..,6}
flag_bias = False               # Bias function
flag_imbalanced = False         # Imbalanced problems flag
ZCAT = 20                       # F function flag {1,..,20}
G = setG(flag_CP,ZCAT)         # g function index {0,1,..,10} 



###############################################################################
# EA algorithm:  NSGA-II
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

###############################################################################
# Solve
###############################################################################
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
                
###############################################################################
# Indicators 
###############################################################################

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

###############################################################################
# Plots 
###############################################################################

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
    
           
       

