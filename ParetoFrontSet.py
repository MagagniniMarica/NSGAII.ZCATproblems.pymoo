"""
@author: Marica Magagnini

Plot Pareto Fronts and Sets
"""

import numpy as np
import matplotlib.pyplot as plt



########################################################
# Definition of F functions
###############################################################################
#NOTA: j=1,..,M è l'indice delle funzioni obiettivo
"""y deve essere un np.array senno non funziona"""
def Fzcat1(yI, j,M):
    if(j==1):
        res = np.ones(yI[0].shape)
        for i in range(M-1):
            res = res*np.sin(yI[i]*np.pi/2)
        return res
    if(j==M):
        return 1- np.sin(yI[0]*np.pi/2)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        res = np.ones(yI[0].shape)
        for i in range(M-j):
            res = res*np.sin(yI[i]*np.pi/2)
        return res* (np.cos(yI[M-j]*np.pi/2))

def Fzcat2(yI, j,M):
    if(j==1):
        res = np.ones(yI[0].shape)
        for i in range(M-1):
            res = res*(1-np.cos(yI[i]*np.pi/2))
        return res
    if(j==M):
        return 1- np.sin(yI[0]*np.pi/2)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        res = np.ones(yI[0].shape)
        for i in range(M-j):
            res = res*(1- np.cos(yI[i]*np.pi/2))
        return res* (1-np.sin(yI[M-j]*np.pi/2))

def Fzcat3(yI, j,M):
    if(j==1):
        return (1/(M-1))*sum(yI)
    if(j==M):
        return 1- yI[0]
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return (1/(M-j+1))*(sum(yI[:M-j])+(1-yI[M-j]))  
    
def Fzcat4(yI, j,M):
    if(j==M):
        return 1-(1/(M-1))*sum(yI)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return yI[j-1]

def Fzcat5(yI, j,M):
    if(j==M):
        res = np.zeros(yI[0].shape)
        for i in range(M-1):
            res = res + (1-yI[i])
        return (np.exp((1/(M-1))*res)**8 -1) / (np.exp(1)**8 - 1)
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
        res = np.zeros(yI[0].shape)
        for i in range(M-1):
            res = res + ((0.5-yI[i])**5)
        return (1/(2*(M-1)*(0.5**5)))*res +1/2
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return yI[j-1]

def Fzcat8(yI, j,M):
    if(j==1):
        res = np.ones(yI[0].shape)
        for i in range(M-1):
            res = res*(1- np.sin(yI[i]*np.pi/2))
        return 1-res
    if(j==M):
        return np.cos(yI[0]*np.pi/2)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        res = np.ones(yI[0].shape)
        for i in range(M-j):
            res = res*(1- np.sin(yI[i]*np.pi/2))
        return 1 - res*(1-np.cos(yI[M-j]*np.pi/2))

def Fzcat9(yI, j,M):
    if(j==1):
        res = np.zeros(yI[0].shape)
        for i in range(M-1):
            res = res + np.sin(yI[i]*np.pi/2)
        return (1/(M-1))*res
    if(j==M):
        return np.cos(yI[0]*np.pi/2)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        res = np.zeros(yI[0].shape)
        for i in range(M-j):
            res = res + np.sin(yI[i]*np.pi/2)
        return ( 1/(M-j+1) ) * (res + np.cos(yI[M-j]*np.pi/2))
        
def Fzcat10(yI, j,M):
    if(j==M):
        ro = 0.02
        res = np.zeros(yI[0].shape)
        for i in range(M-1):
            res = res + (1-yI[i])
        return (1/ro - ((1/(M-1))*res + ro)**(-1) ) / (1/ro - 1/(1+ro))
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return yI[j-1]

def Fzcat11(yI,j,M):
    if(j==1):
        return (1/(M-1))*sum(yI)
    if(j==M):
        K=4 #4 dal codice, 5 da tabella 4 nel paper
        return (np.cos((2*K-1)*yI[0]*np.pi) + 2*yI[0] + 4*K*(1-yI[0]) -1)/(4*K)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return (1/(M-j+1))*(sum(yI[:M-j])+(1-yI[M-j]))  

def Fzcat12(yI,j,M):
    if(j==1):
        res = np.ones(yI[0].shape)
        for i in range(M-1):
            res = res * (1-yI[i])
        return 1-res
    if(j==M):
        K=3 # dal codice
        return (np.cos((2*K-1)*yI[0]*np.pi) + 2*yI[0] + 4*K*(1-yI[0]) -1)/(4*K)
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        res = np.ones(yI[0].shape)
        for i in range(M-j):
            res = res * (1-yI[i])
        return 1- res*yI[M-j]

def Fzcat13(yI,j,M):
    if(j==1):
        res = np.zeros(yI[0].shape)
        for i in range(M-1):
            res = res + np.sin(yI[i]*np.pi/2)
        return 1- (1/(M-1))*res
    if(j==M):
        K=3 # dal codice
        return 1- ((np.cos((2*K-1)*yI[0]*np.pi) + 2*yI[0] + 4*K*(1-yI[0]) -1)/(4*K))
    if(j>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        res = np.zeros(yI[0].shape)
        for i in range(M-j):
            res = res + np.sin(yI[i]*np.pi/2)
        return 1- (1/(M-j+1)) * (res + np.cos(yI[M-j]*np.pi/2))

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
    if M == 2:
        t = np.zeros(225)
        for v in range(0,225):
            if (yI[0][v] <= 0.5)  :
                if j == M:
                    t[v] = (np.exp(1-yI[0][v])**8 -1)/(np.exp(1)**8 -1)
                else:
                    t[v] = yI[0][v]
            else:
                if j == M:
                    t[v] = (np.exp((1/(M-1))*(1-yI[0][v]))**8 -1) /  (np.exp(1)**8 -1)
                else:
                    t[v] = yI[j-1][v]
        return t
               
    if M == 3:
        t = np.zeros(( 15, 15))
        for v in range(0,15):
            for w in range(0,15):
                if ((yI[0][v][w] <= 0.5 ) and (yI[1][v][w] <= 0.5 )) :
                    if j == M:
                        t[v][w] = (np.exp(1-yI[0][v][w])**8 -1)/(np.exp(1)**8 -1)
                    else:
                        t[v][w] = yI[0][v][w]
                else:
                    if j == M:
                        t[v][w] = (np.exp((1/(M-1))*(2-yI[0][v][w]-yI[1][v][w]))**8 -1) /  (np.exp(1)**8 -1)
                    else:
                        t[v][w] = yI[j-1][v][w]
        return t    


def Fzcat18(yI,j,M):
    if M == 2:
        t = np.zeros(225)
        for v in range(0,225):
            if (yI[0][v] <= 0.4 or yI[0][v] >= 0.6)  :
                if j == M:  
                    t[v] = ((0.5-yI[0][v])**5 + 0.5**5) / (2*(0.5**5))
                else:
                    t[v] = yI[0][v]
            else:
                if j == M:
                    
                    t[v] = ((0.5-yI[0][v])**5 / (2*(M-1)*(0.5**5))) + 1/2
                else:
                    t[v] = yI[j-1][v]
        return t
               
    if M == 3:
        t = np.zeros(( 15, 15))
        for v in range(0,15):
            for w in range(0,15):
                if (((yI[0][v][w] <= 0.4 ) and (yI[1][v][w] <= 0.4 )) or ((yI[0][v][w] >= 0.6 ) and (yI[1][v][w] >= 0.6 ))) :
                    if j == M:
                        t[v][w] = ((0.5-yI[0][v][w])**5 + 0.5**5) / (2*(0.5**5))
                    else:
                        t[v][w] = yI[0][v][w]
                else:
                    if j == M:
                        t[v][w] = (((0.5-yI[0][v][w])**5 + (0.5-yI[1][v][w])**5) / (2*(M-1)*(0.5**5))) + 1/2
                    else:
                        t[v][w] = yI[j-1][v][w]
        return t

   
def Fzcat19(yI, j,M):
    if M == 2:
        t = np.zeros(225)
        for v in range(0,225):
            if j == M:  
                t[v] = 1 - yI[0][v] - np.cos(10*np.pi*yI[0][v] + np.pi/2)/(10*np.pi)
            else:
                t[v] = yI[0][v]                
        return t
                   
    if M == 3:
        p = 15
        t = np.zeros(( p, p))
        for v in range(0,p):
            for w in range(0,p):
                if (yI[0][v][w] in [0,0.2])  or (yI[0][v][w] in [0.4,0.6]) :
                    if j == M:
                        t[v][w] = 1 - yI[0][v][w] - np.cos(10*np.pi*yI[0][v][w] + np.pi/2)/(10*np.pi)
                    else:
                        t[v][w] = yI[0][v][w]
                else:
                    if j == M:
                        t[v][w] = 1- (1/(M-1))* (yI[0][v][w]+yI[1][v][w]) - (np.cos( (10*np.pi/(M-1)) *(yI[0][v][w]+yI[1][v][w]) + (np.pi/2) )/(10*np.pi))
                    else:
                        t[v][w] = yI[j-1][v][w]
        return t
  
def Fzcat20(yI, j,M):
    if M == 2:
        t = np.zeros(225)
        for v in range(0,225):
            if (yI[0][v] in [0.1,0.4])  or (yI[0][v] in [0.6,0.9]) :
                if j == M:  
                    t[v] =((0.5- yI[0][v])**5 + 0.5**5) / (2 * (0.5**5))
                else:
                    t[v] = yI[0][v] 
            else:
                if j == M:  
                    t[v] =((0.5- yI[0][v])**5) / (2 * (0.5**5))
                else:
                    t[v] = yI[j-1][v]
                
                           
        return t
                   
    if M == 3:
        t = np.zeros(( 15, 15))
        for v in range(0,15):
            for w in range(0,15):
                if (yI[0][v][w] in [0.1,0.4])  or (yI[0][v][w] in [0.6,0.9]) :
                    if j == M:
                        t[v][w] = ((0.5- yI[0][v][w])**5 + 0.5**5) / (2 * (0.5**5))
                    else:
                        t[v][w] = yI[0][v][w]
                else:
                    if j == M:
                        t[v][w] = ((0.5- yI[0][v][w])**5 + (0.5- yI[1][v][w])**5) / (2 * (0.5**5))
                    else:
                        t[v][w] = yI[j-1][v][w]
        return t
    

###############################################################################
# Definition of g functions 
###############################################################################
def g(yI,G, m,l,n):                                                                                    
    
    if (l<m+1 or l >n):
        raise ValueError("Error: variable index out of bound in fuction g")
        
    theta = (2*np.pi*(l-(m+1)))/n 
    
    if G==0:
        return 0.2210
    elif G==1:
        s = np.zeros(len(yI[0]))
        for i in range(m):
            s = s + np.sin(1.5*np.pi*yI[i] + theta)
        return (1/(2*m))*s + 1/2
    elif G==2:
        s = np.zeros(len(yI[0]))
        for i in range(m):
            s = s + (yI[i]**2)*np.sin(4.5*np.pi*yI[i] + theta)
        return (1/(2*m))*s +1/2
    elif G==3:
        s = np.zeros(len(yI[0]))
        for i in range(m):
            s = s + np.cos(np.pi*yI[i] + theta)**2
        return (1/(m))*s
    elif G==4:
        return (1/(2*m))*sum(yI) *np.cos((4*np.pi/m)*sum(yI) + theta) + 1/2
    elif G==5:
        s = np.zeros(len(yI[0]))
        for i in range(m):
            s = s + np.sin(2*np.pi*yI[i] -1 + theta)**3
        return (1/(2*m))*s + 1/2
    elif G==6:
        s1 = np.zeros(len(yI[0]))
        for i in range(m):
            s1 = s1 + yI[i]**2
        s2 = np.zeros(len(yI[0]))
        for i in range(m):
            s2 = s2 + np.cos(11*np.pi*yI[i]+theta)**3
        
        num = -10 * np.exp(np.sqrt(((1/m)*s1))*(-2/5)) + 10 + np.exp(1) - np.exp((1/m)*s2)
        denum = (-10*np.exp(-2/5) - np.exp(-1) +10 + np.exp(1))
        return  num/denum 
    elif G==7:
        num =(1/m) * sum(yI) + np.exp( np.sin( (7*np.pi/m)*sum(yI)-np.pi/2 +theta) )  - np.exp(-1)
        return num/ (1- np.exp(-1) + np.exp(1))
    elif G==8:
        s = np.zeros(len(yI[0]))
        for i in range(m):
            s = s + abs(np.sin(2.5*np.pi*(yI[i]-0.5) + theta))
        return (1/m)*s
    elif G==9:
        s = np.zeros(len(yI[0]))
        for i in range(m):
            s = s + abs(np.sin(2.5*np.pi*yI[i] - np.pi/2 + theta))
        return (1/(2*m))*sum(yI) - (1/(2*m))*s + 1/2
    elif G==10:
        s = np.zeros(len(yI[0]))
        for i in range(m):
            s = s + np.sin((4*yI[i] - 2)*np.pi + theta)
        return (1/(2*(m**3))) * s**3 + 1/2
    else:
        raise ValueError("Error: G index is not valid")
###############################################################################
# Definition of alpha function
###############################################################################
# Map ZCAT values {1,...,20} to corresponding function names
def Fzcat(ZCAT, yI, i, M):
    zcat_functions = {
        1: Fzcat1,
        2: Fzcat2,
        3: Fzcat3,
        4: Fzcat4,
        5: Fzcat5,
        6: Fzcat6,
        7: Fzcat7,
        8: Fzcat8,
        9: Fzcat9,
        10: Fzcat10,
        11: Fzcat11,
        12: Fzcat12,
        13: Fzcat13,
        14: Fzcat14,
        15: Fzcat15,
        16: Fzcat16,
        17: Fzcat17,
        18: Fzcat18,
        19: Fzcat19,
        20: Fzcat20
    }

    try:
        return zcat_functions[ZCAT](yI, i, M)
    except KeyError:
        raise ValueError(f"Invalid ZCAT value: {ZCAT}. Must be in range 1–20.")
###############################################################################
# Definition of alpha function
###############################################################################
def alpha(yI, i, ZCAT,M):                             
  return (i**2) * Fzcat(ZCAT, yI, i, M)  


###############################################################################
# Function that plot pareto front and set
###############################################################################

# FS: 'F' plot Pareto Front, 'S' plot Pareto set
def Pareto_Front_Set(n_punti, ZCAT, M,G, n,m,FS):
    if M ==2:
        yI = [np.linspace(0, 1, n_punti)]
        a1 = alpha(yI, 1, ZCAT, M, m)
        a2 = alpha(yI, 2, ZCAT, M, m)
        a = np.transpose(np.array([a1, a2]))

        if (ZCAT == 11) or (ZCAT == 12) or (ZCAT == 13) or (ZCAT == 15) or (ZCAT == 16):
            a1g = np.gradient(a1)
            a2g = np.gradient(a2)
            pos = np.where(a2g/a1g > 0)[0]
            a1[pos] = np.nan
            
        
        
        if FS == 'F':
            #Objective space (Pareto front)
            fig = plt.figure(figsize=(7, 6))
            ax = fig.subplots()
            ax.plot(a1, a2, "k-")
            ax.set_xlabel("$f_1(\mathbf{x})$", fontsize=15)
            ax.set_ylabel("$f_2(\mathbf{x})$", fontsize=15)
            #ax.set_xlim(-0.02,1.02)
            #ax.set_ylim(-0.1,4.1)
            plt.title("Pareto Front")
            #plt.show()
        elif FS == 'S':
            if G == 0: # Not complicated Pareto Set
                z = g(yI,0, m,m+1,n)*np.ones(len(yI[0]))
            else:
                z = g(yI,G, m,m+1,n)


            x1 = yI[0] - 1/2
            x2 = 2*z - 1
        
            #Objective space (Pareto front)
            fig = plt.figure(figsize=(7, 6))
            ax = fig.subplots()
            ax.plot(x1, x2,"k-") 
            ax.set_title('Pareto Set')
            if G >0:
                ax.set_title('Complicated Pareto Set')
            ax.set_xlabel("$x_1$", fontsize=15)
            ax.set_ylabel("$x_2$", fontsize=15)
            #plt.show()


    if M ==3:
        
        if (ZCAT == 14 or ZCAT == 15 or ZCAT == 16):
            n_punti = 225
            yI1=np.linspace(0, 1, n_punti)
            yI2=np.linspace(0, 1, n_punti)
            
        
        else:
            (yI1, yI2) = np.meshgrid( np.linspace(0, 1, n_punti), np.linspace(0, 1, n_punti))
        
        yI = [yI1,yI2]
        a1 = alpha(yI, 1, ZCAT, M, m)
        a2 = alpha(yI, 2, ZCAT, M, m)
        a3 = alpha(yI, 3, ZCAT, M, m)
        a = np.concatenate((a1.reshape(-1,1), a2.reshape(-1,1), a3.reshape(-1,1)), axis=1)
        
        # if (ZCAT == 11) or (ZCAT == 12) or (ZCAT == 13) or (ZCAT == 15) or (ZCAT == 16):
        #     a1g = np.gradient(a1)
        #     a2g = np.gradient(a2)
        #     a3g = np.gradient(a3)
        #     pos = np.where(a2g/a1g > 0)[0]
        #     a1[pos] = np.nan

        if FS == 'F':
            if (ZCAT == 14 or ZCAT == 15 or ZCAT == 16):
                #Objective space (Pareto front)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                ax.plot(a1, a2,a3,'k')
                ax.set_xlabel("$f_1(\mathbf{x})$", fontsize=15)
                ax.set_ylabel("$f_2(\mathbf{x})$", fontsize=15)
                ax.set_zlabel("$f_3(\mathbf{x})$", fontsize=15)
                plt.title("Pareto Front")
                #plt.show()
            else:
                #Objective space (Pareto front)
                fig = plt.figure()
                ax = fig.add_subplot(111, projection='3d')
                ax.plot_surface(a1, a2, a3, color='w', edgecolor='k',alpha=0.1)
                 
                ax.set_title('Pareto Front')
                ax.set_xlabel("$f_1(\mathbf{x})$", fontsize=15)
                ax.set_ylabel("$f_2(\mathbf{x})$", fontsize=15)
                ax.set_zlabel("$f_3(\mathbf{x})$", fontsize=15)
                #ax.set_xlim(0,1.25)
                #plt.show()
        if FS == 'S':
            if G == 0: # Not complicated Pareto Set
                n_punti = 15
                (yI1, yI2) = np.meshgrid( np.linspace(0, 1, n_punti), np.linspace(0, 1, n_punti))
                yI = [yI1,yI2]
                z = g(yI,0, m,m+1,n)*np.ones(yI1.shape)
            else:
                z = g(yI,G, m,m+1,n)


            x1 = yI1 - 1/2
            x2 = 2*yI2 - 1
            x3 = 3*z - 3/2
            if (ZCAT == 14 or ZCAT == 15 or ZCAT == 16):
                #Objective space (Pareto front)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                if G == 0:
                    ax.plot_surface(x1, x2, x3,color='w', edgecolor='k',alpha=0.1) 
                    ax.set_title('Pareto Set')
                else:
                    ax.plot(x1, x2,x3,'k')
                    ax.set_title('Complicated Pareto Set')
                ax.set_xlabel("$x_1$", fontsize=15)
                ax.set_ylabel("$x_2$", fontsize=15)
                ax.set_zlabel("$x_3$", fontsize=15)
                #plt.show()
            else:
               #Objective space (Pareto front)
               fig = plt.figure()
               ax = fig.add_subplot(111, projection='3d')
               ax.plot_surface(x1, x2, x3,color='w', edgecolor='k',alpha=0.1) 
               ax.set_title('Pareto Set')
               if G >0:
                   ax.set_title('Complicated Pareto Set')
               ax.set_xlabel("$x_1$", fontsize=15)
               ax.set_ylabel("$x_2$", fontsize=15)
               ax.set_zlabel("$x_3$", fontsize=15)
               #plt.show()
            
        
    return fig, ax, a


