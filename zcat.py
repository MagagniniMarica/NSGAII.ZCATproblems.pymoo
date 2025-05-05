
###############################################################################
#  Function to set G
###############################################################################
# flag_CP : flag to turn on the complex Pareto set
# ZCAT : index of the specific problem {1,2,3,..,19,20}
def setG(flag_CP, ZCAT):
    if not flag_CP:
        return 0

    zcat_to_value = {
        1: 4,  2: 5,  3: 2,  4: 7,  5: 9,
        6: 4,  7: 5,  8: 2,  9: 7, 10: 9,
        11: 3, 12: 10, 13: 1, 14: 6, 15: 8,
        16: 10, 17: 1, 18: 8, 19: 6, 20: 3
    }

    return zcat_to_value.get(ZCAT, 0)
###############################################################################
# Definition of F functions
###############################################################################
# yI normalized position variables vector
# i in {1,..., M} objective function index
# M number of objective functions
"""yI must be an np.array"""
def Fzcat1(yI, i,M):
    if(i==1):
        return np.prod(np.sin(yI*np.pi/2))
    if(i==M):
        return 1- np.sin(yI[0]*np.pi/2)
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:                                               
        return np.prod(np.sin(yI[:M-i]*np.pi/2)) * np.cos(yI[M-i]*(np.pi/2))

def Fzcat2(yI, i,M):
    if(i==1):
        return np.prod(1-np.cos(yI*np.pi/2))
    if(i==M):
        return 1- np.sin(yI[0]*np.pi/2)
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return (np.prod(1- np.cos(yI[:M-i]*np.pi/2))) * (1-np.sin(yI[M-i]*np.pi/2))

def Fzcat3(yI, i,M):
    if(i==1):
        return (1/(M-1))*np.sum(yI)
    if(i==M):
        return 1- yI[0]
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return (1/(M-i+1))*(np.sum(yI[:M-i])+(1-yI[M-i]))  

def Fzcat4(yI, i,M):
    if(i==M):
        return 1-(1/(M-1))*np.sum(yI)
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return yI[i-1]

def Fzcat5(yI, i,M):
    if(i==M):
        return (np.exp((1/(M-1))*np.sum(1-yI))**8 -1) / (np.exp(1)**8 - 1)
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return yI[i-1]

def Fzcat6(yI, i,M):
    if(i==M):
        ro = 0.05
        k = 40
        mu = (1/(M-1))*sum(yI)
        
        a = 1 / (1 + np.exp(2*k*mu-k))
        b = 1/(1+np.exp(k))
        c = 1/(1+np.exp(-k))
        return (a - ro*mu -b +ro) / (c - b + ro)
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return yI[i-1]

def Fzcat7(yI, i,M):
    if(i==M):
        return (1/(2*(M-1)*(0.5**5)))*np.sum((0.5-yI)**5) +1/2
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return yI[i-1]

def Fzcat8(yI, i,M):
    if(i==1):
        return 1- np.prod(1- np.sin(yI*np.pi/2))
    if(i==M):
        return np.cos(yI[0]*np.pi/2)
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return 1 - np.prod(1- np.sin(yI*np.pi/2))*(1-np.cos(yI[M-i]*np.pi/2))

def Fzcat9(yI, i,M):
    if(i==1):
        return (1/(M-1))*np.sum(np.sin(yI*np.pi/2))
    if(i==M):
        return np.cos(yI[0]*np.pi/2)
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return ( 1/(M-i+1) ) * ( np.sum(np.sin(yI[:M-i]*np.pi/2)) + np.cos(yI[M-i]*np.pi/2) )

def Fzcat10(yI, i,M):
    if(i==M):
        ro = 0.02
        return (1/ro - ((1/(M-1))*np.sum(1-yI) + ro)**(-1) ) / (1/ro - 1/(1+ro))
    if(i>M):
        raise ValueError("Error: obiective function index out of bound")
    else:
        return yI[i-1]

def Fzcat11(yI,i,M):
    if(i==1):
        return (1/(M-1))*np.sum(yI)
    if(i==M):
        K=4 #4 dal codice, 5 da tabella 4 nel paper
        return (np.cos((2*K-1)*yI[0]*np.pi) + 2*yI[0] + 4*K*(1-yI[0]) -1)/(4*K)
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return (1/(M-i+1))*(np.sum(yI[:M-i])+(1-yI[M-i]))  
    
def Fzcat12(yI,i,M):
    if(i==1):
        return 1- np.prod(1-yI)
    if(i==M):
        K=3 # dal codice
        return (np.cos((2*K-1)*yI[0]*np.pi) + 2*yI[0] + 4*K*(1-yI[0]) -1)/(4*K)
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return 1- np.prod(1-yI)*yI[M-i]

def Fzcat13(yI,i,M):
    if(i==1):
        return 1- (1/(M-1))*np.sum(np.sin(yI*np.pi/2))
    if(i==M):
        K=3 # dal codice
        return 1- ((np.cos((2*K-1)*yI[0]*np.pi) + 2*yI[0] + 4*K*(1-yI[0]) -1)/(4*K))
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return 1- (1/(M-i+1)) * (np.sum(np.sin(yI*np.pi/2)) + np.cos(yI[M-i]*np.pi/2))

def Fzcat14(yI,i,M):
    if(i==1):
        return np.sin(yI[0]*np.pi/2)**2
    if(M>2 and i==M-1 ):
        return 0.5*(1+np.sin(6*yI[0]*np.pi/2 - np.pi/2))
    if(i==M):
        return np.cos(yI[0]*np.pi/2)
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return np.sin(yI[0]*np.pi/2)**(2 + (i-1)/(M-2))

def Fzcat15(yI,i,M):
    if(i==M):
        K=3 
        return  ((np.cos((2*K-1)*yI[0]*np.pi) + 2*yI[0] + 4*K*(1-yI[0]) -1)/(4*K))
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return yI[0]**(1+(i-1)/(4*M))

def Fzcat16(yI,i,M):
    if(i==1):
        return np.sin(yI[0]*np.pi/2)
    if(M>2 and i==M-1 ):
        return 0.5*(1+np.sin(10*yI[0]*np.pi/2 - np.pi/2))
    if(i==M):
        K=5 
        return ((np.cos((2*K-1)*yI[0]*np.pi) + 2*yI[0] + 4*K*(1-yI[0]) -1)/(4*K))
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        return np.sin(yI[0]*np.pi/2)**(2 + (i-1)/(M-2))
    
def Fzcat17(yI,i,M):
    if i == M:
        if (yI <= 0.5).all():
            return (np.exp(1-yI[0])**8 -1)/(np.exp(1)**8 -1)
        else:
            return (np.exp((1/(M-1))*np.sum(1-yI))**8 -1) /  (np.exp(1)**8 -1)
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        if (yI <= 0.5).all():
            return yI[0]
        else:
            return yI[i-1]

def Fzcat18(yI,i,M):
    if i == M:
        if (yI <= 0.4).all() or (yI >= 0.6).all():
            return ((0.5-yI[0])**5 + 0.5**5) / (2*(0.5**5))
        else:
            return np.sum((0.5-yI)**5) / (2*(M-1)*(0.5**5)) + 1/2
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        if (yI <= 0.4).all() or (yI >= 0.6).all():
            return yI[0]
        else:
            return yI[i-1]        

def Fzcat19(yI,i,M):
    if (yI[0] in [0,0.2]) or  (yI[0] in [0.4,0.6]):
        m = 1
    else:
        m = M-1
    if(i==M):
        if m ==1:
            return 1-yI[0]- (np.cos(10*np.pi*yI[0] + np.pi/2)/(10*np.pi))
        else:
            return 1- (1/(M-1))* np.sum(yI) - (np.cos( (10*np.pi/(M-1)) * np.sum(yI) + (np.pi/2) )/(10*np.pi))
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        if m ==1:
            return yI[0]
        else:
            return yI[i-1]

def Fzcat20(yI,i,M):
    if (yI[0] in [0.1,0.4]) or  (yI[0] in [0.6,0.9]):
        m = 1
    else:
        m = M-1
    if(i==M):
        if m ==1:
            return ((0.5- yI[0])**5 + 0.5**5) / (2 * (0.5**5))
        else:
            return np.sum((0.5 - yI)**5)/(2*(M-1)*(0.5**5)) + 0.5
    if(i>M):
        raise ValueError("Error: objective function index out of bound")
    else:
        if m==1:
            return yI[0]
        else:
            return yI[i-1]

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
# Definition of g functions 
###############################################################################
""""Set g
    m = M-1,
    G in {0,..,10}, 
    l = m+1,..,n,
    yI normalized position variables vector
"""
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
"""
If "flag_bias" true --> bias is applied to vector zII = yII-g(yI),
If false, no bies is applied
"""
def Bias(flag_bias,zII):
    if (flag_bias == True):
        z_bias = np.zeros(len(zII))
        for z in range(len(zII)):
            z_bias[z] = abs(zII[z])**0.05
        return z_bias
    else:
        return zII
        

###############################################################################
# Definition of Z_level functions 
###############################################################################
"""
i = 1,...,M,
w  = Bias(flag_bias,zII)
flag_DL in {1,...,6}
n total number of variables
M number of objective fucntions
m nuber of position variables
"""
def Z_level(w,i, flag_DL,m,n,M):
    J = []
    for l in range(m+1, n+1):
        if (l-m-i)%M ==0:
            J.append(l-m-1)
    
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
        return -0.7* Z_level(w, i, 3, m, n, M) + (10/len(J))*sum(abs(s)**0.002 for s in w if (list(w).index(s)) in J)
    if flag_DL == 6:
        return -0.7* Z_level(w, i, 4, m, n, M) + 10*((1/len(J))*sum(abs(s)for s in w if (list(w).index(s)) in J))**0.002 
 
        
###############################################################################
# Definition of alpha function
###############################################################################
def alpha(yI, i, ZCAT,M):                             
  return (i**2) * Fzcat(ZCAT, yI, i, M):  

###############################################################################
# Definition of beta function
###############################################################################
 #if flag_imbalanced true, inbalance is introduced into beta functions. 
def beta(flag_imbalanced,w, i,flag_DL,m,n,M): 
    if flag_imbalanced == True:
        if(i%2 ==0):
            return (i**2) *  Z_level(w,i, 4,m,n,M)
        else: 
            return (i**2) *  Z_level(w,i, 1,m,n,M)
    else:                                                                      
        return (i**2) *  Z_level(w,i, flag_DL,m,n,M)     

###############################################################################
 # Definition of f objective function 
###############################################################################

def f(yI, i, ZCAT, flag_imbalanced,w,flag_DL,m,n,M):  
    return alpha(yI, i, ZCAT,M) + beta(flag_imbalanced, w, i, flag_DL, m, n, M)
