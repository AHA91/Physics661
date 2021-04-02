#Alex Andronikides
#Assignment #5

import numpy as np
import scipy.integrate as integrate 
import matplotlib.pyplot as plt
import pandas as pd

#################### Numerical Differentiation ###################

################ Part 1 ################

def FDM(f, x, h):
    return (f(x+h)-f(x))/h       


def BDM(f, x, h):
    return (f(x)-f(x-h))/h
    

def CDM(f, x, h):
    return (f(x+h)-f(x-h))/2*h

################ Part 2 ################

def MOC(f, x, h, Method = 'choice'):
    if Method == 'FDM':
        return(f(x+h)-f(x))/h

    elif Method == 'BDM':
        return(f(x)-f(x-h))/h
    
    elif Method == 'CDM':
        return(f(x+h)-f(x-h))/2*h
        
    else:
        print('Error: Must be FDM, BDM, or CDM')

################## Part 3 ###################

def Part3():
    dr1 = []
    dr2 = []
    dr3 = []
    h1 = np.linspace(10**(-15), 10**(-10), 1000)
    h2 = np.linspace(10**(-10), 10**(-5), 10**6)
    h3 = np.linspace(10**(-5),10**(0), 10**7)
    dr1.append(FDM(lambda x: np.exp(x), 0, h1))
    dr2.append(FDM(lambda x: np.exp(x), 0, h2))
    dr3.append(FDM(lambda x: np.exp(x), 0, h3))

    dr1 = np.array(dr1)
    dr2 = np.array(dr2)
    dr3 = np.array(dr3)

    Err1 = np.array(abs(dr1[0]-1))
    Err2 = np.array(abs(dr2[0]-1))
    Err3 = np.array(abs(dr3[0]-1))

    plt.loglog(h1, Err1, color = 'blue')
    plt.loglog(h2, Err2, color = 'blue')
    plt.loglog(h3, Err3, color = 'blue')
    plt.title('f(x) = exp(x)')
    plt.xlabel('h')
    plt.ylabel('Error')

    plt.show()

#################### Numerical Integration ###################

################ Part 1 & 3 ################

def LSM(f, a, b, N):
    LS = 0
    LS_ = []
    x0 = a
    x1 = b
    x = np.linspace(x0, x1,N)
    for i in range(N-1):
        dx = x[i+1] - x[i]
        LS += f(x[i-1])*dx
        LS_.append(LS)

    I = integrate.quad(f, a, b)
    I_ = float(I[0])*(np.ones(len(LS_)))
    ER = abs(LS_ - I_)

    df = pd.DataFrame({'N' : range(N-1), 'Estimated value (LSM)' :LS_,'Calculated value': I_,'Error':ER})
    print(df)

def RSM(f, a, b, N):
    RS = 0
    RS_ = []
    x0 = a
    x1 = b
    x = np.linspace(x0, x1, N)
    for i in range(2,N):
        dx = x[i] - x[i-1]
        RS += f(x[i])*dx
        RS_.append(RS)

    I = integrate.quad(f, a, b)
    I_ = float(I[0])*(np.ones(len(RS_)))
    ER = abs(RS_ - I_)

    df = pd.DataFrame({'N' : range(2,N), 'Estimated value (RSM)' :RS_,'Calculated value': I_,'Error':ER})
    print(df)


def TRP(f,a,b,N):
    LS = 0
    LS_= []
    RS = 0
    RS_= []
    x0 = a
    x1 = b
    x = np.linspace(x0, x1, N)

    for i in range(N-1):
        dx = (x[i+1] - x[i])
        LS += float(f(x[i-1])*dx)
        LS_.append(LS)
    for i in range(2,N):
        dx = (x[i] - x[i-1])
        RS += float(f(x[i])*dx)
        RS_.append(RS)

    SUM = (RS_+LS_)
    print(SUM)

    I = integrate.quad(f, a, b)
    I_ = float(I[0])*(np.ones(len(SUM)))
    ER = abs(SUM - I_)

    df = pd.DataFrame({'N' : range(len(SUM)), 'Estimated value (TRP)' : SUM ,'Calculated value': I_,'Error':ER})
    print(df)  


def MPM(f, a, b, N):
    MP = 0
    MP_ = []
    x0 = a
    x1 = b
    h = (b-a)/N
    x = np.linspace(x0, x1, N)
    for i in range(2,N):
        X = (x[i] + x[i-1])/2
        MP += h*f(X)
        MP_.append(MP)

    I = integrate.quad(f, a, b)
    I_ = float(I[0])*(np.ones(len(MP_)))
    ER = abs(MP_ - I_)
    
    df = pd.DataFrame({'N' : range(2,N), 'Estimated value (MPM)' : MP ,'Calculated value': I_,'Error':ER})
    print(df)




################ Part 2 ################

def MOCI(f, a, b, N, Method = 'choice'):    
    x0 = a
    x1 = b
    x = np.linspace(x0, x1,N)
    
    if Method == 'LSM':
        LS = 0
        for i in range(N-1):
            dx = x[i+1] - x[i]
            LS += f(x[i-1])*dx
        return LS


    elif Method == 'RSM':
        RS = 0
        for i in range(2,N):
            dx = x[i] - x[i-1]
            RS += f(x[i])*dx
        return RS

    elif Method == 'TRP':
        RS = 0
        LS = 0
        for i in range(N-1):
            dx = x[i+1] - x[i]
            LS += f(x[i-1])*dx
            
        for i in range(2,N):
            dx = x[i] - x[i-1]
            RS += f(x[i])*dx

        return (RS+LS)*(0.5)


    elif Method == 'MPM':
        MP = 0
        h = (b-a)/N
        for i in range(2,N):
            X = (x[i] + x[i-1])/2
            MP += h*f(X)
        return MP

    else:
        print('Error: Must be LSM, RSM, TRP, or MPM')
