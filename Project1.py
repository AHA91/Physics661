import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from celluloid import Camera
from mpl_toolkits import mplot3d
import itertools
import random
##########################################Kramers-Kronig Relations
def KKR():
    
    df = np.array(pd.read_csv('extinctionSpectrum.csv'))
    f = (df[:,0]/(3*10**8))*2*np.pi
    H = []
    n = 0
    for j in range(1000): 
        for i in range(999):
            if f[i] != f[j]:
                n = n + ((f[i]*df[i,1])/((f[i]**2)-(f[j]**2)))*(f[i+1]-f[i])
            else:
                n = n + 0
        h = (2/np.pi)*n    ### Why are we adding 1?
        H.append(h)
        n = 0
    plt.plot(f,H)
    plt.plot(f, df[:,1])
    plt.show()

########################################## Random Walk
def RW1D(M,h,Method = 'Method'):
    x = np.zeros(M)
    dX = []
    dX2 = []
    avgdx = []
    H = []
    if Method == '1':
        for N in range(1,M,h):
            n = int(np.sqrt(N))     
            for j in range(n):
                x = sum(random.choices([-1,1],k=N))
                dX_ = x
                dX2_ = dX_**2
                H.append(dX2_)
                dX.append(dX_)
            avgdx.append(np.average(dX))
            dX2.append(np.average(H))
            dX.clear()
            H.clear()
        z = 'Random Choice -1 or 1'
    if Method == '2':
        for N in range(1,M,h):
            n = int(np.sqrt(N))     
            for j in range(n):
                x = sum(np.random.uniform(-1,1,N))
                dX_ = x
                dX2_ = dX_**2
                H.append(dX2_)
                dX.append(dX_)
            avgdx.append(np.average(dX))
            dX2.append(np.average(H))
        z = 'Uniform Distribution'
    if Method == '3':
        for N in range(1,M,h):
            n = int(np.sqrt(N))     
            for j in range(n):
                x = sum(np.random.normal(0,1,N))
                dX_ = x
                dX2_ = dX_**2
                H.append(dX2_)
                dX.append(dX_)
            avgdx.append(np.average(dX))
            dX2.append(np.average(H))
        z = 'Normal Distribution'

    plt.plot(range(1,M,h), avgdx, label = '<x>')
    plt.plot(range(1,M,h), range(1,M,h), label = ' Theoretical <x2>')
    plt.title(z)
    plt.plot(range(1,M,h), dX2, label = '<x2>')
    plt.xlabel('N')
    plt.legend()
    plt.show()


def RW2D(M,H, Method = 'Method'):
    x = np.zeros(M)
    y = np.zeros(M)
    dX = []
    dY = []
    dR = []
    dR2 = []
    r = []
    r2 = []
    avgdR = []
    avgdR2 = []


    if Method == 'Cartesian Uniform':
        for N in range(1,M,H):
            n = int(np.sqrt(N))     
            for j in range(n):
                x = sum(np.random.uniform(-1,1,N))
                y = sum(np.random.uniform(-1,1,N))
                dX = x
                dY = y
                dR_  = np.sqrt(dX**2+dY**2)
                dR2_ = dR_**2
                dR2.append(dR2_)
                dR.append(dR_)
            avgdR.append(np.average(dR))
            avgdR2.append(np.average(dR2))
            dR.clear()
            dR2.clear()

    if Method == 'Cartesian Normal':
        for N in range(1,M,H):
            n = int(np.sqrt(N))     
            for j in range(n):
                x = sum(np.random.normal(-1,1,N))
                y = sum(np.random.normal(-1,1,N))
                dX = x
                dY = y
                dR_  = np.sqrt(dX**2+dY**2)
                dR2_ = dR_**2
                dR2.append(dR2_)
                dR.append(dR_)
            avgdR.append(np.average(dR))
            avgdR2.append(np.average(dR2))
            dR.clear()
            dR2.clear()

            
    
    if Method == 'Polar Uniform':
        for N in range(1,M,H):
            n = int(np.sqrt(N))     
            for j in range(n):
                theta = np.random.uniform(0,2*np.pi, N)
                dX = (np.cos(theta))
                dY = (np.sin(theta))
                tmp = np.sqrt(dY**2+dX**2)
                r_ = sum(np.sqrt(dY**2+dX**2))
                r2_ = sum(tmp**2)
                r.append(r_)
                r2.append(r2_)
            avgdR.append(np.average(r))
            avgdR2.append(np.average(r2))
            r.clear()
            r2.clear()

    if Method == 'Polar Normal':
        for N in range(1,M,H):
            n = int(np.sqrt(N))     
            for j in range(n):
                theta = np.random.normal(0,2*np.pi, N)
                dX = (np.cos(theta))
                dY = (np.sin(theta))
                tmp = np.sqrt(dY**2+dX**2)
                r_ = sum(np.sqrt(dY**2+dX**2))
                r2_ = sum(tmp**2)
                r.append(r_)
                r2.append(r2_)
            avgdR.append(np.average(r))
            avgdR2.append(np.average(r2))
            r.clear()
            r2.clear()
    
    plt.title(Method)   
    plt.plot(range(1,M,H), avgdR, label = '<R>')
    plt.plot(range(1,M,H), avgdR2, label = '<R2>')
    plt.xlabel('N')
    plt.legend()
    plt.show()
    

############################################### Brownian Motion
def Brownian(T,dt,k, Method = 'Method'):
    dt = dt/k
    M = int(T/dt)
    x = np.zeros(M)
    dX = []
    dX2 = []
    avgdx = []
    H = []

    if Method == 'Random Choice': ##Brownian(100,0.1,1,"Random Choice")
        for N in range(1,M):
            n = int(np.sqrt(M))
            for j in range(n):
                x = sum(random.choices([-np.sqrt(1/2),np.sqrt(1/2)],k=N))
                dX_ = x
                dX2_ = N*dX_**2
                H.append(dX2_)
                dX.append(dX_)
            avgdx.append(np.average(dX))
            dX2.append(np.average(H))
            dX.clear()
            H.clear()
    if Method == 'Normal Distribution':  ##Brownian(100,0.1,1,"Normal Distribution")
        for N in range(1,M):
            n = int(np.sqrt(M))
            for j in range(n):
                x = sum(np.random.normal(0,1,N))
                dX_ = x
                dX2_ = dX_**2
                H.append(dX2_)
                dX.append(dX_)
            avgdx.append(np.average(dX))
            dX2.append(N*np.average(H))
            dX.clear()
            H.clear()
        
    plt.plot(range(1,M),dX2)
    plt.title(Method)
    plt.xlabel('N')
    plt.ylabel('<x2>')
    plt.show()

def Brownian3D(Method = 'Method'):
    S = np.zeros((100,3),float)
    A = []
    r_ = []

    if Method == 'Choice':
        for j in range(25):
            for i in range(100):
                r = random.choices([1,2,3])
                if r == [1]:
                    x = random.choices([-np.sqrt(1/2),np.sqrt(1/2)])
                    S[i,0] = x[0]
           
                if r == [2]:
                    y = random.choices([-np.sqrt(1/2),np.sqrt(1/2)])
                    S[i,1] = y[0]
              
                if r == [3]:
                    z = random.choices([-np.sqrt(1/2),np.sqrt(1/2)])
                    S[i,2] = z[0]
            a = np.cumsum(S,axis = 0)
            A.append(a)
            S = np.zeros((100,3),float)
           
    if Method == 'Uniform':
         for j in range(25):
            for i in range(100):
                r = random.choice([1,2,3])
                if r == [1]:
                    x = np.random.uniform(-1,1,100)
                    S[i,0] = x[0]
           
                if r == [2]:
                    y = np.random.uniform(-1,1,100)
                    S[i,1] = y[0]
              
                if r == [3]:
                    z = np.random.uniform(-1,1,100)
                    S[i,2] = z[0]
              
            a = np.cumsum(S,axis = 0)
            A.append(a)
            S = np.zeros((100,3),float)
        



    
    fig, ax = plt.subplots(subplot_kw={'projection':'3d'})
    camera = Camera(fig)

    for i in range(len(a[:,0])):
        for j in range(len(A)):
            a_ = A[j]
            ax.plot(a_[i,0],a_[i,1],a_[i,2],'o')
        camera.snap()
    animation = camera.animate()

    plt.show()
    
                
    

   

    

    
