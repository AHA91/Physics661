import numpy as np
import matplotlib.pyplot as plt

def projectile1(v_0, theta,n):
    g = 9.8
    theta = np.radians(theta)
    x_=[]
    y_=[]

    
    t = np.linspace(0, 50, n)

    for i in t:
        x = v_0*i*np.cos(theta)
        y = v_0*i*np.sin(theta) - (0.5)*g*i**2
            
        x_.append(x)
        y_.append(y)

    
    plt.plot(x_,y_)
    #plt.ylim([0,max(y_)*0.5])
    #plt.xlim([0, max(x_)*0.5])
    plt.show()


def projectile2(v_0,theta,b,dt,n):   ### Incomplete 
    g = 9.8
    theta = np.radians(theta)
    x_=[]
    y_=[]





       

def Gravity(v0,r0, a_r0,dt,n):
    r1 = r0+v0*dt+(a_r0*dt**2)/2
    r_m1 = 0
    r_p1 = 2*r1-r_m1 + a_r0*dt**2
    r = []
    v = []

    for i in range(n):
        ar= (r_p1 - 2*r1 + r_m1)/dt
        v_p1 = v0+((a_r0+ar)/2)*dt
        r_m1 = r1
        r1=r_p1
        r.append(r_p1)
        v.append(v_p1)
        
    plt.plot(v,r)
    plt.show()


def Spring1(k,m,x,v0,dt,N,n):       ###Not working
    t = np.linspace(0,10,N)
    a = []
    v = []
    x = []

    for i in range(1,n):
        a = -(k/m)*x
        v[i] = v[i-1]+a*dt
        x[i] = x[i-1]+v[i-1]*dt

        plt.plot(x,t)
        plt.show()

    
        

        
        
    
    

    
    
    
