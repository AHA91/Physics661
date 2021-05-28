####### Alex Andronikides
####### Phys 661


import random as rand
import numpy as np
import matplotlib.pyplot as plt

###################### Part 1 ##################

def Estpi2D(N):
    
    N_in = 0
    N_out = 0
    t = np.linspace(0,2*np.pi, 100)              

    x = [rand.uniform(-1,1) for i in range(0,N)]     
    y = [rand.uniform(-1,1) for i in range(0,N)]
        
    for i in range(len(x)):
        if np.sqrt(x[i]**2 + y[i]**2) <= 1:
            N_in = N_in + 1
            plt.plot(x[i],y[i],"o", color = "green")
        else:
            N_out = N_out+1
            plt.plot(x[i],y[i],"o", color = "red")
     ### STD
    

    EstPi = N_in*4/N
    Err = (abs(EstPi-np.pi)/np.pi)*100

    print("Estimating Pi:",EstPi)
    print("Error:", Err)

    plt.plot(np.cos(t), np.sin(t))
    plt.show()
  
####################### Part 2 ###################

def Estpi3D(N):
    
    N_in = 0
    N_out = 0
    fig = plt.figure()
    ax = fig.add_subplot(projection = '3d')
    
    
    x = [rand.uniform(0,1)for i in range(0,N)]
    y = [rand.uniform(0,1)for i in range(0,N)]
    z = [rand.uniform(0,1)for i in range(0,N)]

        
    for i in range(len(x)):
        if np.sqrt(x[i]**2 + y[i]**2 + z[i]**2) <= 1:
            N_in = N_in + 1
            ax.scatter(x[i],y[i],z[i],"o", color = "green")
        else:
            N_out = N_out+1
            ax.scatter(x[i],y[i],z[i],"o", color = "red")
        
    EstPi = N_in*4/N
    Err = (abs(EstPi-np.pi)/np.pi)*100

    print("Estimating Pi:",EstPi)
    print("Error:", Err)

    u,v = np.mgrid[0:2*np.pi/4:20j, 0:np.pi/2:20j]
    ax.plot_surface(np.cos(u)*np.sin(v), np.sin(u)*np.sin(v), np.cos(v), alpha = 0.3)
    
    plt.show()
    




