import numpy as np
import matplotlib.pyplot as plt


def Orbit(x0,y0,vx0,vy0,m1,m2,G,t_,dt):
    t = np.arange(0,t_,dt)

    x = np.zeros(len(t))
    y = np.zeros(len(t))
    vx = np.zeros(len(t))
    vy = np.zeros(len(t))
    ax = np.zeros(len(t))
    ay = np.zeros(len(t))
    L = np.zeros(len(t))
    E = np.zeros(len(t))

    x[0] = x0
    y[0] = y0
    vx[0] = vx0
    vy[0] = vy0
    ax[0] = -x0/(np.sqrt(x0**2+y0**2))**3
    ay[0] = -y0/(np.sqrt(x0**2+y0**2))**3

    for i in range(len(t)-1):
        x[i+1] = x[i] + vx[i]*dt + (0.5)*ax[i]
        y[i+1] = y[i] + vy[i]*dt + (0.5)*ay[i]

        ax[i+1] = -m1*G*(x[i+1])/(np.sqrt(x0**2+y0**2))**3
        ay[i+1] = -m1*G*(y[i+1])/(np.sqrt(x0**2+y0**2))**3

        vx[i+1] = vx[i] + (0.5)*(ax[i]+ax[i+1])*dt
        vy[i+1] = vy[i] + (0.5)*(ay[i]+ay[i+1])*dt

    L = vx*np.sqrt(x**2+y**2)
    E = (1/2)*m1*(np.sqrt(vx**2+vy**2))**2-G*m1*m2/(np.sqrt(x**2+y**2))
    
    fig, (ax1, ax2, ax3) = plt.subplots(3)

    ax1.plot(x,y)
    ax1.set_title('Orbit')
    ax1.set_xlabel('x')
    ax1.set_ylabel('y')

    ax2.plot(t,L)
    ax2.set_xlabel('time')
    ax2.set_ylabel('Angular Momentum')

    ax3.plot(t,E)
    ax3.set_xlabel('time')
    ax3.set_ylabel('Total Energy')


    plt.show()

        
                   


    
