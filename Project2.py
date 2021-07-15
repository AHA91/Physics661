import numpy as np
import matplotlib.pyplot as plt
from celluloid import Camera
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D 

##################################################### Projectile  

def projectile(v0,theta,b,b0,beta,dt):   ##projectile(10,45,10,0.9,0.5,0.01)
    theta  = np.radians(theta)
    g = 9.81
    t_= 2*v0*np.sin(theta)/g
    t = np.linspace(0, t_ ,1000) 
    #### No resistance
    
    y1 = v0*np.sin(theta)*t - (0.5)*g*t**2
    x1 = v0*np.cos(theta)*t
    
    plt.plot(x1,y1, label = 'No resistance')

    #### Air resistance

    
    x2 = np.zeros(len(t))
    y2 = np.zeros(len(t))
    vx = np.zeros(len(t))
    vy = np.zeros(len(t))
    ax = np.zeros(len(t))
    ay = np.zeros(len(t))

    vx[0] = v0*np.cos(theta)
    vy[0] = v0*np.sin(theta)

    x2[0]=0
    y2[0]=0

    ##### linear air resistance
    for i in range(len(t)-1):
        ay[i+1] = -b*vy[i]-g
        ax[i+1] = -b*vx[i]

        vy[i+1] = vy[i]+ay[i+1]*dt
        vx[i+1] = vx[i]+ax[i+1]*dt

        x2[i+1] = x2[i] + vx[i+1]*dt
        y2[i+1] = y2[i] + vy[i+1]*dt

    plt.plot(x2,y2,label= 'Linear drag')

    b = np.zeros(len(t))

    ### Exponential resistance
    for i in range(len(t)-1):
        b[i+1] = b0*np.exp(-beta*y2[i])
 
        ay[i+1] = -b[i]*vy[i]-g
        ax[i+1] = -b[i]*vx[i]

        vy[i+1] = vy[i]+ay[i+1]*dt
        vx[i+1] = vx[i]+ax[i+1]*dt

        x2[i+1] = x2[i] + vx[i+1]*dt
        y2[i+1] = y2[i] + vy[i+1]*dt
        

    plt.plot(x2,y2, label = 'Altitude dependence')

    plt.ylim([0,max(y1)])
    plt.legend()
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Projectile Motion') 
    plt.show()


################################################### Orbits 
def Orbit(x0,y0,vx0,vy0,m1,m2,G,t_,dt):  ##Orbit(1,3,7,6,20,1,1,100,0.5)
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
    ax[0] = -x0*G*m1/(np.sqrt(x0**2+y0**2))**3
    ay[0] = -y0*G*m1/(np.sqrt(x0**2+y0**2))**3

    for i in range(len(t)-1):
        x[i+1] = x[i] + vx[i]*dt + (0.5)*ax[i]*dt**2
        y[i+1] = y[i] + vy[i]*dt + (0.5)*ay[i]*dt**2

        ax[i+1] = -m1*G*(x[i+1])/(np.sqrt(x[i+1]**2+y[i+1]**2))**3
        ay[i+1] = -m1*G*(y[i+1])/(np.sqrt(x[i+1]**2+y[i+1]**2))**3

        vx[i+1] = vx[i] + (0.5)*(ax[i]+ax[i])*dt
        vy[i+1] = vy[i] + (0.5)*(ay[i]+ay[i])*dt

        L[i+1] = x[i+1]*vy[i+1]-y[i+1]*vx[i+1]
        E[i+1] = (1/2)*m1*(vx[i+1]**2+vy[i+1]**2)- G*m1*m2/(np.sqrt(x[i+1]**2+y[i+1]**2))
        
    
    fig, (ax1, ax2, ax3) = plt.subplots(3)

    camera = Camera(fig)
    for i in range(len(x)):
        ax1.plot(x,y, color = 'Blue')
        ax1.plot(x[i],y[i],'o', color = "Green")
        camera.snap()
    animation = camera.animate()

    ax2.plot(t,L)
    ax2.set_xlabel('time')
    ax2.set_ylabel('Angular Momentum')

    ax3.plot(t,E)
    ax3.set_xlabel('time')
    ax3.set_ylabel('Total Energy')
    
    plt.show()


########################################################## Ocsillators 
def SingleOscillator(x0,m,k,dt,Method = 'Method'): ##SingleOscillator(-10,1,5,0.1,"  ")
    t  = np.arange(0,10,dt)
    x  = np.zeros(len(t))
    v  = np.zeros(len(t))
    a  = np.zeros(len(t))
    KE = np.zeros(len(t))
    PE = np.zeros(len(t))
    E  = np.zeros(len(t))
    

    x[0] = x0
    #v[0] = v0


    if Method == "E1":   #Euler Method 
                for i in range(0, len(t)-1):
                    x[i+1] = x[i] + v[i]*dt
                    v[i+1] = v[i]-(k/m)*x[i]*dt
                    a[i+1] = -(k/m)*x[i]*dt

                    KE[i] = (1/2)*m*v[i]**2
                    PE[i] = (1/2)*k*x[i]**2
                    E[i]  = KE[i]+PE[i]

     
    if Method == "E2":   #Semi-Implicit Method 
                for i in range(0, len(t)-1):
                    a[i+1] = -(k/m)*x[i]
                    v[i+1] = v[i]-(k/m)*x[i]*dt
                    x[i+1] = x[i] + v[i+1]*dt

                    KE[i] = (1/2)*m*(v[i])**2
                    PE[i] = (1/2)*k*(x[i])**2
                    E[i]  = KE[i]+PE[i]
    
    fig, (ax1, ax2, ax3) = plt.subplots(3)

    camera = Camera(fig)
    for i in range(len(x)):
        ax1.plot(t,x, color = 'Blue')
        ax1.plot(t[i],x[i],'o', color = "Green")
        camera.snap()
    animation = camera.animate()
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position")

    ax2.plot(t,KE,'-' , label ='Kinetic Energy')
    ax2.plot(t,PE,'-', label = 'Potential Energy')
    ax2.plot(t,E,'-', label = 'Total Energy')
    ax2.legend()

    ax3.plot(x,v,label='Phase Space')
    ax3.set_xlabel('Position')
    ax3.set_ylabel('Velocity')
    ax3.legend()
    
    plt.show()
 

def CoupledOscillators(x_1,x_2,m,k,k2,t_,dt): ##CoupledOscillators(-10,5,1,10,5,50,0.1)
    t  = np.arange(0,t_,dt)
    x1 = np.zeros(len(t))
    x2 = np.zeros(len(t))
    v1 = np.zeros(len(t))
    v2 = np.zeros(len(t))
    KE1 = np.zeros(len(t))
    PE1 = np.zeros(len(t))
    KE2 = np.zeros(len(t))
    PE2 = np.zeros(len(t))
    Total_E = np.zeros(len(t))
   

    x1[0] = x_1
    x2[0] = x_2
    #v1[0] = v_1
    #v2[0] = v_2

    
    for i in range(0, len(t)-1):
        a1 = -((k+k2)*x1[i]+k2*x2[i])/m
        a2 = -((k2+k)*x2[i]+k2*x1[i])/m

        v1[i+1] = v1[i] + a1*dt
        v2[i+1] = v2[i] + a2*dt

        x1[i+1] = x1[i] + v1[i+1]*dt
        x2[i+1] = x2[i] + v2[i+1]*dt

      
        KE1[i] = (1/2)*m*v1[i]**2
        KE2[i] = (1/2)*m*v2[i]**2
        PE1[i] = (1/2)*k*x1[i]**2
        PE2[i] = (1/2)*k2*x2[i]**2

        Total_E[i] = ((k*x1[i]**2) + k2*(x2[i]-x1[i])**2 + (k*x2[i]**2) + (m*v1[i]**2) + (m*v2[i]**2))/2


    fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5)

   
    camera = Camera(fig)
    for i in range(len(x1)):
        ax1.plot(t,x1, color = 'Blue')
        ax1.plot(t[i],x1[i],'o', color = "Green")
        ax2.plot(t,x2, color = 'Blue')
        ax2.plot(t[i],x2[i],'o', color = "Green")
        camera.snap()
    animation = camera.animate()
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position")

    ax3.plot(t,KE1,'-', label = 'KE for m1')
    ax3.plot(t,PE1,'-', label = 'PE for m1')
    ax3.legend()

    ax4.plot(t,KE2,'-', label = 'KE for m2')
    ax4.plot(t,PE2,'-', label = 'PE for m2')
    ax4.legend()

    ax5.plot(t,Total_E, label = 'Total Energy')
    ax5.legend()
    
    plt.show()

    
def ThreeMassOscillator(x_1,x_2,x_3,k14,k23,m,t_,dt):  ##ThreeMassOscillator(10,3,-15,10,5,1,50,0.1)
    t  = np.arange(0,t_,dt)
    x1 = np.zeros(len(t))
    x2 = np.zeros(len(t))
    x3 = np.zeros(len(t))
    v1 = np.zeros(len(t))
    v2 = np.zeros(len(t))
    v3 = np.zeros(len(t))

    KE1 = np.zeros(len(t))
    PE1 = np.zeros(len(t))
    KE2 = np.zeros(len(t))
    PE2 = np.zeros(len(t))
    KE3 = np.zeros(len(t))
    PE3 = np.zeros(len(t))

    Total_E = np.zeros(len(t))
    

    x1[0] = x_1
    x2[0] = x_2
    x3[0] = x_3
    #v1[0] = v_1
    #v2[0] = v_2
    #v3[0] = v_3

    for i in range(0, len(t)-1):
        a1 = (-k14*x1[i])+(k23*(x2[i]-x1[i]))/m
        a2 = ((-k23*(x2[i]-x1[i]))+(k23*(x3[i]-x2[i])))/m
        a3 = ((-k23*(x3[i]-x2[i]))+(-k14*x3[i]))/m

        v1[i+1] = v1[i] + a1*dt
        v2[i+1] = v2[i] + a2*dt
        v3[i+1] = v3[i] + a3*dt

        x1[i+1] = x1[i] + v1[i+1]*dt
        x2[i+1] = x2[i] + v2[i+1]*dt
        x3[i+1] = x3[i] + v3[i+1]*dt

        KE1[i] = (1/2)*m*v1[i]**2
        KE2[i] = (1/2)*m*v2[i]**2
        KE3[i] = (1/2)*m*v3[i]**2

        PE1[i] = (1/2)*k14*x1[i]**2
        PE2[i] = (1/2)*k23*x2[i]**2
        PE3[i] = (1/2)*k14*x3[i]**2
                       
        Total_E[i] = (((m*v1[i]**2) + (m*v2[i]**2) + (m*v3[i]**2))/2)+(((k14*x1[i]**2)+(k23*(x2[i]-x1[i])**2)+(k23*(x2[i]-x3[i])**2)+(k14*x3[i]**2))/2)

    fig, (ax1, ax2, ax3, ax4, ax5, ax6, ax7) = plt.subplots(7)
    ax1.set_title('Three Mass Oscillator')


    camera = Camera(fig)
    for i in range(len(x1)):
        ax1.plot(t,x1, color = 'Blue')
        ax1.plot(t[i],x1[i],'o', color = "Green")
        ax2.plot(t,x2, color = 'Blue')
        ax2.plot(t[i],x2[i],'o', color = "Green")
        ax3.plot(t,x3, color = 'Blue')
        ax3.plot(t[i],x3[i],'o', color = "Green")
        camera.snap()
    animation = camera.animate()
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Position")
    
    ax4.plot(t,KE1,'-' , label ='KE for m1')
    ax4.plot(t,PE1,'-', label = 'PE for m1')
    ax4.legend()

    ax5.plot(t,KE2,'-', label = 'KE for m2')
    ax5.plot(t,PE2,'-', label = 'PE for m2')
    ax5.legend()

    ax6.plot(t,KE3,'-' , label ='KE for m3')
    ax6.plot(t,PE3,'-', label = 'PE for m3')
    ax6.legend()

    
    ax7.plot(t,Total_E, label = 'Total Energy')
    ax7.legend()
    
    
    plt.show()

#################################################### Heat Equation 
def heatEQ(uo,uL,L,k,h,t,T0): 
     L = L+1
     r = k/h**2
     time = np.arange(0,t,k)
     x = np.arange(0,L,h)
     u = np.ones(len(x))*T0
     u[0] = uo
     u[-1] = uL
     u_ = []
     u_.append(np.copy(u))
     c = np.copy(u)
     for j in range(len(time)):
          for i in range(len(u)-2):
               c[i+1] = (1-2*r)*u[i] + r*u[i+1] + r*u[i+2]
          u_.append(np.copy(c))
          u = np.copy(c)
     fig = plt.figure()
     camera = Camera(fig)
     for i in range(len(u_)):
         plt.plot(u_[i], color = "blue")
         camera.snap()
     animation = camera.animate()
     plt.xlabel("Position")
     plt.ylabel("Temperature")
     plt.show()


def heatEQ2D(uo, uL, L, h, k, T0x, T0y, t):
     L = L +1
     r = k/h**2
     time = np.arange(0,t,k)
     x = np.arange(0,L,h)
     y = np.arange(0,L,h)
     gx = np.ones(len(x))*T0x
     gy = np.ones(len(y))*T0y
     ux = np.copy(gx)
     uy = np.copy(gy)
     ux[0] = uo
     uy[0] = uo
     ux[-1] = uL
     uy[-1] = uL
     ux_ = []
     uy_ = []
     ux_.append(np.copy(ux))
     uy_.append(np.copy(uy))
     c = np.ones((len(x),len(x)))
     c[0,:] = np.copy(ux)
     c[:,0] = np.copy(uy)
     c[-1,-1] = uo
     u_ = []
     
     for j in range(len(time)):
          for i in range(len(ux)-2):
               for k in range(len(uy)-2):
                    c[i+1,k+1] = ((1-2*r)*ux[i] + r*ux[i+1] + r*ux[i+2])+\
                                 ((1-2*r)*uy[k] + r*uy[k+1] + r*uy[k+2])
          u_.append(np.copy(c))
          ux[1:-2] = np.copy(c[1,1:-2])
          uy[1:-2] = np.copy(c[1:-2,1])
          c[0,:] == ux
          c[:,0] == uy
          
     fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
     x,y = np.meshgrid(x,y)

     camera = Camera(fig)
     mycmap = plt.get_cmap('gist_earth')
     for i in range(len(u_)):
          z = u_[i]

          ax.plot_surface(x,y,z, cmap=mycmap,linewidth=0, antialiased=False)
          camera.snap()
     anim = camera.animate(blit=False, interval=10)


     plt.show()

