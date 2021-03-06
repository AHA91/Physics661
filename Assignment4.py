import numpy as np
import sympy as sp

################# Part 1################

def FPI(f,x0,Itter):
    itr = 0

    while itr < Itter:
        x = f(x0)
        print('Iteration %d: x = %0.3f, f(x) = %0.3f' % (itr, x, f(x)))
        x0 = x
        itr = itr + 1
    print('\n Required root is: %0.3f' % x)
    

################ Part 2 ################
# Printing Iteration issue

def NRM(f,x0,Itter):
    x = sp.Symbol('x')
    f = eval(f)
    df = sp.diff(f,x) 
    itr = 0

    while itr < Itter:
        x_ = x0 - (f.subs(x,x0).evalf())/(df.subs(x,x0).evalf())
        print('Iteration %d: x_ = %0.3f, f(x_) = %0.3f' % (itr, x_, f(x_)))
        x0 = x_ 
        itr = itr + 1
    print('\n Required root is: %0.3f' % x_)


############### Part 3 ##################

def SCM(f,x0,x1,Itter):
    itr = 0

    while itr < Itter:
        if f(x0) == f(x1):
            print("Error: Divide by zero")
            break               
        x2 = x0 - f(x0)*(x1-x0)/(f(x1)-f(x0))
        print('Iteration %d: x2 = %0.3f, f(x2) = %0.3f' % (itr, x2, f(x2)))
        x0 = x1
        x1 = x2                        
        itr = itr + 1
    print('\n Required root is: %0.3f' % x2)
        
       
    
    
