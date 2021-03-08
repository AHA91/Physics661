###Alex Andronikides
###Phys 661 
import sympy as sp

################# Part 1################

def FPI(f,x0,Itter):
    itr = 0
    while itr < Itter:
        x = f(x0)
        print('Iteration %d: x = %0.3f, f(x) = %0.3f' % (itr, x, f(x)))
        x0 = x
        itr = itr + 1
        if f(x) == 0.000:
            print('\n Required root is: %0.3f' % x)
            break
    

################ Part 2 ################


def NRM(f,x0,Itter):
    x = sp.Symbol('x')
    f = eval(f)
    df = sp.diff(f,x) 
    itr = 0
    while itr < Itter:
        x_ = x0 - (f.subs(x,x0).evalf())/(df.subs(x,x0).evalf())
        print('Iteration '+ str(itr) + ": "+'%.3f %.3f' % (x_, f.subs(x_,x)))
        x0 = x_
        itr = itr + 1
        if f.subs(x_,x) == 0.000:
            print('\n Required root is: %0.3f' % x_)
            break


############### Part 3 ##################

def SCM(f,x0,x1,Itter):
    itr = 0
    while itr < Itter:
        x2 = x1 - f(x1)*((x1-x0)/(f(x1)-f(x0)))
        x0 = x1
        x1 = x2
        print('Iteration %d: x2 = %0.3f, f(x2) = %0.3f' % (itr, x2, f(x2)))
        itr = itr + 1
        if f(x2) == 0.000:
            print('\n Required root is: %0.3f' % x2)
            break
         
       
    
    
