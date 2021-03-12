###Alex Andronikides
###Phys 661 
import sympy as sp

################# Part 1################

def FPI(f,x0,Itter):
    itr = 0
    while itr < Itter:
        x = f(x0)
        print('Iteration %d: x = %0.3f, f(x) = %0.3f' % (itr, x, f(x)))
        if f(x) == 0.000:
            print('\n Required root is: %0.3f' % x)
            break
        x0 = x
        itr = itr + 1

    

################ Part 2 ################


def NRM(f,x0,Itter):
    x = sp.Symbol('x')
    f = eval(f)
    df = sp.diff(f,x) 
    itr = 0
    while itr < Itter:
        x_ = x0 - (f.subs(x,x0).evalf())/(df.subs(x,x0).evalf())
        x0 = x_
        print('Iteration %d: x = %0.3f, f(x) = %0.3f' % (itr, x_, f.subs(x, x_)))
        if f.subs(x,x_) == 0.000:
            print('\n Required root is: %0.3f' % x_)
            break
        itr = itr + 1


############### Part 3 ##################

def SCM(f,x0,x1,itter, tol = 10**(-5)): 
    itr = 0
    while itr < itter:
        if f(x0).evalf() - f(x1).evalf() == 0:
            print("\n Divide by Zero Encountered")
            print('\n Closest required root is: %0.3f' % x2)
            break
        x2 = x1 - ((f(x1).evalf()*(x1-x0))/(f(x1).evalf()-f(x0).evalf()))
        print('Iteration %d: x2 = %0.3f, f(x2) = %0.3f' % (itr, x2, f(x2)))
        if f(x2) == tol:
            print('\n Required root is: %0.3f' % x2)
            break
        x0 = x1
        x1 = x2
        itr = itr + 1


    
    
