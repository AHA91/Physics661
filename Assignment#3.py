####### Alex Andronikides
####### Physics 661
####### Assignment #3

import numpy as np


####### Part 1
def bisect01(f,a,b,tol):
    a = int(a)
    b = int(b)
    c = (a+b)/2

    if f(c)==0:
        print('c is root:',c)
    else:
        while abs(f(c))>tol:
            if f(c)*f(a) > 0:
                a = c
                b = b
            else:
                a = a
                b = c
            c = (a+b)/2

        return(c)
    
####### Part 2
def bisect02(f,a,b,n = 6, tol = 10**(-10)):
    X = bisect01(f,a,b, tol)
    return round(X, n)
    


####### Part 3
def Optol(a,b,tol = 10**(-10)):
    X = eval('lambda x:' + input('Function:'))
    h = input("Tolerance or deff to use default: ")
    dec = input("Enter decimal places or default to use default: ")
    if dec == "default":
        n = 6
    else:
        n = int(dec)
    if h == 'deff':
        tol = 10**(-10)
    else:
        tol = float(eval(h))
    Y = bisect02(X,a,b,n,tol)
    return Y
        
