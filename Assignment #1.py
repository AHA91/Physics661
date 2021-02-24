#!/usr/bin/env python
# coding: utf-8

# In[45]:


### Alex Andronikides
### Phys 661


import numpy as np


######################## Part 1 ######################

v = [1,2,3,4,5,6]         # Vector

for i in v: 
    S = np.square(v)    # Squares elements in v
    L = np.sum(S)       # Sums elements in v
    M= np.sqrt(L)       # Square roots Sum 

print("Magnitue |v| =", M)

####################### Part 2 #########################

a = np.array([2,5,4,3,7])           # Vector a
b = np.array([2,4,5,6,7])          # Vector b
M = []
for i in range(len(a)):              #  i will loop through as many elements as A has                                              
    M.append (a[i]*b[i])
    
S = np.sum(M)  
        
print('Scalar Product:', S)
    
####################### Part 3 #########################
  
X = np.array([[1],[2],[3],[4],[5]])                                # nx1 vector
A = np.array([[1,2,3,4,5],[3,5,4,7,5], [7,7,8,3,4]])      # mxn matrix
y = []                                                                           # empty list
B = 0

for i in range(len(A)):             #  i will loop through as many elements as A has 
    for j in range(len(A[1])):    #   j will loop through as many elements that are in a row in  A
        B += A[i,j]*X[j]              #  Multiplies the element A[i,j] with element X[j] and adds them 
    y.append(B)                         #  Appends B to list y
    B = 0                                   #  Sets B back to 0 and loops again
Y = np.array([y]).T                 #  Converts list y into an array and is then transposed (.T) into a column vector

print('y =',Y)

###################### Part 4 ###########################

u = np.array([[1,2,3],[1,2,3],[1,2,3]])       # Vector  u
v = np.array([[1,2,3],[1,2,3],[1,2,3]])       # Vector v


G = len(u)                                                 
s = []                                                         # s is an emtpy list
for i in range(G):                                     # i runs through the range of  G
    if i == 0:
        j,k = 1,2                                            # elements of vectors
        s.append(u[j]*v[k] - u[k]*v[j])        # Multipling and subracting elements of interest and appends to empty list
    elif i == 1:
        j,k = 2,0
        s.append(u[j]*v[k] - u[k]*v[j])
    else:
        j,k = 0,1
        s.append(u[j]*v[k] - u[k]*v[j])
print(' Cross Product = ', s)

