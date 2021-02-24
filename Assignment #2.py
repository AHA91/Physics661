#!/usr/bin/env python
# coding: utf-8

# In[1]:


import random
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
from itertools import product

################# One Fair Dice################
result = []
N = 1000

for i in range (N):
    D1 = random.randint(1,6)
    result.append(D1)
    h = np.array(result)
    z = Counter(h)       
    T = np.array(list(z.items()))

t = [list(range(1,7)) for x in range(1)]
g = list(map(sum,list(product(*t))))
z = Counter(g)
h_ = np.array(list(z.items()))
total = sum(h_[:,1])
h_[:,1] = (h_[:,1]/total)*N

x_ = h_[:,0]
y_ = h_[:,1]  

    
x = T[:,0]
y = T[:,1]

plt.plot(x_,y_,'o', color = "Red")
plt.bar(x,y)


plt.xlabel('Sum of roll')
plt.xticks(np.arange(1, 7, 1)) 
plt.ylabel('Frequency')
plt.title("Sum of side vs Frequency")

plt.show



################# Two Fair Dice################
result = []
N = 1000

for i in range (N):
    D1 = random.randint(1,6)
    D2 = random.randint(1,6)
    DT = D1+D2
    result.append(DT)
    h = np.array(result)
    z = Counter(h)       
    T = np.array(list(z.items()))

t = [list(range(1,7)) for x in range(2)]
g = list(map(sum,list(product(*t))))
z = Counter(g)
h_ = np.array(list(z.items()))
total = sum(h_[:,1])
h_[:,1] = (h_[:,1]/total)*N

x_ = h_[:,0]
y_ = h_[:,1]


x = T[:,0]
y = T[:,1]

plt.plot(x_,y_,'o', color = 'Red')
plt.bar(x,y)


plt.xlabel('Sum of roll')
plt.xticks(np.arange(2, 13, 1)) 
plt.ylabel('Frequency')
plt.title("Sum of side vs Frequency")


plt.show


################# Unfair Dice (1) ################
result = []
D1_ = []
D2_=[]
N = 1000

for i in range (N):
    D1 = random.randint(1,6)
    D2 = random.randint(1,6)    
    
    D1_.append(D1)
    D2_.append(D2)
    d1 = np.array(D1)
    d2 = np.array(D2)
    
    d1_= np.where(d1>4, d1, 1)  #D1 unfair
    
    DT = d1_+ d2
    result.append(DT)
    h = np.array(result)
    z = Counter(h)       
    T = np.array(list(z.items()))

t = [list(range(1,7)) for x in range(2)]
g = list(map(sum,list(product(*t))))
z = Counter(g)
h_ = np.array(list(z.items()))
total = sum(h_[:,1])
h_[:,1] = (h_[:,1]/total)*N

x_ = h_[:,0]
y_ = h_[:,1]  

    
    
x = T[:,0]
y = (T[:,1])

plt.plot(x_,y_,'o', color = "Red")
plt.bar(x,y)

plt.xlabel('Sum of roll')
plt.xticks(np.arange(2, 13, 1)) 
plt.ylabel('Frequency')
plt.title("Unfair Dice (1)\nSum of side vs Frequency")

plt.show


################# Unfair Dice (2) ################
result = []
D1_ = []
D2_=[]
N = 1000

for i in range (N):
    D1 = random.randint(1,6)
    D2 = random.randint(1,6)    
    
    D1_.append(D1)
    D2_.append(D2)
    d1 = np.array(D1)
    d2 = np.array(D2)
    
    d1_= np.where(d1>4, d1, 1)  #D1 unfair
    d2_= np.where(d2>3, d2, 1)  #D2 unfair
    
    DT = d1_+d2_
    result.append(DT)
    h = np.array(result)
    z = Counter(h)       
    T = np.array(list(z.items()))

    
t = [list(range(1,7)) for x in range(2)]
g = list(map(sum,list(product(*t))))
z = Counter(g)
h_ = np.array(list(z.items()))
total = sum(h_[:,1])
h_[:,1] = (h_[:,1]/total)*N

x_ = h_[:,0]
y_ = h_[:,1]  

x = T[:,0]
y = T[:,1]

plt.plot(x_,y_,'o', color = "Red")
plt.bar(x,y)

plt.xlabel('Sum of roll')
plt.xticks(np.arange(2, 13, 1)) 
plt.ylabel('Frequency')
plt.title("Unfair Dice (2)\nSum of side vs Frequency")

plt.show

