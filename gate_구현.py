#!/usr/bin/env python
# coding: utf-8

# In[7]:


# perceptron And gate

def AND(x1,x2):
    w1,w2,theta =0.5, 0.5, 0.8
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    else:
        return 1
    
AND(0,1)
AND(1,1)


# In[10]:


#AND GATE with np
import numpy as np

def AND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1


# In[15]:


import numpy as np

def NAND(x1,x2):
    x = np.array([x1,x2])
    w = np.array([-0.5,-0.5])
    b = 0.7
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1


# In[19]:


NAND(1,1)


# In[21]:


def OR(x1,x2):
    x = np.array([x1,x2])
    w = np.array([0.5,0.5])
    b = -0.1
    tmp = np.sum(w*x)+b
    if tmp <= 0:
        return 0
    else:
        return 1


# In[24]:


OR(0,1)


# In[34]:


def XOR(x1,x2):
    x = np.array([x1,x2])
    s1 = NAND(x1,x2)
    s2 = OR(x1,x2)
    y= AND(s1,s2)
    return y


# In[36]:


XOR(1,1)


# In[35]:


XOR(0,1)


# In[37]:


XOR(1,0)


# In[39]:


XOR(0,0)


# In[ ]:




