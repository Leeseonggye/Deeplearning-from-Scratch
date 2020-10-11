#!/usr/bin/env python
# coding: utf-8

# In[6]:


import matplotlib.pyplot as plt
import numpy as np

x=np.arange(-11,11,0.1)
y=1/(1+np.exp(-x))

plt.plot(x,y)
plt.show()


# In[8]:


def relu(x):
    return np.maximum(0,x)


# In[10]:


relu(-1)


# In[11]:


x=np.arange(-11,10,0.1)
y=relu(x)
plt.plot(x,y)

