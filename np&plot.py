#!/usr/bin/env python
# coding: utf-8

# In[2]:


type(2.34)


# In[7]:


import numpy as np
x= np.array([1.00,2.00,3.00])
print(x)


# In[9]:


type(x)


# In[12]:


y=np.array([2,3,4])
x+y
type(x+y)


# In[27]:


a=np.array([[1,2],[3,4]])
type(a)
b=np.array([[3,4],[5,6]])
a*b
c=np.array([1,2])
a*c


# In[35]:


k=np.array([11,13,15,17,21,22])
k>15
k[k>15]
k[k<15]
k[k%2==0]


# In[41]:


import matplotlib.pyplot as plt

x=np.arange(0,11,0.1)
y=np.exp(x)

plt.plot(x,y)
plt.show()


# In[52]:



x=np.arange(0,11,0.1)
y1=np.sin(x)
y2=np.cos(x)
plt.plot(x,y1,label ="sin")
plt.plot(x,y2,linestyle = "--", label ="cos")
plt.legend()
plt.show()

