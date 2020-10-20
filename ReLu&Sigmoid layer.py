#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np

class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y                
        out = x * y

        return out

    def backward(self, dout):
        dx = dout * self.y  # x와 y를 바꾼다.
        dy = dout * self.x

        return dx, dy


class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        out = x + y

        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1

        return dx, dy

    


# In[6]:


# ReLu layer 구현

class ReLu:
    def __init__(self):
        self.mask = None
    
    def forward(self,x):
        self.mask = (x<=0)  
        out = x.copy()
        out[self.mask] = 0
        
        return out
    
    def Backward(self,dout):
        dout[self.mask] = 0
        dx = dout
        
        return dx
# mask -> True/False 로 이루어진 numpy 배열        
        
x = np.array([[1,-0.5],[2,-1]])
mask = (x<=0)
print(mask)


# In[8]:


# sigmoid 구현

class Sigmoid:
    def __init__(self):
        self.out = None
    
    def forward(self,x):
        out = 1/(1+np.exp(-x))
        self.out = out
        
        return out
    
    def Backward(self,dout):
        
        dx = dout * (1-self.out)*self.out
        
        return dx


# In[ ]:




