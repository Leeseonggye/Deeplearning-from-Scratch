#!/usr/bin/env python
# coding: utf-8

# In[2]:


#SGD

class SGD:
    def __init__(self,lr=0.01):
        self.lr =lr
    def update(self,params,grad):
        for key in params.key():
            params[key] -= self.lr *grads[key]


# In[4]:


#momentum

class Momentum:
    def __init__(self,lr=0.01,momentum=0.9):
        self.lr =lr
        self.momentum = momentum
        self.v = None
    def update(self,parmas,grad):
        if self.v is None:
            self.v = {}
            for key, val in params.items():
                self.v[key] = np.zeros_like(val)
            for key in params.keys():
                self.v[key] = self.momentum * self.v[key] - self.lr * grads[key]
                params[key] += self.v[key]

