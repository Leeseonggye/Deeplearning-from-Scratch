#!/usr/bin/env python
# coding: utf-8

# In[ ]:


class softmaxWithLoss:
    def __init__(self):
        self.loss = None #손실 정의
        self.y = None
        self.t = None
    
    def forward(self,x,t):
        self.t = t
        self.x = softmax(X)
        self.loss = cross_entropy_error(self.y,selt.t)
        return self.loss
    
    def backward(self,dout = 1):
        batch_siz = self.t.shape[0]
        dx = (self.y-self.t)/batch_size
        return dx

