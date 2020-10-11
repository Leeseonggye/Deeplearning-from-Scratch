#!/usr/bin/env python
# coding: utf-8

# In[1]:


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


# In[4]:


# 신경망 구현

import numpy as np

x= np.array([1,2])
w= np.array([[1,3,5],[2,4,6]])
Y= np.dot(x,w)
print(Y)


# In[9]:


# 3 layer network 

def sigmoid(x):
    return 1/(1+np.exp(-x))

def identify_function(x):
    return x
    
def init_network():
    network ={}
    network['W1'] =np.array([[0.1,0.3,0.5],[0.2,0.4,0.6]])
    network['b1'] =np.array([0.1,0.2,0.3])
    network['W2'] =np.array([[0.1,0.4],[0.2,0.5],[0.3,0.6]])
    network['b2'] =np.array([0.1,0.2])
    network['W3'] =np.array([[0.1,0.3],[0.2,0.4]])
    network['b3'] =np.array([0.1,0.2])
    return network

def forward(network,x):
    W1,W2,W3 = network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['b1'], network['b2'], network['b3']
    
    a1 = np.dot(x,W1)+ b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2)+b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3)+b3
    Y = identify_function(a3)
    return Y
    


# In[10]:


network = init_network()
x= np.array([1.0,0.5])
y = forward(network,x)
y

