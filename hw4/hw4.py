#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.io import loadmat

#Be careful with the file path!
data = loadmat('data/hw4.mat')
from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder(sparse=False)
y_onehot = encoder.fit_transform(data['y'])


# In[2]:


def sigmoid(z):
    return 1 / (1 + np.exp(-z))
    
def forward_propagate(X, theta1, theta2):
    one = [1 for _ in range(len(X))]
    one = np.matrix(one)
    X = np.concatenate((one.T,X), axis=1)
    m = X.shape[0]         

    #Write codes here
    #Ver1
    a1=[]
    z2=[]
    a2=[]
    z3=[]
    h=[]
    y_pred=[]
    
    for i in range(m):
        a1 += [X[i]]
        z2 += [np.matmul(a1[i],theta1.T)]
        tmp_one=np.matrix(1)
        tmp_a2 = sigmoid(z2[i])
        a2 += [np.concatenate((tmp_one,tmp_a2),axis=1)]
        z3 += [np.matmul(a2[i],theta2.T)]
        tmp_z3=sigmoid(z3[i])
        tmp2_z3=tmp_z3[:]
        h  += [sigmoid(z3[i])]
        for j in range(np.shape(tmp_z3)[1]):
            if j == np.argmax(tmp_z3):
                tmp2_z3.itemset(j,1)
            else:
                tmp2_z3.itemset(j,0)
        y_pred  += [tmp2_z3]
        
    sp=np.shape(z3[i])[1]
    h=np.array(h).reshape(m,sp)
    y_pred=np.array(y_pred).reshape(m,sp)
    print(np.shape(y_pred))        
    
    return a1, z2, a2, z3, h, y_pred
    
def cost(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    m = X.shape[0]
    X = np.matrix(X)
    y = np.matrix(y)
    # reshape the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    # run the feed-forward pass
    a1, z2, a2, z3, h, y_pred = forward_propagate(X, theta1, theta2)
    # compute the cost
    J = 0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - h[i,:]))
        J += np.sum(first_term - second_term)
        
    J = J / m
    J += (float(learning_rate) / (2*m) * (np.sum(np.power(theta1[:,1:],2)) + np.sum(np.power(theta2[:,1:],2))))
    
    return J


# In[3]:


# initial setup
input_size = 400
hidden_size = 10
num_labels = 10
learning_rate = 1
# randomly initialize a parameter array of the size of the full network's parameters
params = (np.random.random(size=hidden_size * (input_size + 1) + num_labels * (hidden_size + 1)) - 0.5) * 0.2
m = data['X'].shape[0]

X = np.matrix(data['X'])
y = np.matrix(y_onehot)


# # Back propagation

# In[4]:


def forward_propagate_for_bp(X, theta1, theta2,idx):

    #ver2
    a1 = X[idx]
    z2 = np.matmul(a1,theta1.T)
    tmp_one=np.matrix(1)
    tmp_a2 = sigmoid(z2)
    a2 = np.concatenate((tmp_one,tmp_a2),axis=1)
    z3 = np.matmul(a2,theta2.T)
    h  = sigmoid(z3)
    
    return a1, z2, a2, z3, h


# In[5]:


def sigmoid_gradient(z):
    return np.multiply(sigmoid(z), (1 - sigmoid(z)))    

def backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate):
    # unravel the parameter array into parameter matrices for each layer
    theta1 = np.matrix(np.reshape(params[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
    theta2 = np.matrix(np.reshape(params[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
    
    m = X.shape[0]
    one = [1 for _ in range(m)]
    one = np.matrix(one)
    X = np.concatenate((one.T,X), axis=1)
    sumd1=0
    sumd2=0
    #Write codes here
    record_a1=[]
    record_z2=[]
    record_a2=[]
    record_z3=[]
    record_h=[]
    for i in range(m):
        a1, z2, a2, z3, h = forward_propagate_for_bp(X, theta1, theta2,i)
        
        record_a1 += [a1]
        record_z2 += [z2]
        record_a2 += [a2]
        record_z3 += [z3]
        record_h += [h]
        d3=h-y[i]
        d2=np.multiply(np.matmul(d3,theta2[:,1:]),sigmoid_gradient(z2))
        sumd2 += np.matmul(d3.T,a2[:,1:])
        sumd1 += np.matmul(d2.T,a1[:,1:])
    
    record_h=np.array(record_h).reshape(m,10)

    sumd1 = sumd1 / m
    sumd2 = sumd2 / m
    sumd1[:,:] = sumd1[:,:] + (theta1[:,1:]*learning_rate) / m
    sumd2[:,:] = sumd2[:,:] + (theta2[:,1:]*learning_rate) / m
    bias1 = theta1[:,0:1]/m
    bias2 = theta2[:,0:1]/m
    sumd1 = np.concatenate((bias1,sumd1),axis=1)
    sumd2 = np.concatenate((bias2,sumd2),axis=1)
    grad = np.concatenate((np.ravel(sumd1),np.ravel(sumd2)))

#     print("a1 = ",np.shape(record_a1))
#     print("a2 = ",np.shape(record_a2))
#     print("z2 = ",np.shape(record_z2))
#     print("z3 = ",np.shape(record_z3))
#     print("h = ",np.shape(record_h))
#     print("sumd1 = ",np.shape(sumd1))
#     print("sumd2 = ",np.shape(sumd2))
#     print("theta 1 = ",theta1.shape)
#     print("theta 2 = ",theta2.shape)
    
    J=0
    for i in range(m):
        first_term = np.multiply(-y[i,:], np.log(record_h[i,:]))
        second_term = np.multiply((1 - y[i,:]), np.log(1 - record_h[i,:]))
        J += np.sum(first_term - second_term)
        
    J = J / m
    J += (float(learning_rate) / (2*m) * (np.sum(np.power(theta1[:,1:],2)) + np.sum(np.power(theta2[:,1:],2))))
    print("J = ",J)   
    
    return J, grad
    
backprop(params, input_size, hidden_size, num_labels, X, y, learning_rate)


# In[6]:


from scipy.optimize import minimize
# minimize the objective function
fmin = minimize(fun=backprop, x0=params, args=(input_size, hidden_size, num_labels, X, y_onehot, learning_rate), method='TNC', jac=True, options={'maxiter': 250})

fmin


# In[7]:


#Predict bp result
theta1 = np.matrix(np.reshape(fmin.x[:hidden_size * (input_size + 1)], (hidden_size, (input_size + 1))))
theta2 = np.matrix(np.reshape(fmin.x[hidden_size * (input_size + 1):], (num_labels, (hidden_size + 1))))
a1, z2, a2, z3, h , tmp_pred= forward_propagate(X, theta1, theta2)

cnt=0
for i in range(len(h)):
    if np.argmax(tmp_pred[i])==np.argmax(y[i]):
        cnt+=1

accuracy = cnt / float(len(h))
# print(cnt)
print("accuracy = ",accuracy*100)
print("correct = ",cnt)
print("total = ",len(h))


# In[ ]:




