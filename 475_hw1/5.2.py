#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 20:26:28 2018

@author: mac
"""

# import statements
datapath = 'datasets/'
from autograd import numpy as np
# the import statement for matplotlib
import matplotlib.pyplot as plt
# import automatic differentiator to compute gradient module
from autograd import grad 
# load in dataset
csvname = datapath + 'kleibers_law_data.csv'
data = np.loadtxt(csvname,delimiter=',')

# get input and output of dataset
x = data[:-1,:]
y = data[-1:,:] 

# gradient descent function 
# g = variables(NDArray or list of NDArray) – Input variables to compute gradients for.
# alpha, 学习速率，自定
# w = theta1，待求的权值
# max_its，最大迭代次数，自定
def gradient_descent(g,alpha,max_its,w):
    # compute gradient module using autograd
    gradient = grad(g)

    # run the gradient descent loop
    weight_history = [w] # weight history container
    cost_history = [g(w)] # cost function history container
    for k in range(max_its):
        # evaluate the gradient
        grad_eval = gradient(w)

        # take gradient descent step
        w = w - alpha*grad_eval
        
        # record weight and cost
        weight_history.append(w)
        cost_history.append(g(w))
    return weight_history,cost_history
    #找到cost_history里最小的值，按照数组下标找到weight_history里相应的w，即为需要的更新参数theta1
    #h(x) = theta1*x
    


# cost function history plotter
def plot_cost_histories(cost_histories,labels):
    # create figure
    plt.figure()
    
    # loop over cost histories and plot each one
    for j in range(len(cost_histories)):
        history = cost_histories[j]
        label = labels[j]
        plt.plot(history,label = label)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.show() 
    


# least squares cost function for linear regression
def least_squares(w):
    cost = 0
    for p in range(len(y)):
        # get pth input/output pair
        x_p = x[p]
        y_p = y[p]
        
        # form linear combination
        c_p = w[0] + w[1]*x_p
        
        # add least squares for this datapoint
        cost += (c_p - y_p)**2
        
    return cost

def test():
    w = np.random.rand(2,1)
    print(w)
    #np.asarray([1.5,1.5])
    weight, cost = gradient_descent(g = least_squares,alpha = 0.05,max_its = 100,w = w)
    print(weight)
    print(cost)

test()


'''
ten_kg_animal_metrate = w[0] + w[1]*np.log(10)
print ('a 10kg animal requires ' + str(np.exp(ten_kg_animal_metrate[0]* 4.18)) + ' calories')

# plot data with linear fit - this is optional
s = np.linspace(min(x)[0],max(x)[0])
t = w[0] + w[1]*s
plt.plot(s,t,linewidth = 3,color = 'r')
plt.scatter(x,y,linewidth = 1)
plt.xlabel('log of mass (in kgs)')
plt.ylabel('log of metabolic rate (in Js)')
plt.show()
'''