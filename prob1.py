#!/usr/bin/python

#############################################################
# Gaussian process regression #
# Sk. Mashfiqur Rahman #
#############################################################

import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import RepeatedKFold


def kernel_functions(xi,xj,sigma,op):
    kernel = (op == 0)*(np.exp(-(xi - xj)**2/(2. * sigma ** 2))) + \
             (op == 1)*(np.exp(-np.fabs(xi - xj) / sigma))
    return kernel


training = np.loadtxt("crash.txt")
x = np.array(training[:, 0])
N = len(x)
t = np.array(training[:, 1])
x = x / max(x)
t = t / max(t)
sd = np.std(t) / max(t)
beta = 1. / sd ** 2

x_star = np.linspace(x.min(), x.max(), N)
k1 = np.zeros(shape=(N, N),dtype='double')
C = np.zeros(shape=(N, N),dtype='double')
Kc = np.zeros(shape=(N, N),dtype='double')

#y_star = np.zeros(N)
#count = 0
#for sigma in np.logspace(-3.,-1.2, 100):
#    for p in range(0,N):
#        k1[p,:] = kernel_functions(x[p], x_star[:], sigma, 0)
#    for l in range(0, N):
#       Kc[l, :] = kernel_functions(x[l], x[:], sigma, 0)
#        C = np.add(Kc,np.multiply((1./beta),np.identity(N)))
#    y_star = k1.T.dot(np.linalg.inv(C)).dot(t)
#    plt.figure(figsize=(16,12))
#    plt.plot(x, t, c='r', label="training data")
#    plt.plot(x_star, y_star, c='b', label="test kernel")
#    plt.legend(fontsize=22)
#    plt.xlabel("X",fontsize=22)
#    plt.ylabel("Y",fontsize=22)
#    plt.suptitle("GP sigma test",fontsize=24)
#    count += 1
#plt.savefig('best range for sigma.png',dpi=100)

kfo = 1   # kernel function options, 0: squared exponential, 1: exponential

if kfo == 0: # squared exponential
    kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
    avg_MSE = np.zeros(100)
    count = 0
    opt_MSE = 1000.
    opt_sigma = 0.
    for sigma in np.logspace(-3.,-1.2, 100):
        MSE = np.zeros(5)
        for f in range(5):
            for train_index, test_index in kf.split(x):
                train_x, test_x = x[train_index], x[test_index]
                train_y, test_y = t[train_index], t[test_index]
            validation_y = np.zeros(t.shape)
            Nc = len(train_x)
            Nc1 = len(test_x)
            kn = np.zeros(shape=(Nc, Nc1),dtype='double')
            Cn = np.zeros(shape=(Nc, Nc),dtype='double')
            Kcn = np.zeros(shape=(Nc, Nc),dtype='double')
            for p in range(0,Nc):
                kn[p,:] = kernel_functions(train_x[p], test_x[:], sigma, 0)
            for l in range(0,Nc):
                Kcn[l, :] = kernel_functions(train_x[l], train_x[:], sigma, 0)
                Cn = np.add(Kcn,np.multiply((1./beta),np.identity(Nc)))
            y_hat = kn.T.dot(np.linalg.inv(Cn)).dot(train_y)  # y for validation
            MSE[f] = (0.5 * np.linalg.norm(y_hat - test_y) ** 2.) / Nc
        avg_MSE[count] = np.average(MSE)
        if not np.isnan(np.average(MSE)) and np.average(MSE) < opt_MSE:
            opt_sigma = sigma
        count += 1
    print("optimum sigma={}".format(opt_sigma))

    x_star = np.linspace(x.min(), x.max(), N)
    y_star = np.zeros(N)
    for p in range(0,N):
        k1[p,:] = kernel_functions(x[p], x_star[:], opt_sigma, 0)
    for l in range(0, N):
        Kc[l, :] = kernel_functions(x[l], x[:], opt_sigma, 0)
        C = np.add(Kc,np.multiply((1./beta),np.identity(N)))
    y_star = k1.T.dot(np.linalg.inv(C)).dot(t)

    plt.figure(figsize=(16,12))
    plt.plot(x, t, c='r', label="training data")
    plt.plot(x_star, y_star, c='b', label="squared exponential kernel")
    plt.legend(fontsize=22)
    plt.xlabel("X",fontsize=22)
    plt.ylabel("Y",fontsize=22)
    plt.title("Gaussian process regression ($\sigma$={:.6f})".format(opt_sigma), fontsize=24)
    plt.savefig("squared_exponential_kernel.png", format='png')
    plt.show()

if kfo == 1:  # exponential
    kf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=None)
    avg_MSE = np.zeros(100)
    count = 0
    opt_MSE = 1000.
    opt_sigma = 0.
    for sigma in np.logspace(-3.,-0.8, 100):
        MSE = np.zeros(5)
        for f in range(5):
            for train_index, test_index in kf.split(x):
                train_x, test_x = x[train_index], x[test_index]
                train_y, test_y = t[train_index], t[test_index]
            validation_y = np.zeros(t.shape)
            Nc = len(train_x)
            Nc1 = len(test_x)
            kn = np.zeros(shape=(Nc, Nc1),dtype='double')
            Cn = np.zeros(shape=(Nc, Nc),dtype='double')
            Kcn = np.zeros(shape=(Nc, Nc),dtype='double')
            for p in range(0,Nc):
                kn[p,:] = kernel_functions(train_x[p], test_x[:], sigma, 1)
            for l in range(0,Nc):
                Kcn[l, :] = kernel_functions(train_x[l], train_x[:], sigma, 1)
                Cn = np.add(Kcn,np.multiply((1./beta),np.identity(Nc)))
            y_hat = kn.T.dot(np.linalg.inv(Cn)).dot(train_y)  # y for validation
            MSE[f] = (0.5 * np.linalg.norm(y_hat - test_y) ** 2.) / Nc
        avg_MSE[count] = np.average(MSE)
        if not np.isnan(np.average(MSE)) and np.average(MSE) < opt_MSE:
            opt_sigma = sigma
        count += 1
    print("optimum sigma={}".format(opt_sigma))

    x_star = np.linspace(x.min(), x.max(), N)
    y_star = np.zeros(N)
    for p in range(0,N):
        k1[p,:] = kernel_functions(x[p], x_star[:], opt_sigma, 1)
    for l in range(0, N):
        Kc[l, :] = kernel_functions(x[l], x[:], opt_sigma, 1)
        C = np.add(Kc,np.multiply((1./beta),np.identity(N)))
    y_star = k1.T.dot(np.linalg.inv(C)).dot(t)

    plt.figure(figsize=(16,12))
    plt.plot(x, t, c='r', label="training data")
    plt.plot(x_star, y_star, c='b', label="exponential kernel")
    plt.legend(fontsize=22)
    plt.xlabel("X",fontsize=22)
    plt.ylabel("Y",fontsize=22)
    plt.title("Gaussian process regression ($\sigma$={:.6f})".format(opt_sigma), fontsize=24)
    plt.savefig("exponential_kernel.png", format='png')
    plt.show()
