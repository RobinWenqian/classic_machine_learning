import numpy as np
#sigmoid function
def nonlin (x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#input dataset
X = np.array([[0,0,1],[0,1,1],[1,0,1],[1,1,1]])

#output dataset
y=np.array([[0,0,1,1]]).T

#seed random numbers
np.random.seed(1)

#initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1))-1 #need be saved

for iter in range(10000):
    #forward propagation
    L0 = X
    L1 = nonlin(np.dot(L0,syn0))
    #miss configuration
    L1_error = y - L1
    #slope of the sigmoid at values in l1
    L1_delta = L1_error * nonlin(L1,True)
    #update weights
    syn0 += np.dot(L0.T,L1_delta)

print ("output after training")
print (L1)