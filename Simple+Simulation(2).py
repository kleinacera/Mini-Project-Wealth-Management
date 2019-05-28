
# coding: utf-8

# In[27]:


# -*- coding: utf-8 -*-
"""
Created on Thu Sep  7 20:03:58 2018

@author: Yungchi
"""

#%%
import numpy as np
# Define the correlation matrix
# Stocks, bonds and cash
CorrMat=np.array([[1,0.4,0.1],[0.4,1,0.3],[0.1,0.3,1]])
# Define the vector of standard deviations
SD=np.array([[0.2],[0.07],[0.02]])
# Define expected returns
ExpRet=np.array([[0.07],[0.03],[0.02]])
# Define the remaining life of this person
T_b=30
T_a=30
# Define the number of simulations
NumSim=100000
# Define the asset allocation
Alloc_b = np.array([[0.60],[0.40],[0.00]])
Alloc_a = np.array([[0.30],[0.60],[0.10]])
# Define the balance of the account today
Balance = 0
income_b = 20000
income_a = -30000

# End of definitions
# Calculations start here

# Compute the covariance matrix
OuterProduct=np.dot(SD,SD.T)
CovMat=CorrMat*OuterProduct
nCol=CorrMat.shape[1]
# Generate a matrix of simulated normal standard returns
SimRet=np.random.normal(loc=0.0,scale=1.0,size=(int(NumSim/2)*(T_b+T_a),nCol))
# Variance reduction technique: antithetic variates
TempRet=-SimRet
SimRet=np.vstack([SimRet,TempRet])
# Make data lognormal instead of normal
SimRet = np.exp(SimRet)
# Change variance and covariance of random numbers
SDMat = np.linalg.cholesky(CovMat)
SimRet = SimRet@SDMat
# Change mean of random numbers
ColAvg = np.mean(SimRet,axis=0)
SimRet = SimRet - ColAvg
SimRet = SimRet + ExpRet.T
# Assuming annual rebalancing, multiply random returns by allocation and 
#  obtain a vector of portfolio returns
SimRet_b = SimRet[0:NumSim*T_b]@Alloc_b
SimRet_a = SimRet[NumSim*T_b:NumSim*(T_b+T_a)]@Alloc_a
SimRet=np.vstack([SimRet_b,SimRet_a])
# Reshape the vector into a matrix of T rows
SimRet = SimRet.reshape(T_b+T_a,NumSim)
# Allocate a matrix with results
SimLives = np.zeros([T_b+T_a+1,NumSim])
# The first row is a constant, the value of the portfolio today
SimLives[0,] = Balance
for i in range(1,T_b+1):
    SimLives[i,]=SimLives[i-1,]*(1+SimRet[i-1,])+income_b
for i in range(T_b+1,T_b+T_a+1):
    SimLives[i,]=SimLives[i-1,]*(1+SimRet[i-1,])+income_a
# Extract relevant percentiles
PercentOut = np.percentile(SimLives,[10,25,50,75,90],axis=1)
#  Create picture with log vertical axis
import matplotlib.pyplot as plt
plt.semilogy(PercentOut.T)
plt.title('Simulation of future wealth')
plt.xlabel('Years from today')
plt.ylabel('Portfolio value')
plt.show()

PercentOut[0,]
IRR = np.irr(PercentOut[0,])
IRR

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
a[0:2]
a[2]
b = 1000
print(b)


#%%
#Class 2
import numpy as np
# Define the correlation matrix
# Stocks, bonds and cash
CorrMat=np.array([[1,0.4,0.1],[0.4,1,0.3],[0.1,0.3,1]])
# Define the vector of standard deviations
SD=np.array([[0.2],[0.07],[0.02]])
# Define expected returns
ExpRet=np.array([[0.07],[0.03],[0.02]])
# Define the remaining life of this person
T_b=30
T_a=30
# Define the number of simulations
NumSim=100000
# Define the asset allocation
Alloc_y = np.array([[0.60],[0.40],[0.00]])
Alloc_o = np.array([[0.30],[0.60],[0.10]])
# Define the balance of the account today
Balance = 0
# Define annual contribution
income_b = 20000
# Define the retirement need
income_a = -30000

# End of definitions
# Calculations start here

# Compute the covariance matrix
OuterProduct=np.dot(SD,SD.T)
CovMat=CorrMat*OuterProduct
nCol=CorrMat.shape[1]
# Generate a matrix of simulated normal standard returns
SimRet=np.random.normal(loc=0.0,scale=1.0,size=(int(NumSim/2)*(T_b+T_a),nCol))
# Variance reduction technique: antithetic variates
TempRet=-SimRet
SimRet=np.vstack([SimRet,TempRet])
# Make data lognormal instead of normal
SimRet = np.exp(SimRet)
# Change variance and covariance of random numbers
SDMat = np.linalg.cholesky(CovMat)
SimRet = SimRet@SDMat
# Change mean of random numbers
ColAvg = np.mean(SimRet,axis=0)
SimRet = SimRet - ColAvg
SimRet = SimRet + ExpRet.T
# Assuming annual rebalancing, multiply random returns by allocation and 
#  obtain a vector of portfolio returns
SimRet_b = SimRet[0:NumSim*T_b]@Alloc_b
SimRet_a = SimRet[NumSim*T_b:NumSim*(T_b+T_a)]@Alloc_a
SimRet=np.vstack([SimRet_b,SimRet_a])
# Create a vector of cash flow
Cashflows = np.zeros([T_b+1],1)
# Reshape the vector into a matrix of T rows
SimRet = SimRet.reshape(T_b+T_a,NumSim)
# Allocate a matrix with results
SimLives = np.zeros([T_b+T_a+1,NumSim])
# The first row is a constant, the value of the portfolio today
SimLives[0,] = Balance
for i in range(1,T_b+1):
    SimLives[i,]=SimLives[i-1,]*(1+SimRet[i-1,])+income_b
for i in range(T_b+1,T_b+T_a+1):
    SimLives[i,]=SimLives[i-1,]*(1+SimRet[i-1,])+income_a
# Extract relevant percentiles
PercentOut = np.percentile(SimLives,[10,25,50,75,90],axis=1)
#  Create picture with log vertical axis
import matplotlib.pyplot as plt
plt.semilogy(PercentOut.T)
plt.title('Simulation of future wealth')
plt.xlabel('Years from today')
plt.ylabel('Portfolio value')
plt.show()

PercentOut[0,]
IRR = np.irr(PercentOut[0,])
IRR

a = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
a[0:2]
a[2]
b = 1000
print(b)

# Define a vector of probabilities of death
Mortality = np.zeros([T-Retire,1])
for i in range(0,T-retire):
    Mortality[i]=(i+1)/(Mortality.shape[0])

# Computer average account balance at death
Death = Mortality.T@Simlives[Retire+1:T+1,]
Death = np.mean(Death)



r_min = pbar.min()
r_max = pbar.max()
optimal_mus = np.arange(r_min,r_max,(r_max-r_min)/9)

# convex optimization

