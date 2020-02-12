import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import itertools
import sklearn.linear_model
data = np.loadtxt("notas_andes.dat", skiprows=1)
ones=np.ones((69,1))
#Introduzco una linea de unos para contar los datos para C Esto hace que los datos queden 
#ordenados como 0 representando a C e i representando a mi
ndata=np.concatenate((ones,data),axis=1)
sig=0.1
def prob(PAR):
    P=np.log(1/(sig*np.sqrt(2*np.pi)))-1/(2*sig**2) *np.sum(np.array([PAR@ndata[i,0:5]-ndata[i,5] for i in range(ndata.shape[0])])**2)
    return P
def metro(PAR):
    PARN=PAR+np.random.normal(loc=0.0,scale=0.1,size=5)
    while np.sum(PARN<0)>0:
        PARN=PAR+np.random.normal(loc=0.0,scale=0.1,size=5)       
    A=np.min([1,np.exp(prob(PARN)-prob(PAR))])
    #print(prob(PARN) , prob(PAR))
    r=np.random.random()
    if r<A:
        print(PARN)
        PAR=PARN
        Prob=prob(PARN)
    else:
        Prob=prob(PAR)
    return PAR,Prob
N=100000
PAR=np.random.normal(5)
#PAR=np.array([1,0.5,0.5,0.5,0.5])
Prob=np.zeros(N)
PARL=np.zeros((N,5))
for i in range(N):
    [PARL[i,:],Prob[i]]=metro(PAR)   
    PAR=PARL[i,:]