import numpy as np
import matplotlib.pyplot as plt
import sklearn.model_selection
import sklearn.linear_model
data = np.loadtxt("notas_andes.dat", skiprows=1)
ones=np.ones((69,1))
#Introduzco una linea de unos para contar los datos para C Esto hace que los datos queden 
#ordenados como 0 representando a C e i representando a mi
ndata=np.concatenate((ones,data),axis=1)
XMAT=np.array([[np.sum(ndata[:,i]*ndata[:,j]) for j in range(5)]  for i in range(5) ])
YMAT=np.array([np.sum(ndata[:,5]*ndata[:,j] ) for j in range(5)])
PAR=YMAT@np.linalg.pinv(XMAT)
YPRED=ndata[:,0:5]@PAR
INCM=2*np.linalg.inv(2/(0.1**2) *XMAT)
S2= 1/(68) *np.sum([(PAR@ndata[i,0:5]-ndata[i,5])**2 for i in range(69)])
output=r'C=%4.3f'%PAR[0]+'  m_1=%4.3f'%PAR[1]+'  m_2=%4.3f'%PAR[2]+'  m_3=%4.3f'%PAR[3]+'  m_4=%4.3f'%PAR[4]+'  \n S^2=%4.3f'%S2
print(output)
print("Matriz de covarianza")
print(np.round(INCM,4))