import numpy as np
import matplotlib.pyplot as plt

def univariate_kalman(y,Z,H,T,Q,R):
	a = np.zeros((T.shape[0],y.shape[0]+1)) # diffuse prior assumed
	a[0][0] = y[0]
	P = np.ones((a.shape[0],a.shape[0],y.shape[0]+1))*(10**7) # diffuse prior asumed

	K = np.zeros(a.shape[0])
	v = np.zeros(y.shape[0])
	F = np.zeros((H.shape[0],H.shape[1],y.shape[0]))

	for t in xrange(0,y.shape[0]):
		v[t] = y[t] - np.dot(Z,a[:,t])

		F[:,:,t] = np.dot(np.dot(Z,P[:,:,t]),Z.T) + H.flatten()[0]

		K = np.dot(np.dot(T,P[:,:,t]),Z.T)/(F[:,:,t]).flatten()[0]

		a[:,t+1] = np.dot(T,a[:,t]) + np.dot(K,v[t]) 

		P[:,:,t+1] = np.dot(np.dot(T,P[:,:,t]),T.T) + np.dot(np.dot(R,Q),R.T) - F[:,:,t].flatten()[0]*np.dot(np.array([K]).T,np.array([K]))

	return a, F, v

def univariate_kalman_fcst(y,Z,H,T,Q,R,h):
	a = np.zeros((T.shape[0],y.shape[0]+1+h)) # diffuse prior assumed
	a[0][0] = y[0]
	P = np.ones((a.shape[0],a.shape[0],y.shape[0]+1+h))*(10**7) # diffuse prior asumed

	K = np.zeros(a.shape[0]+h)
	v = np.zeros(y.shape[0]+h)
	F = np.zeros((H.shape[0],H.shape[1],y.shape[0]+h))

	for t in xrange(0,y.shape[0]+h):
		if t >= y.shape[0]:
			v[t] = 0
			F[:,:,t] = 10**7
			K = np.zeros(a.shape[0])
		else:
			v[t] = y[t] - np.dot(Z,a[:,t])
			F[:,:,t] = np.dot(np.dot(Z,P[:,:,t]),Z.T) + H.flatten()[0]
			K = np.dot(np.dot(T,P[:,:,t]),Z.T)/(F[:,:,t]).flatten()[0]

		a[:,t+1] = np.dot(T,a[:,t]) + np.dot(K,v[t]) 

		P[:,:,t+1] = np.dot(np.dot(T,P[:,:,t]),T.T) + np.dot(np.dot(R,Q),R.T) - F[:,:,t].flatten()[0]*np.dot(np.array([K]).T,np.array([K]))

	return a, P
