import numpy as np
import matplotlib.pyplot as plt
import sys
if sys.version_info < (3,):
    range = xrange

def univariate_KFS(y,Z,H,T,Q,R,mu):
	""" Kalman filtering and smoothing for univariate time series

	Notes
	----------

	y = mu + Za_t + e_t         where   e_t ~ N(0,H)  MEASUREMENT EQUATION
	a_t = Ta_t-1 + Rn_t    where   n_t ~ N(0,Q)  STATE EQUATION

	Parameters
	----------
	y : np.array
		The time series data

	Z : np.array
		Design matrix for state matrix a

	H : np.array
		Covariance matrix for measurement noise

	T : np.array
		Design matrix for lagged state matrix in state equation

	Q : np.array
		Covariance matrix for state evolution noise

	R : np.array
		Scale matrix for state equation covariance matrix

	mu : float
		Constant term for measurement equation

	Returns
	----------
	alpha : np.array
		Smoothed states

	V : np.array
		Variance of smoothed states
	"""		

	# Filtering matrices
	a = np.zeros((T.shape[0],y.shape[0]+1)) 
	a[0][0] = np.mean(y[0:5]) # Initialization
	P = np.ones((a.shape[0],a.shape[0],y.shape[0]+1))*(10**7) # diffuse prior asumed
	L = np.zeros((a.shape[0],a.shape[0],y.shape[0]+1))
	K = np.zeros((a.shape[0],y.shape[0]))
	v = np.zeros(y.shape[0])
	F = np.zeros((H.shape[0],H.shape[1],y.shape[0]))

	# Smoothing matrices
	N = np.zeros((a.shape[0],a.shape[0],y.shape[0]+1))
	V = np.zeros((a.shape[0],a.shape[0],y.shape[0]+1))
	alpha = np.zeros((T.shape[0],y.shape[0]+1)) 
	r = np.zeros((T.shape[0],y.shape[0]+1)) 

	# FORWARDS (FILTERING)
	for t in range(0,y.shape[0]):
		v[t] = y[t] - np.dot(Z,a[:,t]) - mu

		F[:,:,t] = np.dot(np.dot(Z,P[:,:,t]),Z.T) + H.ravel()[0]

		K[:,t] = np.dot(np.dot(T,P[:,:,t]),Z.T)/(F[:,:,t]).ravel()[0]

		L[:,:,t] = T - np.dot(K[:,t],Z)

		a[:,t+1] = np.dot(T,a[:,t]) + np.dot(K[:,t],v[t]) 

		P[:,:,t+1] = np.dot(np.dot(T,P[:,:,t]),T.T) + np.dot(np.dot(R,Q),R.T) - F[:,:,t].ravel()[0]*np.dot(np.array([K[:,t]]).T,np.array([K[:,t]]))

	# BACKWARDS (SMOOTHING)
	for t in reversed(range(y.shape[0])):
		if t != 0:
			L[:,:,t] = T - np.dot(K[:,t],Z)
			r[:,t-1] = np.dot(Z.T,v[t])/(F[:,:,t]).ravel()[0]
			N[:,:,t-1] = np.dot(Z.T,Z)/(F[:,:,t]).ravel()[0] + np.dot(np.dot(L[:,:,t].T,N[:,:,t]),L[:,:,t])
			alpha[:,t] = a[:,t] + np.dot(P[:,:,t],r[:,t-1])
			V[:,:,t] = P[:,:,t] - np.dot(np.dot(P[:,:,t],N[:,:,t-1]),P[:,:,t])
		else:
			alpha[:,t] = a[:,t]
			V[:,:,t] = P[:,:,t] 

	return alpha, V

def univariate_kalman(y,Z,H,T,Q,R,mu):
	""" Kalman filtering for univariate time series

	Notes
	----------

	y = Za_t + e_t         where   e_t ~ N(0,H)  MEASUREMENT EQUATION
	a_t = Ta_t-1 + Rn_t    where   n_t ~ N(0,Q)  STATE EQUATION

	Parameters
	----------
	y : np.array
		The time series data

	Z : np.array
		Design matrix for state matrix a

	H : np.array
		Covariance matrix for measurement noise

	T : np.array
		Design matrix for lagged state matrix in state equation

	Q : np.array
		Covariance matrix for state evolution noise

	R : np.array
		Scale matrix for state equation covariance matrix

	mu : float
		Constant term for measurement equation

	Returns
	----------
	a : np.array
		Filtered states

	P : np.array
		Filtered variances

	K : np.array
		Kalman Gain matrices

	F : np.array
		Signal-to-noise term

	v : np.array
		Residuals
	"""			

	a = np.zeros((T.shape[0],y.shape[0]+1)) 
	a[0][0] = np.mean(y[0:5]) # Initialization
	P = np.ones((a.shape[0],a.shape[0],y.shape[0]+1))*(10**7) # diffuse prior asumed

	K = np.zeros((a.shape[0],y.shape[0]))
	v = np.zeros(y.shape[0])
	F = np.zeros((H.shape[0],H.shape[1],y.shape[0]))

	for t in range(0,y.shape[0]):
		v[t] = y[t] - np.dot(Z,a[:,t]) - mu

		F[:,:,t] = np.dot(np.dot(Z,P[:,:,t]),Z.T) + H.ravel()[0]

		K[:,t] = np.dot(np.dot(T,P[:,:,t]),Z.T)/(F[:,:,t]).ravel()[0]

		a[:,t+1] = np.dot(T,a[:,t]) + np.dot(K[:,t],v[t]) 

		P[:,:,t+1] = np.dot(np.dot(T,P[:,:,t]),T.T) + np.dot(np.dot(R,Q),R.T) - F[:,:,t].ravel()[0]*np.dot(np.array([K[:,t]]).T,np.array([K[:,t]]))

	return a, P, K, F, v

def univariate_kalman_smoother(y,Z,T,a,P,K,F,v):
	""" Kalman filtering for univariate time series

	Notes
	----------

	y = Za_t + e_t         where   e_t ~ N(0,H)  MEASUREMENT EQUATION
	a_t = Ta_t-1 + Rn_t    where   n_t ~ N(0,Q)  STATE EQUATION

	Parameters
	----------
	y : np.array
		The time series data

	Z : np.array
		Design matrix for state matrix a

	H : np.array
		Covariance matrix for measurement noise

	T : np.array
		Design matrix for lagged state matrix in state equation

	a : np.array
		Filtered states

	P : np.array
		Filtered variances

	K : np.array
		Kalman gain matrices

	F : np.array
		Signal-to-noise term

	v : np.array
		Residuals

	Returns
	----------
	alpha : np.array
		Smoothed states

	V : np.array
		Variance of smoothed states
	"""			

	N = np.zeros((a.shape[0],a.shape[0],y.shape[0]+1))
	L = np.zeros((a.shape[0],a.shape[0],y.shape[0]+1))
	V = np.zeros((a.shape[0],a.shape[0],y.shape[0]+1))
	alpha = np.zeros((T.shape[0],y.shape[0]+1)) 
	r = np.zeros((T.shape[0],y.shape[0]+1)) 

	for t in reversed(range(y.shape[0])):
		if t != 0:
			L[:,:,t] = T - np.dot(K[:,t],Z)
			r[:,t-1] = np.dot(Z.T,v[t])/(F[:,:,t]).ravel()[0]
			N[:,:,t-1] = np.dot(Z.T,Z)/(F[:,:,t]).ravel()[0] + np.dot(np.dot(L[:,:,t].T,N[:,:,t]),L[:,:,t])
			alpha[:,t] = a[:,t] + np.dot(P[:,:,t],r[:,t-1])
			V[:,:,t] = P[:,:,t] - np.dot(np.dot(P[:,:,t],N[:,:,t-1]),P[:,:,t])
		else:
			alpha[:,t] = a[:,t]
			V[:,:,t] = P[:,:,t] 
	return alpha, V

def univariate_kalman_fcst(y,Z,H,T,Q,R,mu,h):
	""" Kalman filtering for univariate time series

	Notes
	----------

	y = Za_t + e_t         where   e_t ~ N(0,H)  MEASUREMENT EQUATION
	a_t = Ta_t-1 + Rn_t    where   n_t ~ N(0,Q)  STATE EQUATION

	Parameters
	----------
	y : np.array
		The time series data

	Z : np.array
		Design matrix for state matrix a

	H : np.array
		Covariance matrix for measurement noise

	T : np.array
		Design matrix for lagged state matrix in state equation

	Q : np.array
		Covariance matrix for state evolution noise

	R : np.array
		Scale matrix for state equation covariance matrix

	mu : float
		Constant term for measurement equation

	Returns
	----------
	a : np.array
		Forecasted states

	P : np.array
		Variance of forecasted states
	"""			

	a = np.zeros((T.shape[0],y.shape[0]+1+h))
	a[0][0] = np.mean(y[0:5]) # Initialization
	P = np.ones((a.shape[0],a.shape[0],y.shape[0]+1+h))*(10**7) # diffuse prior asumed

	K = np.zeros((a.shape[0],y.shape[0]+h))
	v = np.zeros(y.shape[0]+h)
	F = np.zeros((H.shape[0],H.shape[1],y.shape[0]+h))

	for t in range(0,y.shape[0]+h):
		if t >= y.shape[0]:
			v[t] = 0
			F[:,:,t] = 10**7
			K[:,t] = np.zeros(a.shape[0])
		else:
			v[t] = y[t] - np.dot(Z,a[:,t]) - mu
			F[:,:,t] = np.dot(np.dot(Z,P[:,:,t]),Z.T) + H.ravel()[0]
			K[:,t] = np.dot(np.dot(T,P[:,:,t]),Z.T)/(F[:,:,t]).ravel()[0]

		a[:,t+1] = np.dot(T,a[:,t]) + np.dot(K[:,t],v[t]) 

		P[:,:,t+1] = np.dot(np.dot(T,P[:,:,t]),T.T) + np.dot(np.dot(R,Q),R.T) - F[:,:,t].ravel()[0]*np.dot(np.array([K[:,t]]).T,np.array([K[:,t]]))

	return a, P




def nl_univariate_KFS(y,Z,H,T,Q,R,mu):
	""" Kalman filtering and smoothing for univariate time series

	Notes
	----------

	y = mu + Za_t + e_t         where   e_t ~ N(0,H)  MEASUREMENT EQUATION
	a_t = Ta_t-1 + Rn_t    where   n_t ~ N(0,Q)  STATE EQUATION

	Parameters
	----------
	y : np.array
		The time series data

	Z : np.array
		Design matrix for state matrix a

	H : np.array
		Covariance matrix for measurement noise

	T : np.array
		Design matrix for lagged state matrix in state equation

	Q : np.array
		Covariance matrix for state evolution noise

	R : np.array
		Scale matrix for state equation covariance matrix

	mu : float
		Constant term for measurement equation

	Returns
	----------
	alpha : np.array
		Smoothed states

	V : np.array
		Variance of smoothed states
	"""		

	# Filtering matrices
	a = np.zeros((T.shape[0],y.shape[0]+1)) # Initialization
	P = np.ones((a.shape[0],a.shape[0],y.shape[0]+1))*(10**7) # diffuse prior asumed
	L = np.zeros((a.shape[0],a.shape[0],y.shape[0]+1))
	K = np.zeros((a.shape[0],y.shape[0]))
	v = np.zeros(y.shape[0])
	F = np.zeros((1,1,y.shape[0]))

	# Smoothing matrices
	N = np.zeros((a.shape[0],a.shape[0],y.shape[0]))
	V = np.zeros((a.shape[0],a.shape[0],y.shape[0]))
	alpha = np.zeros((T.shape[0],y.shape[0])) 
	r = np.zeros((T.shape[0],y.shape[0])) 

	# FORWARDS (FILTERING)
	for t in range(0,y.shape[0]):
		v[t] = y[t] - np.dot(Z,a[:,t]) - mu[t]

		F[:,:,t] = np.dot(np.dot(Z,P[:,:,t]),Z.T) + H[t].ravel()[0]

		K[:,t] = np.dot(np.dot(T,P[:,:,t]),Z.T)/(F[:,:,t]).ravel()[0]

		L[:,:,t] = T - np.dot(K[:,t],Z)

		if t != (y.shape[0]-1):
		
			a[:,t+1] = np.dot(T,a[:,t]) + np.dot(K[:,t],v[t]) 

			P[:,:,t+1] = np.dot(np.dot(T,P[:,:,t]),T.T) + np.dot(np.dot(R,Q),R.T) - F[:,:,t].ravel()[0]*np.dot(np.array([K[:,t]]).T,np.array([K[:,t]]))

	# BACKWARDS (SMOOTHING)
	for t in reversed(range(y.shape[0])):
		if t != 0:
			L[:,:,t] = T - np.dot(K[:,t],Z)
			r[:,t-1] = np.dot(Z.T,v[t])/(F[:,:,t]).ravel()[0]
			N[:,:,t-1] = np.dot(Z.T,Z)/(F[:,:,t]).ravel()[0] + np.dot(np.dot(L[:,:,t].T,N[:,:,t]),L[:,:,t])
			alpha[:,t] = a[:,t] + np.dot(P[:,:,t],r[:,t-1])
			V[:,:,t] = P[:,:,t] - np.dot(np.dot(P[:,:,t],N[:,:,t-1]),P[:,:,t])
		else:
			alpha[:,t] = a[:,t]
			V[:,:,t] = P[:,:,t] 

	return alpha, V

def nl_univariate_kalman(y,Z,H,T,Q,R,mu):
	""" Kalman filtering for univariate time series

	Notes
	----------

	y = Za_t + e_t         where   e_t ~ N(0,H)  MEASUREMENT EQUATION
	a_t = Ta_t-1 + Rn_t    where   n_t ~ N(0,Q)  STATE EQUATION

	Parameters
	----------
	y : np.array
		The time series data

	Z : np.array
		Design matrix for state matrix a

	H : np.array
		Covariance matrix for measurement noise

	T : np.array
		Design matrix for lagged state matrix in state equation

	Q : np.array
		Covariance matrix for state evolution noise

	R : np.array
		Scale matrix for state equation covariance matrix

	mu : float
		Constant term for measurement equation

	Returns
	----------
	a : np.array
		Filtered states

	P : np.array
		Filtered variances

	K : np.array
		Kalman Gain matrices

	F : np.array
		Signal-to-noise term

	v : np.array
		Residuals
	"""			

	a = np.zeros((T.shape[0],y.shape[0]+1)) # Initialization
	P = np.ones((a.shape[0],a.shape[0],y.shape[0]+1))*(10**7) # diffuse prior asumed

	K = np.zeros((a.shape[0],y.shape[0]))
	v = np.zeros(y.shape[0])
	F = np.zeros((1,1,y.shape[0]))

	for t in range(0,y.shape[0]):
		v[t] = y[t] - np.dot(Z,a[:,t]) - mu[t]

		F[:,:,t] = np.dot(np.dot(Z,P[:,:,t]),Z.T) + H[t].ravel()[0]

		K[:,t] = np.dot(np.dot(T,P[:,:,t]),Z.T)/(F[:,:,t]).ravel()[0]

		a[:,t+1] = np.dot(T,a[:,t]) + np.dot(K[:,t],v[t]) 

		P[:,:,t+1] = np.dot(np.dot(T,P[:,:,t]),T.T) + np.dot(np.dot(R,Q),R.T) - F[:,:,t].ravel()[0]*np.dot(np.array([K[:,t]]).T,np.array([K[:,t]]))

	return a, P, K, F, v


def dl_univariate_KFS(y,Z,H,T,Q,R,mu):
	""" Kalman filtering and smoothing for univariate time series

	Notes
	----------

	y = mu + Za_t + e_t         where   e_t ~ N(0,H)  MEASUREMENT EQUATION
	a_t = Ta_t-1 + Rn_t    where   n_t ~ N(0,Q)  STATE EQUATION

	Parameters
	----------
	y : np.array
		The time series data

	Z : np.array
		Design matrix for state matrix a

	H : np.array
		Covariance matrix for measurement noise

	T : np.array
		Design matrix for lagged state matrix in state equation

	Q : np.array
		Covariance matrix for state evolution noise

	R : np.array
		Scale matrix for state equation covariance matrix

	mu : float
		Constant term for measurement equation

	Returns
	----------
	alpha : np.array
		Smoothed states

	V : np.array
		Variance of smoothed states
	"""		

	# Filtering matrices
	a = np.zeros((T.shape[0],y.shape[0]+1)) 
	P = np.ones((a.shape[0],a.shape[0],y.shape[0]+1))*(10**7) # diffuse prior asumed
	L = np.zeros((a.shape[0],a.shape[0],y.shape[0]+1))
	K = np.zeros((a.shape[0],y.shape[0]))
	v = np.zeros(y.shape[0])
	F = np.zeros((H.shape[0],H.shape[1],y.shape[0]))

	# Smoothing matrices
	N = np.zeros((a.shape[0],a.shape[0],y.shape[0]+1))
	V = np.zeros((a.shape[0],a.shape[0],y.shape[0]+1))
	alpha = np.zeros((T.shape[0],y.shape[0]+1)) 
	r = np.zeros((T.shape[0],y.shape[0]+1)) 

	# FORWARDS (FILTERING)
	for t in range(0,y.shape[0]):
		v[t] = y[t] - np.dot(Z[t],a[:,t]) - mu

		F[:,:,t] = np.dot(np.dot(Z[t],P[:,:,t]),Z[t].T) + H.ravel()[0]

		K[:,t] = np.dot(np.dot(T,P[:,:,t]),Z[t].T)/(F[:,:,t]).ravel()[0]

		L[:,:,t] = T - np.dot(K[:,t],Z[t])

		a[:,t+1] = np.dot(T,a[:,t]) + np.dot(K[:,t],v[t]) 

		P[:,:,t+1] = np.dot(np.dot(T,P[:,:,t]),T.T) + np.dot(np.dot(R,Q),R.T) - F[:,:,t].ravel()[0]*np.dot(np.array([K[:,t]]).T,np.array([K[:,t]]))

	# BACKWARDS (SMOOTHING)
	for t in reversed(range(y.shape[0])):
		if t != 0:
			L[:,:,t] = T - np.dot(K[:,t],Z[t])
			r[:,t-1] = np.dot(Z[t].T,v[t])/(F[:,:,t]).ravel()[0]
			N[:,:,t-1] = np.dot(Z[t].T,Z[t])/(F[:,:,t]).ravel()[0] + np.dot(np.dot(L[:,:,t].T,N[:,:,t]),L[:,:,t])
			alpha[:,t] = a[:,t] + np.dot(P[:,:,t],r[:,t-1])
			V[:,:,t] = P[:,:,t] - np.dot(np.dot(P[:,:,t],N[:,:,t-1]),P[:,:,t])
		else:
			alpha[:,t] = a[:,t]
			V[:,:,t] = P[:,:,t] 

	return alpha, V

def dl_univariate_kalman(y,Z,H,T,Q,R,mu):
	""" Kalman filtering for univariate time series

	Notes
	----------

	y = Za_t + e_t         where   e_t ~ N(0,H)  MEASUREMENT EQUATION
	a_t = Ta_t-1 + Rn_t    where   n_t ~ N(0,Q)  STATE EQUATION

	Parameters
	----------
	y : np.array
		The time series data

	Z : np.array
		Design matrix for state matrix a

	H : np.array
		Covariance matrix for measurement noise

	T : np.array
		Design matrix for lagged state matrix in state equation

	Q : np.array
		Covariance matrix for state evolution noise

	R : np.array
		Scale matrix for state equation covariance matrix

	mu : float
		Constant term for measurement equation

	Returns
	----------
	a : np.array
		Filtered states

	P : np.array
		Filtered variances

	K : np.array
		Kalman Gain matrices

	F : np.array
		Signal-to-noise term

	v : np.array
		Residuals
	"""			

	a = np.zeros((T.shape[0],y.shape[0]+1)) 
	P = np.ones((a.shape[0],a.shape[0],y.shape[0]+1))*(10**7) # diffuse prior asumed

	K = np.zeros((a.shape[0],y.shape[0]))
	v = np.zeros(y.shape[0])
	F = np.zeros((H.shape[0],H.shape[1],y.shape[0]))

	for t in range(0,y.shape[0]):
		v[t] = y[t] - np.dot(Z[t],a[:,t]) - mu

		F[:,:,t] = np.dot(np.dot(Z[t],P[:,:,t]),Z[t].T) + H.ravel()[0]

		K[:,t] = np.dot(np.dot(T,P[:,:,t]),Z[t].T)/(F[:,:,t]).ravel()[0]

		a[:,t+1] = np.dot(T,a[:,t]) + np.dot(K[:,t],v[t]) 

		P[:,:,t+1] = np.dot(np.dot(T,P[:,:,t]),T.T) + np.dot(np.dot(R,Q),R.T) - F[:,:,t].ravel()[0]*np.dot(np.array([K[:,t]]).T,np.array([K[:,t]]))

	return a, P, K, F, v

def dl_univariate_kalman_fcst(y,Z,H,T,Q,R,mu,h):
	""" Kalman filtering for univariate time series

	Notes
	----------

	y = Za_t + e_t         where   e_t ~ N(0,H)  MEASUREMENT EQUATION
	a_t = Ta_t-1 + Rn_t    where   n_t ~ N(0,Q)  STATE EQUATION

	Parameters
	----------
	y : np.array
		The time series data

	Z : np.array
		Design matrix for state matrix a

	H : np.array
		Covariance matrix for measurement noise

	T : np.array
		Design matrix for lagged state matrix in state equation

	Q : np.array
		Covariance matrix for state evolution noise

	R : np.array
		Scale matrix for state equation covariance matrix

	mu : float
		Constant term for measurement equation

	Returns
	----------
	a : np.array
		Forecasted states

	P : np.array
		Variance of forecasted states
	"""			

	a = np.zeros((T.shape[0],y.shape[0]+1+h))
	P = np.ones((a.shape[0],a.shape[0],y.shape[0]+1+h))*(10**7) # diffuse prior asumed

	K = np.zeros((a.shape[0],y.shape[0]+h))
	v = np.zeros(y.shape[0]+h)
	F = np.zeros((H.shape[0],H.shape[1],y.shape[0]+h))

	for t in range(0,y.shape[0]+h):
		if t >= y.shape[0]:
			v[t] = 0
			F[:,:,t] = 10**7
			K[:,t] = np.zeros(a.shape[0])
		else:
			v[t] = y[t] - np.dot(Z[t],a[:,t]) - mu
			F[:,:,t] = np.dot(np.dot(Z[t],P[:,:,t]),Z[t].T) + H.ravel()[0]
			K[:,t] = np.dot(np.dot(T,P[:,:,t]),Z[t].T)/(F[:,:,t]).ravel()[0]

		a[:,t+1] = np.dot(T,a[:,t]) + np.dot(K[:,t],v[t]) 

		P[:,:,t+1] = np.dot(np.dot(T,P[:,:,t]),T.T) + np.dot(np.dot(R,Q),R.T) - F[:,:,t].ravel()[0]*np.dot(np.array([K[:,t]]).T,np.array([K[:,t]]))

	return a, P