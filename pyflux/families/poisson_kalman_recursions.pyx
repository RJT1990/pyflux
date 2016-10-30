import numpy as np
cimport numpy as np
cimport cython

# TO DO: REFACTOR AND COMBINE THESE SCRIPTS TO USE A SINGLE KALMAN FILTER/SMOOTHER SCRIPT
# Main differences between these functions are whether they treat certain matrices as
# constant or not

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

def nld_univariate_KFS(y,Z,H,T,Q,R,mu):
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
        v[t] = y[t] - np.dot(Z[t],a[:,t]) - mu[t]

        F[:,:,t] = np.dot(np.dot(Z[t],P[:,:,t]),Z[t].T) + H[t].ravel()[0]

        K[:,t] = np.dot(np.dot(T,P[:,:,t]),Z[t].T)/(F[:,:,t]).ravel()[0]

        L[:,:,t] = T - np.dot(K[:,t],Z[t])

        if t != (y.shape[0]-1):
        
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
