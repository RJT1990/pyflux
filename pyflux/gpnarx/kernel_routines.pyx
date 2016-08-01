import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def SE_K_matrix(np.ndarray[double,ndim=2] X, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((X.shape[0],X.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,X.shape[0]):
        for j in range(0,X.shape[0]):
            if i == j:
                K[i,j] = parm[2]
            else:
                K[i,j] = parm[2]*np.exp(-0.5*np.sum(np.power(X[i] - X[j],2))/np.power(parm[1],2))
    return K

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def SE_Kstar_matrix(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Xstar, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((X.shape[0],Xstar.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,X.shape[0]):
        for j in range(0,Xstar.shape[0]):
            if i == j:
                K[i,j] = parm[2]
            else:
                K[i,j] = parm[2]*np.exp(-0.5*np.sum(np.power(X[i] - Xstar[j],2))/np.power(parm[1],2))
    return K

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def SE_Kstarstar_matrix(np.ndarray[double,ndim=2] Xstar, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((Xstar.shape[0],Xstar.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,Xstar.shape[0]):
        for j in range(0,Xstar.shape[0]):
            if i == j:
                K[i,j] = parm[2]
            else:
                K[i,j] = parm[2]*np.exp(-0.5*np.sum(np.power(Xstar[i] - Xstar[j],2))/np.power(parm[1],2))
    return K

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def SE_K_arbitrary_X_matrix(np.ndarray[double,ndim=2] Xstar1, np.ndarray[double,ndim=2] Xstar2, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((Xstar1.shape[0],Xstar2.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,Xstar1.shape[0]):
        for j in range(0,Xstar2.shape[0]):
            if i == j:
                K[i,j] = parm[2]
            else:
                K[i,j] = parm[2]*np.exp(-0.5*np.sum(np.power(Xstar1[i] - Xstar2[j],2))/np.power(parm[1],2))
    
    return K



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def OU_K_matrix(np.ndarray[double,ndim=2] X, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((X.shape[0],X.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,X.shape[0]):
        for j in range(0,X.shape[0]):
            if i == j:
                K[i,j] = parm[2]
            else:
                K[i,j] = parm[2]*np.exp(-np.sum(np.abs(X[i] - X[j]))/parm[1])

    return K

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def OU_Kstar_matrix(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Xstar, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((X.shape[0],Xstar.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,X.shape[0]):
        for j in range(0,Xstar.shape[0]):
            if i == j:
                K[i,j] = parm[2]
            else:
                K[i,j] = parm[2]*np.exp(-np.sum(np.abs(X[i] - Xstar[j]))/parm[1])
    return K

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def OU_Kstarstar_matrix(np.ndarray[double,ndim=2] Xstar, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((Xstar.shape[0],Xstar.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,Xstar.shape[0]):
        for j in range(0,Xstar.shape[0]):
            if i == j:
                K[i,j] = parm[2]
            else:
                K[i,j] = parm[2]*np.exp(-np.sum(np.abs(Xstar[i] - Xstar[j]))/parm[1])
    return K

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def OU_K_arbitrary_X_matrix(np.ndarray[double,ndim=2] Xstar1, np.ndarray[double,ndim=2] Xstar2, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((Xstar1.shape[0],Xstar2.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,Xstar1.shape[0]):
        for j in range(0,Xstar2.shape[0]):
            if i == j:
                K[i,j] = parm[2]
            else:
                K[i,j] = parm[2]*np.exp(-np.sum(np.abs(Xstar1[i] - Xstar2[j]))/parm[1])
    
    return K



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ARD_K_matrix(np.ndarray[double,ndim=2] X, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((X.shape[0],X.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,X.shape[0]):
        for j in range(0,X.shape[0]):
            if i == j:
                K[i,j] = parm[-1]
            else:
                K[i,j] = parm[-1]*np.exp(-0.5*np.sum(np.power(X[i] - X[j],2)/np.power(parm[1:-1],2)))
    
    return K


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ARD_Kstar_matrix(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Xstar, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((X.shape[0],Xstar.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0, X.shape[0]):
        for j in range(0, Xstar.shape[0]):
            if i == j:
                K[i,j] = parm[-1]
            else:
                K[i,j] = parm[-1]*np.exp(-0.5*np.sum(np.power(X[i] - Xstar[j],2)/np.power(parm[1:-1],2)))
    
    return K

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ARD_Kstarstar_matrix(np.ndarray[double,ndim=2] Xstar, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((Xstar.shape[0],Xstar.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0, Xstar.shape[0]):
        for j in range(0, Xstar.shape[0]):
            if i == j:
                K[i,j] = parm[-1]
            else:
                K[i,j] = parm[-1]*np.exp(-0.5*np.sum(np.power(Xstar[i] - Xstar[j],2)/np.power(parm[1:-1],2)))
    
    return K

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def ARD_K_arbitrary_X_matrix(np.ndarray[double,ndim=2] Xstar1, np.ndarray[double,ndim=2] Xstar2, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((Xstar1.shape[0],Xstar2.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0, Xstar1.shape[0]):
        for j in range(0, Xstar2.shape[0]):
            if i == j:
                K[i,j] = parm[-1]
            else:
                K[i,j] = parm[-1]*np.exp(-0.5*np.sum(np.power(Xstar1[i] - Xstar2[j],2)/np.power(parm[1:-1],2)))
    
    return K




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def RQ_K_matrix(np.ndarray[double,ndim=2] X, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((X.shape[0],X.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,X.shape[0]):
        for j in range(0,X.shape[0]):
            K[i,j] = parm[1]*(1.0 + 0.5*np.sum(np.power(X[i] - X[j],2))/((parm[2]**2)*parm[1]))**-parm[1]

    return K


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def RQ_Kstar_matrix(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Xstar, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((X.shape[0],Xstar.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,X.shape[0]):
        for j in range(0,Xstar.shape[0]):
            K[i,j] = parm[3]*(1.0 + 0.5*np.sum(np.power(X[i] - Xstar[j],2))/((parm[2]**2)*parm[1]))**-parm[1]

    return K

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def RQ_Kstarstar_matrix(np.ndarray[double,ndim=2] Xstar, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((Xstar.shape[0],Xstar.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,Xstar.shape[0]):
        for j in range(0,Xstar.shape[0]):
            K[i,j] = parm[3]*(1.0 + 0.5*np.sum(np.power(Xstar[i] - Xstar[j],2))/((parm[2]**2)*parm[1]))**-parm[1]

    return K

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def RQ_K_arbitrary_X_matrix(np.ndarray[double,ndim=2] Xstar1, np.ndarray[double,ndim=2] Xstar2, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((Xstar1.shape[0],Xstar2.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,Xstar1.shape[0]):
        for j in range(0,Xstar2.shape[0]):
            K[i,j] = parm[3]*(1.0 + 0.5*np.sum(np.power(Xstar1[i] - Xstar2[j],2))/((parm[2]**2)*parm[1]))**-parm[1]

    return K



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def Periodic_K_matrix(np.ndarray[double,ndim=2] X, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((X.shape[0],X.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,X.shape[0]):
        for j in range(0,X.shape[0]):
            if i == j:
                K[i,j] = parm[2]
            else:
                K[i,j] = parm[2]*np.exp(-np.sum(np.power(-2.0*np.sin(X[i] - X[j]),2))/parm[1])
    return K


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def Periodic_Kstar_matrix(np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] Xstar, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((X.shape[0],Xstar.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,X.shape[0]):
        for j in range(0,Xstar.shape[0]):
            if i == j:
                K[i,j] = parm[2]
            else:
                K[i,j] = parm[2]*np.exp(-np.sum(np.power(-2.0*np.sin(X[i] - Xstar[j]),2))/parm[1])
    return K

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def Periodic_Kstarstar_matrix(np.ndarray[double,ndim=2] Xstar, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((Xstar.shape[0],Xstar.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,Xstar.shape[0]):
        for j in range(0,Xstar.shape[0]):
            if i == j:
                K[i,j] = parm[2]
            else:
                K[i,j] = parm[2]*np.exp(-np.sum(np.power(-2.0*np.sin(Xstar[i] - Xstar[j]),2))/parm[1])
    return K

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def Periodic_K_arbitrary_X_matrix(np.ndarray[double,ndim=2] Xstar1, np.ndarray[double,ndim=2] Xstar2, double[:] parm):

    cdef np.ndarray[double, ndim=2, mode="c"] K = np.zeros((Xstar1.shape[0],Xstar2.shape[0]))
    cdef Py_ssize_t i, j

    for i in range(0,Xstar1.shape[0]):
        for j in range(0,Xstar2.shape[0]):
            if i == j:
                K[i,j] = parm[2]
            else:
                K[i,j] = parm[2]*np.exp(-np.sum(np.power(-2.0*np.sin(Xstar1[i] - Xstar2[j]),2))/parm[1])
    return K