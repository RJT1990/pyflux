import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_recursion(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int ar_terms, int sc_terms, int Y_len, score_function, link, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[0]/(1.0-np.sum(parameters[1:(ar_terms+1)]))
        else:
            theta[t] += np.dot(parameters[1:1+ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[1+ar_terms:1+ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = score_function(Y[t], link(theta[t]), scale, shape, skewness)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gasx_recursion(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int ar_terms, int sc_terms, int Y_len, score_function, link, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[ar_terms+sc_terms]/(1-np.sum(parameters[:ar_terms]))
        else:
            theta[t] += np.dot(parameters[:ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[ar_terms:ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = score_function(Y[t], link(theta[t]), scale, shape, skewness)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llev_recursion(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int Y_len, score_function, link, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta[t-1] + parameters[0]*model_scores[t-1]

        model_scores[t] = score_function(Y[t], link(theta[t]), scale, shape, skewness)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llt_recursion(double[:] parameters, double[:] theta, double[:] theta_t, double[:] model_scores, double[:] Y, 
    int Y_len, score_function, link, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta_t[t-1] + theta[t-1] + parameters[0]*model_scores[t-1]
            theta_t[t] = theta_t[t-1] + parameters[1]*model_scores[t-1]

        model_scores[t] = score_function(Y[t], link(theta[t]), scale, shape, skewness)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_reg_recursion(double[:] parameters, double[:] theta, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] coefficients, np.ndarray[double,ndim=2] model_scores, 
    double[:] Y, int Y_len, score_function, link, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        theta[t] = np.dot(X[t],coefficients[:,t])
        model_scores[:,t] = score_function(X[t],Y[t],link(theta[t]),scale,shape,skewness)
        coefficients[:,t+1] = coefficients[:,t] + parameters[0:X.shape[1]]*model_scores[:,t] 

    return theta, model_scores, coefficients