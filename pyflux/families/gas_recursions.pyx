import numpy as np
import scipy.special as sp
cimport numpy as np
cimport cython

from libc.math cimport exp, abs, M_PI

cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b

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
def gas_recursion_exponential_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[0]/(1.0-np.sum(parameters[1:(ar_terms+1)]))
        else:
            theta[t] += np.dot(parameters[1:1+ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[1+ar_terms:1+ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = double_min(double_max(1.0 - (exp(theta[t])*Y[t]), -10000), 10000)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_recursion_exponential_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[0]/(1.0-np.sum(parameters[1:(ar_terms+1)]))
        else:
            theta[t] += np.dot(parameters[1:1+ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[1+ar_terms:1+ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = double_min(double_max(1.0 - (exp(theta[t])*Y[t]), -10000), 10000)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_recursion_laplace_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[0]/(1.0-np.sum(parameters[1:(ar_terms+1)]))
        else:
            theta[t] += np.dot(parameters[1:1+ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[1+ar_terms:1+ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = (Y[t]-theta[t])/(scale*abs(Y[t]-theta[t]))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_recursion_laplace_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[0]/(1.0-np.sum(parameters[1:(ar_terms+1)]))
        else:
            theta[t] += np.dot(parameters[1:1+ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[1+ar_terms:1+ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = ((Y[t]-theta[t])/float(scale*abs(Y[t]-theta[t]))) / (-(np.power(Y[t]-theta[t],2) - np.power(abs(theta[t]-Y[t]),2))/(scale*np.power(abs(theta[t]-Y[t]),3)))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_recursion_normal_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[0]/(1.0-np.sum(parameters[1:(ar_terms+1)]))
        else:
            theta[t] += np.dot(parameters[1:1+ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[1+ar_terms:1+ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = (Y[t]-theta[t])/np.power(scale,2)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_recursion_normal_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[0]/(1.0-np.sum(parameters[1:(ar_terms+1)]))
        else:
            theta[t] += np.dot(parameters[1:1+ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[1+ar_terms:1+ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = Y[t]-theta[t]
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_recursion_poisson_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[0]/(1.0-np.sum(parameters[1:(ar_terms+1)]))
        else:
            theta[t] += np.dot(parameters[1:1+ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[1+ar_terms:1+ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = double_min(double_max(Y[t] - exp(theta[t]), -10000), 10000)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_recursion_poisson_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[0]/(1.0-np.sum(parameters[1:(ar_terms+1)]))
        else:
            theta[t] += np.dot(parameters[1:1+ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[1+ar_terms:1+ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = double_min(double_max(Y[t]/exp(theta[t]) - 1.0, -10000), 10000)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_recursion_t_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[0]/(1.0-np.sum(parameters[1:(ar_terms+1)]))
        else:
            theta[t] += np.dot(parameters[1:1+ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[1+ar_terms:1+ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(Y[t]-theta[t],2)/shape))
        
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_recursion_t_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[0]/(1.0-np.sum(parameters[1:(ar_terms+1)]))
        else:
            theta[t] += np.dot(parameters[1:1+ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[1+ar_terms:1+ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(Y[t]-theta[t],2)/shape))/((shape+1)*((np.power(scale,2)*shape) - np.power(Y[t]-theta[t],2))/np.power((np.power(scale,2)*shape) + np.power(Y[t]-theta[t],2),2))
    
    return theta, model_scores



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_recursion_cauchy_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[0]/(1.0-np.sum(parameters[1:(ar_terms+1)]))
        else:
            theta[t] += np.dot(parameters[1:1+ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[1+ar_terms:1+ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = 2.0*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(Y[t]-theta[t],2)))
        
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_recursion_cauchy_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[0]/(1.0-np.sum(parameters[1:(ar_terms+1)]))
        else:
            theta[t] += np.dot(parameters[1:1+ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[1+ar_terms:1+ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = 2.0*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(Y[t]-theta[t],2)))/((2.0)*((np.power(scale,2)) - np.power(Y[t]-theta[t],2))/np.power((np.power(scale,2)) + np.power(Y[t]-theta[t],2),2))
    
    return theta, model_scores



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_recursion_skewt_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[0]/(1.0-np.sum(parameters[1:(ar_terms+1)]))
        else:
            theta[t] += np.dot(parameters[1:1+ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[1+ar_terms:1+ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(M_PI)*sp.gamma(shape/2.0))
        mean = theta[t] + (skewness - (1.0/skewness))*scale*m1
        if (Y[t]-theta[t])>=0:
            model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(skewness*scale,2) + (np.power(Y[t]-theta[t],2)/shape))
        else:
            model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(skewness*(Y[t]-theta[t]),2)/shape))

    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_recursion_skewt_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, 
    int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[0]/(1.0-np.sum(parameters[1:(ar_terms+1)]))
        else:
            theta[t] += np.dot(parameters[1:1+ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[1+ar_terms:1+ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(M_PI)*sp.gamma(shape/2.0))
        mean = theta[t] + (skewness - (1.0/skewness))*scale*m1
        if (Y[t]-theta[t])>=0:
            model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(skewness*scale,2) + (np.power(Y[t]-theta[t],2)/shape))
        else:
            model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(skewness*(Y[t]-theta[t]),2)/shape))

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
def gasx_recursion_exponential_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[ar_terms+sc_terms]/(1-np.sum(parameters[:ar_terms]))
        else:
            theta[t] += np.dot(parameters[:ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[ar_terms:ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = double_min(double_max(1.0 - (exp(theta[t])*Y[t]), -10000), 10000)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gasx_recursion_exponential_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[ar_terms+sc_terms]/(1-np.sum(parameters[:ar_terms]))
        else:
            theta[t] += np.dot(parameters[:ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[ar_terms:ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = double_min(double_max(1.0 - (exp(theta[t])*Y[t]), -10000), 10000)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gasx_recursion_laplace_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[ar_terms+sc_terms]/(1-np.sum(parameters[:ar_terms]))
        else:
            theta[t] += np.dot(parameters[:ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[ar_terms:ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = (Y[t]-theta[t])/(scale*abs(Y[t]-theta[t]))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gasx_recursion_laplace_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[ar_terms+sc_terms]/(1-np.sum(parameters[:ar_terms]))
        else:
            theta[t] += np.dot(parameters[:ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[ar_terms:ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = ((Y[t]-theta[t])/float(scale*abs(Y[t]-theta[t]))) / (-(np.power(Y[t]-theta[t],2) - np.power(abs(theta[t]-Y[t]),2))/(scale*np.power(abs(theta[t]-Y[t]),3)))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gasx_recursion_normal_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[ar_terms+sc_terms]/(1-np.sum(parameters[:ar_terms]))
        else:
            theta[t] += np.dot(parameters[:ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[ar_terms:ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = (Y[t]-theta[t])/np.power(scale,2)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gasx_recursion_normal_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[ar_terms+sc_terms]/(1-np.sum(parameters[:ar_terms]))
        else:
            theta[t] += np.dot(parameters[:ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[ar_terms:ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = Y[t]-theta[t]
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gasx_recursion_poisson_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[ar_terms+sc_terms]/(1-np.sum(parameters[:ar_terms]))
        else:
            theta[t] += np.dot(parameters[:ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[ar_terms:ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = Y[t]-exp(theta[t])
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gasx_recursion_poisson_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[ar_terms+sc_terms]/(1-np.sum(parameters[:ar_terms]))
        else:
            theta[t] += np.dot(parameters[:ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[ar_terms:ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = Y[t]/exp(theta[t]) - 1.0
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gasx_recursion_t_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[ar_terms+sc_terms]/(1-np.sum(parameters[:ar_terms]))
        else:
            theta[t] += np.dot(parameters[:ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[ar_terms:ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(Y[t]-theta[t],2)/shape))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gasx_recursion_t_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[ar_terms+sc_terms]/(1-np.sum(parameters[:ar_terms]))
        else:
            theta[t] += np.dot(parameters[:ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[ar_terms:ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(Y[t]-theta[t],2)/shape))/((shape+1)*((np.power(scale,2)*shape) - np.power(Y[t]-theta[t],2))/np.power((np.power(scale,2)*shape) + np.power(Y[t]-theta[t],2),2))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gasx_recursion_cauchy_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[ar_terms+sc_terms]/(1-np.sum(parameters[:ar_terms]))
        else:
            theta[t] += np.dot(parameters[:ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[ar_terms:ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = 2.0*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(Y[t]-theta[t],2)))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gasx_recursion_cauchy_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[ar_terms+sc_terms]/(1-np.sum(parameters[:ar_terms]))
        else:
            theta[t] += np.dot(parameters[:ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[ar_terms:ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        model_scores[t] = 2.0*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(Y[t]-theta[t],2)))/((2.0)*((np.power(scale,2)) - np.power(Y[t]-theta[t],2))/np.power((np.power(scale,2)) + np.power(Y[t]-theta[t],2),2))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gasx_recursion_skewt_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[ar_terms+sc_terms]/(1-np.sum(parameters[:ar_terms]))
        else:
            theta[t] += np.dot(parameters[:ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[ar_terms:ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(M_PI)*sp.gamma(shape/2.0))
        mean = theta[t] + (skewness - (1.0/skewness))*scale*m1
        if (Y[t]-theta[t])>=0:
            model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(skewness*scale,2) + (np.power(Y[t]-theta[t],2)/shape))
        else:
            model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(skewness*(Y[t]-theta[t]),2)/shape))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gasx_recursion_skewt_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int ar_terms, int sc_terms, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = parameters[ar_terms+sc_terms]/(1-np.sum(parameters[:ar_terms]))
        else:
            theta[t] += np.dot(parameters[:ar_terms],theta[(t-ar_terms):t][::-1]) + np.dot(parameters[ar_terms:ar_terms+sc_terms],model_scores[(t-sc_terms):t][::-1])

        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(M_PI)*sp.gamma(shape/2.0))
        mean = theta[t] + (skewness - (1.0/skewness))*scale*m1
        if (Y[t]-theta[t])>=0:
            model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(skewness*scale,2) + (np.power(Y[t]-theta[t],2)/shape))
        else:
            model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(skewness*(Y[t]-theta[t]),2)/shape))
    
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
def gas_llev_recursion_exponential_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta[t-1] + parameters[0]*model_scores[t-1]

        model_scores[t] = double_min(double_max(1.0 - (exp(theta[t])*Y[t]), -10000), 10000)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llev_recursion_exponential_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta[t-1] + parameters[0]*model_scores[t-1]

        model_scores[t] = double_min(double_max(1.0 - (exp(theta[t])*Y[t]), -10000), 10000)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llev_recursion_laplace_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta[t-1] + parameters[0]*model_scores[t-1]

        model_scores[t] = (Y[t]-theta[t])/(scale*abs(Y[t]-theta[t]))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llev_recursion_laplace_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta[t-1] + parameters[0]*model_scores[t-1]

        model_scores[t] = ((Y[t]-theta[t])/float(scale*abs(Y[t]-theta[t]))) / (-(np.power(Y[t]-theta[t],2) - np.power(abs(theta[t]-Y[t]),2))/(scale*np.power(abs(theta[t]-Y[t]),3)))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llev_recursion_normal_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta[t-1] + parameters[0]*model_scores[t-1]

        model_scores[t] = (Y[t]-theta[t])/np.power(scale,2)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llev_recursion_normal_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta[t-1] + parameters[0]*model_scores[t-1]

        model_scores[t] = Y[t] - theta[t]
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llev_recursion_poisson_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta[t-1] + parameters[0]*model_scores[t-1]

        model_scores[t] = Y[t] - exp(theta[t])
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llev_recursion_poisson_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta[t-1] + parameters[0]*model_scores[t-1]

        model_scores[t] = Y[t]/exp(theta[t]) - 1.0
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llev_recursion_t_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta[t-1] + parameters[0]*model_scores[t-1]

        model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(Y[t]-theta[t],2)/shape))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llev_recursion_t_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta[t-1] + parameters[0]*model_scores[t-1]

        model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(Y[t]-theta[t],2)/shape))/((shape+1)*((np.power(scale,2)*shape) - np.power(Y[t]-theta[t],2))/np.power((np.power(scale,2)*shape) + np.power(Y[t]-theta[t],2),2))
    
    return theta, model_scores


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llev_recursion_cauchy_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta[t-1] + parameters[0]*model_scores[t-1]

        model_scores[t] = 2.0*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(Y[t]-theta[t],2)))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llev_recursion_cauchy_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta[t-1] + parameters[0]*model_scores[t-1]

        model_scores[t] = 2.0*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(Y[t]-theta[t],2)))/((2.0)*((np.power(scale,2)) - np.power(Y[t]-theta[t],2))/np.power((np.power(scale,2)) + np.power(Y[t]-theta[t],2),2))
    
    return theta, model_scores


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llev_recursion_skewt_orderone(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta[t-1] + parameters[0]*model_scores[t-1]

        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(M_PI)*sp.gamma(shape/2.0))
        mean = theta[t] + (skewness - (1.0/skewness))*scale*m1
        if (Y[t]-theta[t])>=0:
            model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(skewness*scale,2) + (np.power(Y[t]-theta[t],2)/shape))
        else:
            model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(skewness*(Y[t]-theta[t]),2)/shape))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llev_recursion_skewt_ordertwo(double[:] parameters, double[:] theta, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta[t-1] + parameters[0]*model_scores[t-1]

        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(M_PI)*sp.gamma(shape/2.0))
        mean = theta[t] + (skewness - (1.0/skewness))*scale*m1
        if (Y[t]-theta[t])>=0:
            model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(skewness*scale,2) + (np.power(Y[t]-theta[t],2)/shape))
        else:
            model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(skewness*(Y[t]-theta[t]),2)/shape))
    
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
def gas_llt_recursion_exponential_orderone(double[:] parameters, double[:] theta, double[:] theta_t, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta_t[t-1] + theta[t-1] + parameters[0]*model_scores[t-1]
            theta_t[t] = theta_t[t-1] + parameters[1]*model_scores[t-1]

        model_scores[t] = double_min(double_max(1.0 - (exp(theta[t])*Y[t]), -10000), 10000)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llt_recursion_exponential_ordertwo(double[:] parameters, double[:] theta, double[:] theta_t, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta_t[t-1] + theta[t-1] + parameters[0]*model_scores[t-1]
            theta_t[t] = theta_t[t-1] + parameters[1]*model_scores[t-1]

        model_scores[t] = double_min(double_max(1.0 - (exp(theta[t])*Y[t]), -10000), 10000)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llt_recursion_laplace_orderone(double[:] parameters, double[:] theta, double[:] theta_t, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta_t[t-1] + theta[t-1] + parameters[0]*model_scores[t-1]
            theta_t[t] = theta_t[t-1] + parameters[1]*model_scores[t-1]

        model_scores[t] = (Y[t]-theta[t])/(scale*abs(Y[t]-theta[t]))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llt_recursion_laplace_ordertwo(double[:] parameters, double[:] theta, double[:] theta_t, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta_t[t-1] + theta[t-1] + parameters[0]*model_scores[t-1]
            theta_t[t] = theta_t[t-1] + parameters[1]*model_scores[t-1]

        model_scores[t] = ((Y[t]-theta[t])/float(scale*abs(Y[t]-theta[t]))) / (-(np.power(Y[t]-theta[t],2) - np.power(abs(theta[t]-Y[t]),2))/(scale*np.power(abs(theta[t]-Y[t]),3)))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llt_recursion_normal_orderone(double[:] parameters, double[:] theta, double[:] theta_t, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta_t[t-1] + theta[t-1] + parameters[0]*model_scores[t-1]
            theta_t[t] = theta_t[t-1] + parameters[1]*model_scores[t-1]

        model_scores[t] = (Y[t] - theta[t])/np.power(scale,2)
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llt_recursion_normal_ordertwo(double[:] parameters, double[:] theta, double[:] theta_t, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta_t[t-1] + theta[t-1] + parameters[0]*model_scores[t-1]
            theta_t[t] = theta_t[t-1] + parameters[1]*model_scores[t-1]

        model_scores[t] = Y[t] - theta[t]
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llt_recursion_poisson_orderone(double[:] parameters, double[:] theta, double[:] theta_t, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta_t[t-1] + theta[t-1] + parameters[0]*model_scores[t-1]
            theta_t[t] = theta_t[t-1] + parameters[1]*model_scores[t-1]

        model_scores[t] = Y[t] - exp(theta[t])
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llt_recursion_poisson_ordertwo(double[:] parameters, double[:] theta, double[:] theta_t, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta_t[t-1] + theta[t-1] + parameters[0]*model_scores[t-1]
            theta_t[t] = theta_t[t-1] + parameters[1]*model_scores[t-1]

        model_scores[t] = (Y[t]/exp(theta[t])) - 1.0
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llt_recursion_t_orderone(double[:] parameters, double[:] theta, double[:] theta_t, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta_t[t-1] + theta[t-1] + parameters[0]*model_scores[t-1]
            theta_t[t] = theta_t[t-1] + parameters[1]*model_scores[t-1]

        model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(Y[t]-theta[t],2)/shape))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llt_recursion_t_ordertwo(double[:] parameters, double[:] theta, double[:] theta_t, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta_t[t-1] + theta[t-1] + parameters[0]*model_scores[t-1]
            theta_t[t] = theta_t[t-1] + parameters[1]*model_scores[t-1]

        model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(Y[t]-theta[t],2)/shape))/((shape+1)*((np.power(scale,2)*shape) - np.power(Y[t]-theta[t],2))/np.power((np.power(scale,2)*shape) + np.power(Y[t]-theta[t],2),2))
    
    return theta, model_scores



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llt_recursion_cauchy_orderone(double[:] parameters, double[:] theta, double[:] theta_t, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta_t[t-1] + theta[t-1] + parameters[0]*model_scores[t-1]
            theta_t[t] = theta_t[t-1] + parameters[1]*model_scores[t-1]

        model_scores[t] = 2.0*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(Y[t]-theta[t],2)))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llt_recursion_cauchy_ordertwo(double[:] parameters, double[:] theta, double[:] theta_t, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta_t[t-1] + theta[t-1] + parameters[0]*model_scores[t-1]
            theta_t[t] = theta_t[t-1] + parameters[1]*model_scores[t-1]

        model_scores[t] = 2.0*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(Y[t]-theta[t],2)))/((2.0)*((np.power(scale,2)) - np.power(Y[t]-theta[t],2))/np.power((np.power(scale,2)) + np.power(Y[t]-theta[t],2),2))
    
    return theta, model_scores


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llt_recursion_skewt_orderone(double[:] parameters, double[:] theta, double[:] theta_t, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta_t[t-1] + theta[t-1] + parameters[0]*model_scores[t-1]
            theta_t[t] = theta_t[t-1] + parameters[1]*model_scores[t-1]

        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(M_PI)*sp.gamma(shape/2.0))
        mean = theta[t] + (skewness - (1.0/skewness))*scale*m1
        if (Y[t]-theta[t])>=0:
            model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(skewness*scale,2) + (np.power(Y[t]-theta[t],2)/shape))
        else:
            model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(skewness*(Y[t]-theta[t]),2)/shape))
    
    return theta, model_scores

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_llt_recursion_skewt_ordertwo(double[:] parameters, double[:] theta, double[:] theta_t, double[:] model_scores, double[:] Y, int Y_len, double scale, double shape, double skewness, int max_lag):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        if t < max_lag:
            theta[t] = 0.0
        else:
            theta[t] = theta_t[t-1] + theta[t-1] + parameters[0]*model_scores[t-1]
            theta_t[t] = theta_t[t-1] + parameters[1]*model_scores[t-1]

        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(M_PI)*sp.gamma(shape/2.0))
        mean = theta[t] + (skewness - (1.0/skewness))*scale*m1
        if (Y[t]-theta[t])>=0:
            model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(skewness*scale,2) + (np.power(Y[t]-theta[t],2)/shape))
        else:
            model_scores[t] = ((shape+1)/shape)*(Y[t]-theta[t])/(np.power(scale,2) + (np.power(skewness*(Y[t]-theta[t]),2)/shape))
    
    return theta, model_scores



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_reg_recursion(double[:] parameters, double[:] theta, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] coefficients, np.ndarray[double,ndim=2] model_scores, 
    double[:] Y, int Y_len, score_function, link, double scale, double shape, double skewness):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        theta[t] = np.dot(X[t],coefficients[:,t])
        model_scores[:,t] = score_function(X[t],Y[t],link(theta[t]),scale,shape,skewness)
        coefficients[:,t+1] = coefficients[:,t] + parameters[0:X.shape[1]]*model_scores[:,t] 

    return theta, model_scores, coefficients

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_reg_recursion_exponential_orderone(double[:] parameters, double[:] theta, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] coefficients, np.ndarray[double,ndim=2] model_scores, 
    double[:] Y, int Y_len, double scale, double shape, double skewness):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        theta[t] = np.dot(X[t],coefficients[:,t])
        model_scores[:,t] = X[t]*(1.0 - exp(theta[t])*Y[t])
        coefficients[:,t+1] = coefficients[:,t] + parameters[0:X.shape[1]]*model_scores[:,t] 

    return theta, model_scores, coefficients

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_reg_recursion_exponential_ordertwo(double[:] parameters, double[:] theta, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] coefficients, np.ndarray[double,ndim=2] model_scores, 
    double[:] Y, int Y_len, double scale, double shape, double skewness):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        theta[t] = np.dot(X[t],coefficients[:,t])
        model_scores[:,t] = X[t]*(1.0 - exp(theta[t])*Y[t])
        coefficients[:,t+1] = coefficients[:,t] + parameters[0:X.shape[1]]*model_scores[:,t] 

    return theta, model_scores, coefficients

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_reg_recursion_laplace_orderone(double[:] parameters, double[:] theta, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] coefficients, np.ndarray[double,ndim=2] model_scores, 
    double[:] Y, int Y_len, double scale, double shape, double skewness):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        theta[t] = np.dot(X[t],coefficients[:,t])
        model_scores[:,t] = X[t]*(Y[t]-theta[t])/(scale*abs(Y[t]-theta[t]))
        coefficients[:,t+1] = coefficients[:,t] + parameters[0:X.shape[1]]*model_scores[:,t] 

    return theta, model_scores, coefficients

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_reg_recursion_laplace_ordertwo(double[:] parameters, double[:] theta, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] coefficients, np.ndarray[double,ndim=2] model_scores, 
    double[:] Y, int Y_len, double scale, double shape, double skewness):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        theta[t] = np.dot(X[t],coefficients[:,t])
        model_scores[:,t] = X[t]*(Y[t]-theta[t])/(scale*abs(Y[t]-theta[t]))
        coefficients[:,t+1] = coefficients[:,t] + parameters[0:X.shape[1]]*model_scores[:,t] 

    return theta, model_scores, coefficients

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_reg_recursion_normal_orderone(double[:] parameters, double[:] theta, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] coefficients, np.ndarray[double,ndim=2] model_scores, 
    double[:] Y, int Y_len, double scale, double shape, double skewness):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        theta[t] = np.dot(X[t],coefficients[:,t])
        model_scores[:,t] = X[t]*(Y[t] - theta[t])
        coefficients[:,t+1] = coefficients[:,t] + parameters[0:X.shape[1]]*model_scores[:,t] 

    return theta, model_scores, coefficients

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_reg_recursion_normal_ordertwo(double[:] parameters, double[:] theta, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] coefficients, np.ndarray[double,ndim=2] model_scores, 
    double[:] Y, int Y_len, double scale, double shape, double skewness):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        theta[t] = np.dot(X[t],coefficients[:,t])
        model_scores[:,t] = X[t]*(Y[t] - theta[t])
        coefficients[:,t+1] = coefficients[:,t] + parameters[0:X.shape[1]]*model_scores[:,t] 

    return theta, model_scores, coefficients

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_reg_recursion_poisson_orderone(double[:] parameters, double[:] theta, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] coefficients, np.ndarray[double,ndim=2] model_scores, 
    double[:] Y, int Y_len, double scale, double shape, double skewness):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        theta[t] = np.dot(X[t],coefficients[:,t])
        model_scores[:,t] = X[t]*(Y[t] - exp(theta[t]))
        coefficients[:,t+1] = coefficients[:,t] + parameters[0:X.shape[1]]*model_scores[:,t] 

    return theta, model_scores, coefficients

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_reg_recursion_poisson_ordertwo(double[:] parameters, double[:] theta, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] coefficients, np.ndarray[double,ndim=2] model_scores, 
    double[:] Y, int Y_len, double scale, double shape, double skewness):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        theta[t] = np.dot(X[t],coefficients[:,t])
        model_scores[:,t] = X[t]*(Y[t]/exp(theta[t]) - 1.0)
        coefficients[:,t+1] = coefficients[:,t] + parameters[0:X.shape[1]]*model_scores[:,t] 

    return theta, model_scores, coefficients

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_reg_recursion_t_orderone(double[:] parameters, double[:] theta, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] coefficients, np.ndarray[double,ndim=2] model_scores, 
    double[:] Y, int Y_len, double scale, double shape, double skewness):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        theta[t] = np.dot(X[t],coefficients[:,t])
        model_scores[:,t] = ((shape+1)/shape)*((Y[t]-theta[t])*X[t])/(np.power(scale,2)+np.power((Y[t]-theta[t]),2)/shape)
        coefficients[:,t+1] = coefficients[:,t] + parameters[0:X.shape[1]]*model_scores[:,t] 

    return theta, model_scores, coefficients

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_reg_recursion_t_ordertwo(double[:] parameters, double[:] theta, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] coefficients, np.ndarray[double,ndim=2] model_scores, 
    double[:] Y, int Y_len, double scale, double shape, double skewness):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        theta[t] = np.dot(X[t],coefficients[:,t])
        model_scores[:,t] = ((shape+1)/shape)*((Y[t]-theta[t])*X[t])/(np.power(scale,2)+np.power((Y[t]-theta[t]),2)/shape)
        coefficients[:,t+1] = coefficients[:,t] + parameters[0:X.shape[1]]*model_scores[:,t] 

    return theta, model_scores, coefficients



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_reg_recursion_cauchy_orderone(double[:] parameters, double[:] theta, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] coefficients, np.ndarray[double,ndim=2] model_scores, 
    double[:] Y, int Y_len, double scale, double shape, double skewness):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        theta[t] = np.dot(X[t],coefficients[:,t])
        model_scores[:,t] = 2.0*((Y[t]-theta[t])*X[t])/(np.power(scale,2)+np.power((Y[t]-theta[t]),2))
        coefficients[:,t+1] = coefficients[:,t] + parameters[0:X.shape[1]]*model_scores[:,t] 

    return theta, model_scores, coefficients

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_reg_recursion_cauchy_ordertwo(double[:] parameters, double[:] theta, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] coefficients, np.ndarray[double,ndim=2] model_scores, 
    double[:] Y, int Y_len, double scale, double shape, double skewness):

    cdef Py_ssize_t t

    for t in range(0,Y_len):
        theta[t] = np.dot(X[t],coefficients[:,t])
        model_scores[:,t] = 2.0*((Y[t]-theta[t])*X[t])/(np.power(scale,2)+np.power((Y[t]-theta[t]),2))
        coefficients[:,t+1] = coefficients[:,t] + parameters[0:X.shape[1]]*model_scores[:,t] 

    return theta, model_scores, coefficients




@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_reg_recursion_skewt_orderone(double[:] parameters, double[:] theta, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] coefficients, np.ndarray[double,ndim=2] model_scores, 
    double[:] Y, int Y_len, double scale, double shape, double skewness):

    cdef Py_ssize_t t

    for t in range(0, Y_len):
        theta[t] = np.dot(X[t],coefficients[:,t])

        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(M_PI)*sp.gamma(shape/2.0))
        mean = theta[t] + (skewness - (1.0/skewness))*scale*m1
        if (Y[t]-theta[t])>=0:
            model_scores[:,t] = ((shape+1)/shape)*(X[t]*(Y[t]-theta[t]))/(np.power(skewness*scale,2) + (np.power(Y[t]-theta[t],2)/shape))
        else:
            model_scores[:,t] = ((shape+1)/shape)*(X[t]*(Y[t]-theta[t]))/(np.power(scale,2) + (np.power(skewness*(Y[t]-theta[t]),2)/shape))

        coefficients[:,t+1] = coefficients[:,t] + parameters[0:X.shape[1]]*model_scores[:,t] 

    return theta, model_scores, coefficients

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def gas_reg_recursion_skewt_ordertwo(double[:] parameters, double[:] theta, np.ndarray[double,ndim=2] X, np.ndarray[double,ndim=2] coefficients, np.ndarray[double,ndim=2] model_scores, 
    double[:] Y, int Y_len, double scale, double shape, double skewness):

    cdef Py_ssize_t t

    for t in range(0, Y_len):
        theta[t] = np.dot(X[t],coefficients[:,t])

        m1 = (np.sqrt(shape)*sp.gamma((shape-1.0)/2.0))/(np.sqrt(M_PI)*sp.gamma(shape/2.0))
        mean = theta[t] + (skewness - (1.0/skewness))*scale*m1
        if (Y[t]-theta[t])>=0:
            model_scores[:,t] = ((shape+1)/shape)*(X[t]*(Y[t]-theta[t]))/(np.power(skewness*scale,2) + (np.power(Y[t]-theta[t],2)/shape))
        else:
            model_scores[:,t] = ((shape+1)/shape)*(X[t]*(Y[t]-theta[t]))/(np.power(scale,2) + (np.power(skewness*(Y[t]-theta[t]),2)/shape))

        coefficients[:,t+1] = coefficients[:,t] + parameters[0:X.shape[1]]*model_scores[:,t] 

    return theta, model_scores, coefficients