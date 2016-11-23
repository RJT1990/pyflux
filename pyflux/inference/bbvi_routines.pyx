import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def alpha_recursion(double[:] alpha0, np.ndarray[double,ndim=2] grad_log_q, np.ndarray[double,ndim=2] gradient, int param_no):

    cdef Py_ssize_t lambda_i

    for lambda_i in range(param_no):
         alpha0[lambda_i] = np.cov(grad_log_q[lambda_i],gradient[lambda_i])[0][1]  

    return alpha0

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def log_p_posterior(np.ndarray[double,ndim=2] z, neg_posterior):

    cdef Py_ssize_t i
    cdef np.ndarray[double, ndim=1, mode="c"] result = np.zeros(z.shape[0], dtype=np.float64) 

    for i in range(z.shape[0]):
         result[i] = -neg_posterior(z[i])

    return result

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def mb_log_p_posterior(np.ndarray[double,ndim=2] z, neg_posterior, int mini_batch):

    cdef Py_ssize_t i
    cdef np.ndarray[double, ndim=1, mode="c"] result = np.zeros(z.shape[0], dtype=np.float64) 

    for i in range(z.shape[0]):
         result[i] = -neg_posterior(z[i], mini_batch)

    return result