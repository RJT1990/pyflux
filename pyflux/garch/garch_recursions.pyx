import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def garch_recursion(double[:] parameters, double[:] sigma2, int q_terms, int p_terms, int Y_len, int max_lag):

    cdef Py_ssize_t t, k

    if p_terms != 0:
        for t in range(0,Y_len):
            if t < max_lag:
                sigma2[t] = parameters[0]/(1-np.sum(parameters[(q_terms+1):(q_terms+p_terms+1)]))
            elif t >= max_lag:
                for k in range(0,p_terms):
                    sigma2[t] += parameters[1+q_terms+k]*(sigma2[t-1-k])

    return sigma2