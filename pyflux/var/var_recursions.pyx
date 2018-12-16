import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def create_design_matrix(np.ndarray[double,ndim=2] Z, np.ndarray[double,ndim=2] data, int Y_len,
                         np.ndarray[int,ndim=1] ar_idx_list, int max_lag):

    cdef Py_ssize_t row_count, lag, reg

    row_count = 1

    for lag in range(ar_idx_list):
        for reg in range(Y_len):
            Z[row_count, :] = data[reg][(max_lag-ar_idx_list[lag]):-ar_idx_list[lag]]
            row_count += 1

    return Z

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def custom_covariance_matrix(np.ndarray[double,ndim=2] cov_matrix, int Y_len,
                             int used_lags, np.ndarray[double,ndim=1] parm):

    cdef Py_ssize_t quick_count, i, k, index

    quick_count = 0
    
    for i in range(0,Y_len):
        for k in range(0,Y_len):
            if i >= k:
                index = Y_len + used_lags*(Y_len**2) + quick_count
                quick_count += 1
                cov_matrix[i,k] = parm[index]

    return np.dot(cov_matrix,cov_matrix.T)

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def var_likelihood(float ll1, int mu_shape, np.ndarray[double,ndim=2] diff, np.ndarray[double,ndim=2] inverse):

    cdef Py_ssize_t t
    cdef float ll2

    ll2 = 0

    for t in range(0,mu_shape):
        ll2 += np.dot(np.dot(diff[t].T,inverse),diff[t])

    return -(ll1 -0.5*ll2)
