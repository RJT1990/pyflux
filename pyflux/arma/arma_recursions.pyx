import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def arima_recursion(double[:] parameters, double[:] mu, double[:] Y, 
	int max_lag, int Y_len, int ar_terms, int ma_terms):
	cdef Py_ssize_t t, k

	for t in range(max_lag, Y_len):
		for k in range(0, ma_terms):
			mu[t] += parameters[1+ar_terms+k]*(Y[t-1-k]-mu[t-1-k])

	return mu

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def arimax_recursion(double[:] parameters, double[:] mu, double[:] Y, 
	int max_lag, int Y_len, int ar_terms, int ma_terms):
	cdef Py_ssize_t t, k

	for t in range(max_lag, Y_len):
		for k in range(0, ma_terms):
			mu[t] += parameters[ar_terms+k]*(Y[t-1-k]-mu[t-1-k])

	return mu