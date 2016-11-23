import numpy as np
cimport numpy as np
cimport cython

cdef inline double double_max(double a, double b): return a if a >= b else b
cdef inline double double_min(double a, double b): return a if a <= b else b

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def arima_recursion(double[:] parameters, double[:] mu, double[:] link_mu, double[:] Y, 
	int max_lag, int Y_len, int ar_terms, int ma_terms):
	"""
	Cythonized moving average recursion for ARIMA model class
	"""
	cdef Py_ssize_t t, k

	for t in range(max_lag, Y_len):
		for k in range(0, ma_terms):
			mu[t] += parameters[1+ar_terms+k]*(Y[t-1-k]-link_mu[t-1-k])

	return mu

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def arima_recursion_poisson(double[:] parameters, double[:] mu, double[:] link_mu, double[:] Y, 
	int max_lag, int Y_len, int ar_terms, int ma_terms):
	"""
	Cythonized moving average recursion for ARIMA model class - Poisson
	"""
	cdef Py_ssize_t t, k

	for t in range(max_lag, Y_len):
		for k in range(0, ma_terms):
			mu[t] += parameters[1+ar_terms+k]*(double_min(double_max(Y[t-1-k]-link_mu[t-1-k],-1000),1000))

	return mu

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def arima_recursion_normal(double[:] parameters, double[:] mu, double[:] Y, 
	int max_lag, int Y_len, int ar_terms, int ma_terms):
	"""
	Cythonized moving average recursion for ARIMA model class - Gaussian errors
	"""
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
	"""
	Cythonized moving average recursion for ARIMAX model class
	"""
	cdef Py_ssize_t t, k

	for t in range(max_lag, Y_len):
		for k in range(0, ma_terms):
			mu[t] += parameters[ar_terms+k]*(Y[t-1-k]-mu[t-1-k])

	return mu