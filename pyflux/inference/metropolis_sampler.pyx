import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def metropolis_sampler(int sims_to_do, np.ndarray[double,ndim=2] phi, 
    posterior, np.ndarray[double,ndim=2] a_rate, np.ndarray[double,ndim=2] rnums, 
    np.ndarray[double,ndim=2] crit):

    cdef Py_ssize_t i
    cdef float post_prop, lik_rate, old_lik
    cdef np.ndarray[double, ndim=1, mode="c"] phi_prop 

    old_lik = -posterior(phi[0]) # Initial posterior

    # Sampling time!
    for i in range(1,sims_to_do):
        phi_prop = phi[i-1] + rnums[i]
        post_prop = -posterior(phi_prop)
        lik_rat = np.exp(post_prop - old_lik)

        if crit[i] < lik_rat:
            phi[i] = phi_prop
            a_rate[i] = 1
            old_lik = post_prop
        else:
            phi[i] = phi[i-1]

    return phi, a_rate
