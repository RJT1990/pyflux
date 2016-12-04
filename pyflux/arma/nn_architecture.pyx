import numpy as np
cimport numpy as np
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def neural_network_tanh(double[:] Y, np.ndarray[double,ndim=2,mode="c"] X, double[:] z, int units, int layers, int ar):
    """
    Cythonized neural network
    """
    cdef Py_ssize_t unit, layer
    cdef int params_used

    # Input layer
    cdef np.ndarray[double,ndim=2,mode="c"] first_layer_output = np.zeros((X.shape[1], units))
    for unit in range(units):
        first_layer_output[:,unit] = np.tanh(np.matmul(np.transpose(X), z[unit*(ar+1):((unit+1)*(ar+1))]))

        params_used = ((units)*(ar+1))

        hidden_layer_output = np.zeros((X.shape[1], units, layers-1))
        for layer in range(1, layers):
            for unit in range(units):
                if layer == 1:
                    hidden_layer_output[:, unit,layer-1] = np.tanh(np.matmul(first_layer_output,
                        z[params_used+unit*(units)+((layer-1)*units**2):((params_used+(unit+1)*units)+((layer-1)*units**2))]))
                else:
                    hidden_layer_output[:, unit,layer-1] = np.tanh(np.matmul(hidden_layer_output[:,:,layer-1],
                        z[params_used+unit*(units)+((layer-1)*units**2):((params_used+(unit+1)*units)+((layer-1)*units**2))]))

        params_used = params_used + (layers-1)*units**2

        # Output layer
        if layers == 1:
            mu = np.matmul(first_layer_output, z[params_used:params_used+units])
        else:
            mu = np.matmul(hidden_layer_output[:,:,-1], z[params_used:params_used+units])

    return mu

@cython.boundscheck(False)
@cython.wraparound(False)
@cython.cdivision(True)
def neural_network_tanh_mb(double[:] Y, np.ndarray[double, ndim=2] X, double[:] z, int units, int layers, int ar):
    """
    Cythonized neural network with minibatch
    """
    cdef Py_ssize_t unit, layer
    cdef int params_used

    # Input layer
    cdef np.ndarray[double,ndim=2,mode="c"] first_layer_output = np.zeros((X.shape[1], units))
    for unit in range(units):
        first_layer_output[:,unit] = np.tanh(np.matmul(np.transpose(X), z[unit*(ar+1):((unit+1)*(ar+1))]))

        params_used = ((units)*(ar+1))

        hidden_layer_output = np.zeros((X.shape[1], units, layers-1))
        for layer in range(1, layers):
            for unit in range(units):
                if layer == 1:
                    hidden_layer_output[:, unit,layer-1] = np.tanh(np.matmul(first_layer_output,
                        z[params_used+unit*(units)+((layer-1)*units**2):((params_used+(unit+1)*units)+((layer-1)*units**2))]))
                else:
                    hidden_layer_output[:, unit,layer-1] = np.tanh(np.matmul(hidden_layer_output[:,:,layer-1],
                        z[params_used+unit*(units)+((layer-1)*units**2):((params_used+(unit+1)*units)+((layer-1)*units**2))]))

        params_used = params_used + (layers-1)*units**2

        # Output layer
        if layers == 1:
            mu = np.matmul(first_layer_output, z[params_used:params_used+units])
        else:
            mu = np.matmul(hidden_layer_output[:,:,-1], z[params_used:params_used+units])

    return mu