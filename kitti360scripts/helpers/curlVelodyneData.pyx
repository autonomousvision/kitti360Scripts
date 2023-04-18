# cython methods to speed-up curling

import numpy as np
cimport cython
cimport numpy as np
import ctypes

np.import_array()

cdef extern from "curlVelodyneData_impl.c":
        void curl_velodyne_data (const double* velo_in,
                                 double*       velo_out,
                                 const unsigned int pt_num,
                                 const double* r,
                                 const double* t)


cdef tonumpyarray(double* data, unsigned long long size, unsigned long long dim):
    if not (data and size >= 0 and dim >= 0): 
            raise ValueError
    return np.PyArray_SimpleNewFromData(2, [size, dim], np.NPY_FLOAT64, <void*>data)

@cython.boundscheck(False)
def cCurlVelodyneData( np.ndarray[np.float64_t , ndim=2] veloIn   ,
                       np.ndarray[np.float64_t , ndim=2] veloOut  ,
                       np.ndarray[np.float64_t , ndim=1] r,
                       np.ndarray[np.float64_t , ndim=1] t):
    cdef np.ndarray[np.float64_t     , ndim=2, mode="c"] veloIn_c
    cdef np.ndarray[np.float64_t     , ndim=2, mode="c"] veloOut_c
    cdef np.ndarray[np.float64_t     , ndim=1, mode="c"] r_c
    cdef np.ndarray[np.float64_t     , ndim=1, mode="c"] t_c
    
    veloIn_c  = np.ascontiguousarray(veloIn , dtype=np.float64  )
    veloOut_c = np.ascontiguousarray(veloOut, dtype=np.float64  )
    r_c       = np.ascontiguousarray(r      , dtype=np.float64  )
    t_c       = np.ascontiguousarray(t      , dtype=np.float64  )
    
    cdef np.uint32_t pt_num   = veloIn.shape[0]
    cdef np.uint32_t pt_dim   = veloIn.shape[1]
    
    curl_velodyne_data(&veloIn_c[0,0], &veloOut_c[0,0], pt_num, &r_c[0], &t_c[0])
    
    veloOut = np.ascontiguousarray(tonumpyarray(&veloOut_c[0,0], pt_num, pt_dim))
    
    return np.copy(veloOut)
