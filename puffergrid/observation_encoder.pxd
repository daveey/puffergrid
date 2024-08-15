# distutils: language=c++

from libcpp.vector cimport vector
from libcpp.string cimport string

from puffergrid.grid_object cimport GridObject

ctypedef int ObsType

cdef class ObservationEncoder:
    cdef encode(self, const GridObject *obj, ObsType[:] obs)
    cdef vector[string] feature_names(self)
