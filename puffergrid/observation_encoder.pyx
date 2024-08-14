# distutils: language=c++

from libcpp.vector cimport vector
from libcpp.string cimport string
from puffergrid.grid_object cimport GridObject

cdef class ObservationEncoder:
    cdef encode(self, const GridObject *obj, int[:] obs):
        pass

    cdef vector[string] feature_names(self):
        return vector[string]()
