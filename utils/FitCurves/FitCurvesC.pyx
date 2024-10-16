# cython: language_level=3

## Compile here, in-place, with:
# cythonize -i FitCurvesC.pyx

cdef extern from "FitCurves.c":
    ctypedef struct Point2:
        double x
        double y
    ctypedef Point2 Vector2
    ctypedef Point2 *BezierCurve
    
    void FitCurve(Point2 *d, int nPts, double error)
    
    cdef void (*DrawBezierCurve)(int n, BezierCurve curve)

cdef extern from "GGVecLib.c":
    pass

cdef void MyDrawBezierCurve(int n, BezierCurve curve) noexcept with gil:
    assert curve_in_progress is not None
    
    bezier = []
    for i in range( n+1 ):
        bezier.append( ( curve[i].x, curve[i].y ) )
    
    curve_in_progress.append( bezier )

DrawBezierCurve = MyDrawBezierCurve

import numpy
import ctypes

curve_in_progress = None
def fitCurve( vertices, error = 1e-1 ):
    '''
    Given an N-by-2 numpy array 'vertices' of 2D vertices representing a line strip,
    returns an N-by-4-by-2 numpy.array of N cubic bezier curves approximating 'vertices'.
    The first and last points are preserved.
    '''
    
    global curve_in_progress
    assert curve_in_progress is None
    
    ## Make sure the input values have their data in a way easy to access from C.
    vertices = numpy.ascontiguousarray( numpy.asarray( vertices, dtype = ctypes.c_double ) )
    
    ## 'vertices' must be 2D
    assert vertices.shape[1] == 2
    
    cdef double[:, ::1] vertices_memview = vertices
    
    ## This calls a callback function that appends to the global variable 'curve_in_progress'.
    curve_in_progress = []
    FitCurve(
        <Point2*>&vertices_memview[0,0],
        len( vertices ),
        error
        )
    result = numpy.asarray( curve_in_progress )
    curve_in_progress = None
    
    return result

def test_simple( N = 10 ):
    print( 'test_simple( %d )' % N )
    
    from pprint import pprint
    
    assert N > 1
    
    line_strip = numpy.zeros( ( N, 2 ) )
    line_strip[:,0] = numpy.linspace( 0, 1, N )
    line_strip[:,1] = numpy.linspace( -1, 1, N )
    pprint( line_strip )
    
    beziers = fitCurve( line_strip )
    pprint( beziers )

def main():
    import sys
    
    N = 10
    if len( sys.argv ) > 1: N = int( sys.argv[1] )
    
    test_simple( N )

if __name__ == '__main__': main()
