from numpy import *
Real = float64

## The smooth min distance constant, the "infinity" in the L_infinity norm.
## Denis tells me that infinity equals 4 ("in some cases it is closer to 5 or even 6 under most unusual
## circumstances. Never exceeded 8 in civilian applications.").
## 4 works, but is maybe low.
L_infinity = 3

def distancesSqr_to_edges( pts, edges ):
    '''
    Input parameter 'pts' has dimensions (x,y,...) x #pts.
    Input parameter 'edges' has dimensions ... x #edges x 2 endpoints x N coordinates (x,y,...).
    Returns an array of distances squared with dimensions #edges x #pts.
    '''
    
    pts = asarray( pts, Real )
    ## pts has dimensions N (x,y,...) x #pts
    edges = asarray( edges, Real )
    ## edges has dimensions ... x #edges x 2 endpoints x N coordinates (x,y,...)
    
    N = pts.shape[0]
    
    assert len( pts.shape ) == 2 and pts.shape[0] == N
    assert edges.shape[-2] == 2 and edges.shape[-1] == N
    #print ('pts.shape:', pts.shape)
    #print ('edges.shape:', edges.shape)
    
    
    ## get distance squared to each edge:
    ##   let p = black_pixel_pos, a = endpoint0, b = endpoint1, d = ( b-a ) / dot( b-a,b-a )
    ##   dot( p-a, d ) < 0 => dot( p-a, p-a )
    ##   dot( p-a, d ) > 1 => dot( p-b, p-b )
    ##   else              => dot( dot( p-a, d ) * (b-a) - p, same )
    p_a = pts[newaxis,...] - edges[...,:,0,:,newaxis]
    p_b = pts[newaxis,...] - edges[...,:,1,:,newaxis]
    ## p_a and p_b have dimensions ... x #edges x N coordinates (x,y,...) x #pts
    b_a = edges[...,:,1,:] - edges[...,:,0,:]
    ## b_a has dimensions ... x #edges x N coordinates (x,y,...)
    d = b_a / ( b_a**2 ).sum( -1 )[...,newaxis]
    ## d has same dimensions as b_a
    assert b_a.shape == d.shape
    cond = ( p_a * d[...,newaxis] ).sum( -2 )
    ## cond has dimensions ... x #edges x #pts
    assert cond.shape[-2:] == (edges.shape[-3], pts.shape[-1])
    
    cond_lt_zero = cond < 0
    cond_gt_one = cond > 1
    cond_else = logical_not( logical_or( cond_lt_zero, cond_gt_one ) )
    ## cond_* have dimensions ... x #edges x #pts
    
    #distancesSqr = empty( cond.shape, Real )
    ## distancesSqr has dimensions ... x #edges x #pts
    #assert distancesSqr.shape[-2:] == (edges.shape[-3], pts.shape[-1])
    
    ## else case
    # distancesSqr = p_a - cond[:,newaxis,:] * b_a[...,newaxis]
    # distancesSqr = ( distancesSqr**2 ).sum( 1 )
    # <=>
    # distancesSqr = ( ( p_a - cond[:,newaxis,:] * b_a[...,newaxis] )**2 ).sum( -2 )
    # <=>
    # distancesSqr[ cond_else ] = ( ( p_a - cond[:,newaxis,:] * b_a[...,newaxis] )**2 ).sum( -2 )[ cond_else ]
    distancesSqr = ( ( p_a - cond[:,newaxis,:] * b_a[...,newaxis] )**2 ).sum( -2 )
    # <=>
    # distancesSqr[ cond_else ] = ( (
    #    swapaxes( p_a, -1, -2 )[ cond_else ] - swapaxes( cond[:,newaxis,:] * b_a[...,newaxis], -1, -2 )[ cond_else ]
    #    )**2 ).sum( -1 )
    
    ## < 0 case
    # distancesSqr[ cond < 0 ] = ( p_a**2 ).sum( -2 )[ cond < 0 ]
    # <=>
    distancesSqr[ cond_lt_zero ] = ( p_a**2 ).sum( -2 )[ cond_lt_zero ]
    # <=>
    # distancesSqr[ cond_lt_zero ] = ( swapaxes( p_a, -1, -2 )[ cond_lt_zero ]**2 ).sum( -1 )
    
    ## > 1 case
    # distancesSqr[ cond > 1 ] = ( p_b**2 ).sum( -2 )[ cond > 1 ]
    # <=>
    distancesSqr[ cond_gt_one ] = ( p_b**2 ).sum( -2 )[ cond_gt_one ]
    # <=>
    # distancesSqr[ cond_gt_one ] = ( swapaxes( p_b, -1, -2 )[ cond_gt_one ]**2 ).sum( -1 )
    
    #print 'distancesSqr:', distancesSqr
    #print 'distances:', sqrt( distancesSqr.min(0) )
    
    return distancesSqr

def min_distance_to_edges( pts, edges ):
    '''
    Input parameter 'pts' has dimensions 2 (x,y) x #pts.
    Input parameter 'edges' has dimensions ... x #edges x 2 endpoints x 2 coordinates (x,y).
    Returns an array of the minimum distance to edges with dimensions #pts.
    '''
    return sqrt( distancesSqr_to_edges( pts, edges ).min( -2 ) )

def smooth_min_distance_to_edges( pts, edges ):
    '''
    Input parameter 'pts' has dimensions 2 (x,y) x #pts.
    Input parameter 'edges' has dimensions ... x #edges x 2 endpoints x 2 coordinates (x,y).
    Returns an array of the smooth minimum distance to edges with dimensions #pts.
    '''
    
    distancesSqr = distancesSqr_to_edges( pts, edges )
    ## distancesSqr has dimensions ... x #edges x #pts
    
    ## Take the L_infinity norm of all distances from a point to the edges.
    ## We want the min, not the max, so we negate the exponent to get 1/( \sum 1/dist_i ).
    ## We also need to take the square root of distancesSqr, which we'll fold into our exponent.
    distances = sum( distancesSqr**(-L_infinity/2.), -2 )**(-1./L_infinity)
    ## distances has dimensions ... x #pts
    return distances


def distances_and_pt_gradients_to_edges( pts, edges ):
    '''
    Input parameter 'pts' has dimensions (x,y,...) x #pts.
    Input parameter 'edges' has dimensions ... x #edges x 2 endpoints x N coordinates (x,y,...).
    Returns a tuple, where the first element is
    an array of distances with dimensions #edges x #pts
    and the second element is
    an array of gradients with respect to 'pts' with dimensions #edges x N coordinates (x,y,...) x #pts.
    '''
    
    pts = asarray( pts, Real )
    ## pts has dimensions N (x,y,...) x #pts
    edges = asarray( edges, Real )
    ## edges has dimensions ... x #edges x 2 endpoints x N coordinates (x,y,...)
    
    N = pts.shape[0]
    
    assert len( pts.shape ) == 2 and pts.shape[0] == N
    assert edges.shape[-2] == 2 and edges.shape[-1] == N
    
    ## get distance squared to each edge:
    ##   let p = black_pixel_pos, a = endpoint0, b = endpoint1, d = ( b-a ) / dot( b-a,b-a )
    ##   dot( p-a, d ) < 0 => dot( p-a, p-a )
    ##   dot( p-a, d ) > 1 => dot( p-b, p-b )
    ##   else              => dot( dot( p-a, d ) * (b-a) - p, same )
    p_a = pts[newaxis,...] - edges[...,:,0,:,newaxis]
    p_b = pts[newaxis,...] - edges[...,:,1,:,newaxis]
    ## p_a and p_b have dimensions ... x #edges x N coordinates (x,y,...) x #pts
    b_a = edges[...,:,1,:] - edges[...,:,0,:]
    ## b_a has dimensions ... x #edges x N coordinates (x,y,...)
    d = b_a / ( b_a**2 ).sum( -1 )[...,newaxis]
    ## d has same dimensions as b_a
    assert b_a.shape == d.shape
    cond = ( p_a * d[...,newaxis] ).sum( -2 )
    ## cond has dimensions ... x #edges x #pts
    assert cond.shape[-2:] == (edges.shape[-3], pts.shape[-1])
    
    cond_lt_zero = cond < 0
    cond_gt_one = cond > 1
    cond_else = logical_not( logical_or( cond_lt_zero, cond_gt_one ) )
    ## cond_* have dimensions ... x #edges x #pts
    
    #distancesSqr = empty( cond.shape, Real )
    ## distancesSqr has dimensions ... x #edges x #pts
    #assert distancesSqr.shape[-2:] == (edges.shape[-3], pts.shape[-1])
    
    ## else case
    # distancesSqr = p_a - cond[:,newaxis,:] * b_a[...,newaxis]
    # distancesSqr = ( distancesSqr**2 ).sum( 1 )
    # <=>
    # distancesSqr = ( ( p_a - cond[:,newaxis,:] * b_a[...,newaxis] )**2 ).sum( -2 )
    # <=>
    # distancesSqr[ cond_else ] = ( ( p_a - cond[:,newaxis,:] * b_a[...,newaxis] )**2 ).sum( -2 )[ cond_else ]
    distancesSqr = ( ( p_a - cond[:,newaxis,:] * b_a[...,newaxis] )**2 ).sum( -2 )
    # <=>
    # distancesSqr[ cond_else ] = ( (
    #    swapaxes( p_a, -1, -2 )[ cond_else ] - swapaxes( cond[:,newaxis,:] * b_a[...,newaxis], -1, -2 )[ cond_else ]
    #    )**2 ).sum( -1 )
    
    ## < 0 case
    # distancesSqr[ cond < 0 ] = ( p_a**2 ).sum( -2 )[ cond < 0 ]
    # <=>
    distancesSqr[ cond_lt_zero ] = ( p_a**2 ).sum( -2 )[ cond_lt_zero ]
    # <=>
    # distancesSqr[ cond_lt_zero ] = ( swapaxes( p_a, -1, -2 )[ cond_lt_zero ]**2 ).sum( -1 )
    
    ## > 1 case
    # distancesSqr[ cond > 1 ] = ( p_b**2 ).sum( -2 )[ cond > 1 ]
    # <=>
    distancesSqr[ cond_gt_one ] = ( p_b**2 ).sum( -2 )[ cond_gt_one ]
    # <=>
    # distancesSqr[ cond_gt_one ] = ( swapaxes( p_b, -1, -2 )[ cond_gt_one ]**2 ).sum( -1 )
    
    #print 'distancesSqr:', distancesSqr
    #print 'distances:', sqrt( distancesSqr.min(0) )
    
    ## distancesSqr is now distances
    sqrt( distancesSqr, distancesSqr )
    ## we'll just rename distancesSqr
    distances = distancesSqr
    ## distances has dimensions ... x #edges x #pts
    del distancesSqr
    
    #set_trace()
    
    #distances_grad = empty( p_a.shape, Real )
    ## distances_grad has dimensions ... x #edges x N coordinates (x,y,...) x #pts
    
    ## else case
    #cond_else = cond_else[...,newaxis,:].repeat( 2, -2 )
    ## L is the vector to pts from the closest points on the edges
    L = p_a - cond[:,newaxis,:] * b_a[...,newaxis]
    ## L has dimensions ... x #edges x N coordinates (x,y,...) x #pts
    cond_else = cond_else[...,newaxis,:].repeat( 2, -2 )
    #distances_grad[ cond_else ] = L[ cond_else ] - ( ( L * d[...,newaxis] ).sum( -2 )[...,:,newaxis,:] * b_a[...,newaxis] )[ cond_else ]
    #distances_grad[ cond_else ] = ( L - ( L * d[...,newaxis] ).sum( -2 )[...,:,newaxis,:] * b_a[...,newaxis] )[ cond_else ]
    distances_grad = L - ( ( L * d[...,newaxis] ).sum( -2 )[...,:,newaxis,:] * b_a[...,newaxis] )
    
    ## < 0 case
    cond_lt_zero = cond_lt_zero[...,newaxis,:].repeat( 2, -2 )
    # distances_grad[ cond < 0 ] = ( p_a / sqrt(sum( p_a**2, -2 ))[...,newaxis,:] )[ cond < 0 ]
    distances_grad[ cond_lt_zero ] = p_a[ cond_lt_zero ]
    
    ## > 1 case
    cond_gt_one = cond_gt_one[...,newaxis,:].repeat( 2, -2 )
    # distances_grad[ cond > 1 ] = ( p_b / sqrt(sum( p_b**2, -2 ))[...,newaxis,:] )[ cond > 1 ]
    distances_grad[ cond_gt_one ] = p_b[ cond_gt_one ]
    
    ## All of gradients have the distance as the denominator.
    distances_grad /= distances[...,newaxis,:]
    
    return distances, distances_grad

def min_distances_and_pt_gradients_to_edges( pts, edges ):
    '''
    Input parameter 'pts' has dimensions 2 (x,y) x #pts.
    Input parameter 'edges' has dimensions #edges x 2 endpoints x 2 coordinates (x,y).
    Returns a tuple, where the first element is
    an array of distances with dimensions #pts
    and the second element is
    an array of gradient values with dimensions 2 coordinates (x,y) x #pts.
    '''
    
    distances, gradients = distances_and_pt_gradients_to_edges( pts, edges )
    
    ## use [ argmin, range(other_dim) ] to index into arrays with the min elements from distances
    mins = distances.argmin(-2)
    dims = distances.shape
    distances = distances[ mins, range(dims[-1]) ]
    gradients = row_stack( ( gradients[...,0,:][ mins, range(dims[-1]) ], gradients[...,1,:][ mins, range(dims[-1]) ] ) )
    
    return distances, gradients


def smooth_distances_and_pt_gradients_to_edges( pts, edges ):
    '''
    Input parameter 'pts' has dimensions 2 (x,y) x #pts.
    Input parameter 'edges' has dimensions ... x #edges x 2 endpoints x 2 coordinates (x,y).
    Returns a tuple, where the first element is
    an array of smooth min distances to edges with dimensions ... x #pts
    and the second element is
    an array of gradient values with dimensions ... x 2 coordinates (x,y) x #pts.
    '''
    
    distances, gradients = distances_and_pt_gradients_to_edges( pts, edges )
    ## distances has dimensions ... x #edges x #pts
    ## gradients has dimensions ... x #edges x 2 coordinates (x,y) x #pts
    
    ## Take the L_infinity norm of all distances from a point to the edges.
    ## We want the min, not the max, so we negate the exponent to get 1/( \sum 1/dist_i ).
    sum_distances = sum( distances**(-L_infinity), -2 )
    ## sum_distances has dimensions ... x #pts
    smooth_distances = sum_distances**(-1./L_infinity)
    ## smooth_distances has dimensions ... x #pts
    
    #smooth_gradients = empty( tuple( distances.shape[:-2] ) + (2,) + ( distances.shape[-1], ), Real )
    smooth_gradients = (
        ( (-1./L_infinity) * sum_distances**( -1./L_infinity - 1 ) )[...,newaxis,:]
        *
        ( ( -L_infinity * distances**( -L_infinity - 1 ) )[...,newaxis,:] * gradients ).sum( -3 )
        )
    ## smooth_gradients has dimensions ... x 2 coordinates (x,y) x #pts
    
    return smooth_distances, smooth_gradients

kDefaultTraceDt = 5e-1
def trace_pts( pts, distances, eval_gradient, trajectories = None, dt = kDefaultTraceDt ):
    '''
    Input parameter 'pts' has dimensions 2 (x,y) x #pts.
    Input parameter 'distances' has dimensions #pts and specifies how far to trace the integral
    lines from each point in 'pts'.
    Input parameter eval_gradient is a function which takes an array of points (dimensions 2 (x,y) x #pts)
    and returns the gradient direction as an array of vectors (dimensions 2 (x,y) x #pts).
    Output parameter 'trajectories', if not None, records the trajectory (list of points)
    for each point in 'pts.'
    
    Returns 'pts' after tracing along the respective integral lines by 'distances'.
    '''
    #set_trace()
    
    if None != trajectories: trajectories[:] = [ pts.T.tolist() ]
    
    distances = asarray( distances )
    
    ## timestep size
    #dt = 5e-1
    
    trace = pts.copy()
    ## trace has dimensions 2 (x,y) x #pts
    
    ## Instead of normalizing them all the time,
    ## I can just rescale s.t. the gradient at the starting point is normalized.
    ## Denis suggests I don't normalize every time, but then I move less distance than asked for.
    kNormalizeOnce = False
    norms = None
    while distances.max() > 0.:
        gradients = eval_gradient( trace )
        ## gradients has dimensions 2 (x,y) x #pts
        
        ## gradients aren't necessarily unit length.
        if norms is None or not kNormalizeOnce:
            norms = 1. / sqrt( ( gradients**2 ).sum( -2 ) )
            ## norms has dimensions #pts
        
        #print gradients
        trace += (maximum(minimum( distances, dt ), 0.) * norms) * gradients
        distances -= dt
        
        if None != trajectories: trajectories.append( trace.T.tolist() )
    
    ## transpose trajectories
    #if None != trajectories: trajectories[:] = [ [trajectories[i][j] for i in xrange(len(trajectories))] for j in xrange(len(trajectories[0])) ]
    if None != trajectories: trajectories[:] = list( asarray(trajectories).swapaxes( 0, 1 ) )
    
    #set_trace()
    return trace

def trace_pts_in_smooth_distance_field( pts, distances, edges, trajectories = None, dt = kDefaultTraceDt ):
    '''
    Input parameter 'pts' has dimensions 2 (x,y) x #pts.
    Input parameter 'distances' has dimensions #pts and specifies how far to trace the integral
    lines from each point in 'pts'.
    Input parameter 'edges' has dimensions #edges x 2 endpoints x 2 coordinates (x,y).
    Output parameter 'trajectories', if not None, records the trajectory (list of points)
    for each point in 'pts.'
    Output parameter 'trajectories', if not None, records the trajectory (list of points)
    for each point in 'pts.'
    
    Returns 'pts' after tracing along the respective integral lines by 'distances'
    in the smooth distance field.
    '''
    grad_eval = lambda p: smooth_distances_and_pt_gradients_to_edges( p, edges )[1]
    return trace_pts( pts, distances, grad_eval, trajectories, dt )

def test_all():
    pts = [(0,0), (10,0), (0,10), (-2,-8)]
    
    edgess = []
    edgess.append( [[(-1,10), (1,10)], [(-1,.5), (1,.5)], [(1,0), (10,0)], [(-10,0), (-5,0)]] )
    edgess.append( [ [(y0, x0), (y1,x1)] for [(x0,y0),(x1,y1)] in edgess[0] ] )
    edgess.append( [ [(x1, y1), (x0,y0)] for [(x0,y0),(x1,y1)] in edgess[0] ] )
    
    pts = [(2,1)]
    # edgess = [ [[(-1,1), (1,1)]] ]
    edgess = [ [[(.25, .25), (.75, .25)], [(.25, .25), (.25, .75)], [(.75, .25), (.75, .75)], [(.25, .75), (.75, .75)]] ]
    
    print('========= testing distancesSqr_to_edges() ===========')
    #test_func( pts, edgess, 'distancesSqr_to_edges', lambda x,y: sqrt( distancesSqr_to_edges( x,y ) ) )
    test_func( pts, edgess, 'distancesSqr_to_edges', distancesSqr_to_edges )
    
    print('the sqrt(min) of distanceSqr_to_edges() should be similar to smooth_min_distance_to_edges():')
    test_func( pts, edgess, 'min_distance_to_edges', min_distance_to_edges )
    test_func( pts, edgess, 'smooth_min_distance_to_edges', smooth_min_distance_to_edges )
    
    print('========= testing distances_and_pt_gradients_to_edges() ===========')
    test_func( pts, edgess, 'distances_and_pt_gradients_to_edges', distances_and_pt_gradients_to_edges )

def unit_test_distances_and_pt_gradients_to_edges():
    edges = [[(.25, .25), (.75, .25)]]
    pts = [(.5, 0.), (.5,1.), (0.,.25), (1.,.25), (0.,0.), (1.,0.)]
    result = distances_and_pt_gradients_to_edges( asarray( pts, Real ).T, array( edges, Real ) )
    
    assert ( (result[0] - array( [ 0.25, 0.75, 0.25, 0.25, 0.35355339, 0.35355339] ))**2 ).sum() < 1e-8
    assert ( (result[1] - array( [[ 0., 0., -1., 1., -0.70710678, 0.70710678], [-1., 1., 0., 0., -0.70710678, -0.70710678]] ))**2 ).sum() < 1e-8
    print ('passed unit test distances_and_pt_gradients_to_edges()')

def test_func( pts, edgess, func_name, func ):
    print ('pts:', pts)
    
    for edges in edgess:
        print ('edges:', edges)
        print (func_name + ':', func( array( pts, Real ).T, array( edges, Real ) ))
        #for edge in edges:
        #    print 'edge:', edge
        #    print func_name + ':', func( array( pts, Real ).T, array( [edge], Real ) )

def test_gradients():
    
    ## a square
    edges = [[(.25, .25), (.75, .25)], [(.25, .25), (.25, .75)], [(.75, .25), (.75, .75)], [(.25, .75), (.75, .75)]]
    
    ## a grid of points [0..1, 0..1]
    N = 100
    pts = ( mgrid[0:N,0:N] / float(N-1) ).reshape( 2, N*N )
    
    distances, gradients = distances_and_pt_gradients_to_edges( pts, array( edges, Real ) )
    
    ## use [ argmin, range(other_dim) ] to index into arrays with the min elements from distances
    mins = distances.argmin(-2)
    dims = distances.shape
    #print mins
    #set_trace()
    distances = ( distances[ mins, range(dims[-1]) ] ).reshape( N,N )
    gradients_x = ( gradients[...,0,:][ mins, range(dims[-1]) ] ).reshape( N,N )
    gradients_y = ( gradients[...,1,:][ mins, range(dims[-1]) ] ).reshape( N,N )
    #set_trace()
    
    #mindist, mingrad = min_distances_and_pt_gradients_to_edges( pts, array( edges, Real ) )
    #assert all( mindist.reshape( N,N ) == distances )
    #assert all( mingrad[0,:].reshape( N,N ) == gradients_x )
    #assert all( mingrad[1,:].reshape( N,N ) == gradients_y )
    
    import imageio
    imageio.imwrite( 'tmp-distances.tiff', distances )
    imageio.imwrite( 'tmp-grad_x.tiff', gradients_x )
    imageio.imwrite( 'tmp-grad_y.tiff', gradients_y )
    #scipy.misc.toimage( distances.reshape( N,N ), cmin=0, cmax=255).save( 'tmp.tiff' )
    
    
    smooth_distances, smooth_gradients = smooth_distances_and_pt_gradients_to_edges( pts, array( edges, Real ) )
    
    smooth_distances = smooth_distances.reshape( N,N )
    smooth_gradients_x = smooth_gradients[0,:].reshape( N,N )
    smooth_gradients_y = smooth_gradients[1,:].reshape( N,N )
    #set_trace()
    
    import scipy
    imageio.imwrite( 'tmp-distances-smooth.tiff', smooth_distances )
    imageio.imwrite( 'tmp-grad-smooth_x.tiff', smooth_gradients_x )
    imageio.imwrite( 'tmp-grad-smooth_y.tiff', smooth_gradients_y )
    #scipy.misc.toimage( distances.reshape( N,N ), cmin=0, cmax=255).save( 'tmp.tiff' )

def test_trace():
    
    cases = []
    
    edges = [[(.25, .25), (.75, .25)]]
    pts = [(.5, 0.), (.5,1.), (0.,.25), (1.,.25), (0.,0.), (1.,0.)]
    cases.append( (pts, edges) )
    
    ## a square
    edges = [[(.25, .25), (.75, .25)], [(.25, .25), (.25, .75)], [(.75, .25), (.75, .75)], [(.25, .75), (.75, .75)]]
    pts = [(.3,.3)]
    cases.append( (pts, edges) )
    
    for pts, edges in cases:
        edges = array( edges, Real )
        pts = array( pts, Real ).T
        
        print( 'pts:', pts )
        print( 'edges:', edges )
        
        for dist_grad_eval in ( min_distances_and_pt_gradients_to_edges, smooth_distances_and_pt_gradients_to_edges ):
            print ('running function:', dist_grad_eval)
            
            grad_eval = lambda p: dist_grad_eval( p, edges )[1]
            
            ## this shouldn't go anywhere (we're asking it to return immediately)
            traces = trace_pts( pts, zeros( pts.shape[-1] ), grad_eval )
            assert ( (traces - pts)**2 ).sum() <= 1e-5
            
            print ('traces:', traces)
            print ('distance travelled:', sqrt( ( (traces - pts)**2 ).sum(-2) ))
            
            ## ideally this should travel distance 1
            traces = trace_pts( pts, ones( pts.shape[-1] ), grad_eval )
            #assert ( ( ( (traces - pts)**2 ).sum(-2) - ones( pts.shape ) )**2 ).sum() <= 1e-5
            
            print( 'traces:', traces)
            print( 'distance travelled:', sqrt( ( (traces - pts)**2 ).sum(-2) ))

def main():
    unit_test_distances_and_pt_gradients_to_edges()
    
    test_all()
    
    test_gradients()
    
    test_trace()

if __name__ == '__main__':
    main()
