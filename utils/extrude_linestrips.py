#!/usr/bin/env python3

## Author: Yotam Gingold <yotam@yotamgingold.com>
## License: CC0

import numpy as np

def extrude_linestrips( zs, linestrips, areloops = None ):
    '''
    Given:
        zs: A sequence of z values to use for extruding `linestrips`. For example, `linespace( 0, 1, 10 )`.
        linestrips: A sequence of linestrips. Each linestrip is a sequences of x,y points.
        areloops (optional): A sequence of booleans with the same length as `linestrips`. If the corresponding boolean is True, the line strip is closed. If this parameter is not passed, it is assumed to be False.
    Returns:
        V: A sequence of x,y points.
        F: A sequence of triplets of indices indexing into `vertices`.
    '''
    
    if areloops is None:
        areloops = [False] * len( linestrips )
    
    V = []
    F = []
    for linestrip, isloop in zip( linestrips, areloops ):
        Vl, Fl = extrude_linestrip( zs, linestrip, isloop )
        ## Fl indexes into Vl. But Vl will be appended to V.
        ## So Vl[0] will be at V[ len(V) ]. Add this offset.
        F.extend( ( np.array( Fl ) + len(V) ).tolist() )
        ## Now extend the vertices. (Do this after accessing len(V) above.)
        V.extend( Vl )
    
    return V, F

def extrude_linestrip( zs, linestrip, isloop = None ):
    '''
    Given:
        zs: A sequence of z values to use for extruding `linestrips`. For example, `linespace( 0, 1, 10 )`.
        linestrip: Each linestrip is a sequences of x,y points.
        isloop (optional): If True, the last point is connected to the first point. The default is False.
    Returns:
        V: A sequence of x,y points.
        F: A sequence of triplets of indices indexing into `vertices`.
    '''
    
    if isloop is None: isloop = False
    
    ## Copy the point z times to extrude it into a stack.
    V = []
    for x,y in linestrip: V.extend( [ ( x, y, z ) for z in zs ] )
    ## Map from a ( linestrip index, z index ) pair to its 1D offset into V
    def lz2index( l_index, z_index ): return l_index*len(zs) + z_index
    
    ## Create a square (two triangles) for each edge.
    '''
    x1,y1,z2 ---- x2,y2,z2
       |   \          |
       |      \       |
       |         \    |
    x1,y1,z1 ---- x2,y2,z1
    '''
    F = []
    for l_index in range(len(linestrip)-1):
        for z_index in range(len(zs)-1):
            F.append( ( lz2index( l_index + 0, z_index + 0 ), lz2index( l_index + 1, z_index + 0 ), lz2index( l_index + 0, z_index + 1 ) ) )
            F.append( ( lz2index( l_index + 1, z_index + 0 ), lz2index( l_index + 1, z_index + 1 ), lz2index( l_index + 0, z_index + 1 ) ) )
    
    if isloop:
        l_index = len(linestrip)-1
        ## Wrap around so that the last point connects to the first point. That means we use 0 instead of l_index+1
        for z_index in range(len(zs)-1):
            F.append( ( lz2index( l_index + 0, z_index + 0 ), lz2index(           0, z_index + 0 ), lz2index( l_index + 0, z_index + 1 ) ) )
            F.append( ( lz2index(           0, z_index + 0 ), lz2index(           0, z_index + 1 ), lz2index( l_index + 0, z_index + 1 ) ) )
    
    return V, F

def test_extrude():
    import polyscope as ps
    ps.init()
    
    ## z = -1, 0, 1
    zs0 = np.linspace( -1, 1, 3 )
    ## Make an L shape
    linestrip0 = [ ( 0,2 ), ( 0,1 ), ( 0,0 ), ( 1,0 ) ]
    ## Extrude
    V0,F0 = extrude_linestrip( zs0, linestrip0 )
    V0 = np.asarray( V0 )
    ps.register_surface_mesh( "linestrip0", V0, F0, smooth_shade = False )
    
    ## z = 2 to 3 in 10 steps
    zs1 = np.linspace( 2, 3, 10 )
    ## Make a box
    linestrip1 = [ ( 0.1,0.1 ), ( 1.1,0.1 ), ( 1.1,1.1 ), ( 0.1,1.1 ) ]
    ## Extrude
    V1,F1 = extrude_linestrip( zs1, linestrip1, True )
    V1 = np.asarray( V1 )
    ps.register_surface_mesh( "linestrip1", V1, F1, smooth_shade = False )
    
    ## Extrude together as strips (default areloops parameter)
    Vall,Fall = extrude_linestrips( np.linspace( -5, -4, 5 ), [ linestrip0, linestrip1 ] )
    Vall = np.asarray( Vall )
    ps.register_surface_mesh( "both as strips", Vall, Fall, smooth_shade = False )
    
    ## Extrude together with the box as a loop
    Vallloop,Fallloop = extrude_linestrips( np.linspace( -3, -2, 5 ), [ linestrip0, linestrip1 ], [ False, True ] )
    Vallloop = np.asarray( Vallloop )
    ps.register_surface_mesh( "strip and loop", Vallloop, Fallloop, smooth_shade = False )
    
    ps.show()

if __name__ == '__main__':
    test_extrude()