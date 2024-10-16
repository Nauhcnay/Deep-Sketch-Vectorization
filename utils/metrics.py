from __future__ import print_function, division
from numpy import *
import numpy as np
from tqdm import tqdm
from PIL import Image
# from tools import *

def load_image_from_path_as_array( path ):
    '''
    Given:
        path: a path to an image file
    Returns:
        arr: the image data loaded as a rows-by-columns
             array of floating point values between [0,1].
    '''
    
    
    img = Image.open( path ).convert( 'L' )
    arr = asfarray( img ) / 255.
    return arr

def find_fraction_of_image_inside_mask( img_A, mask ):
    '''
    Given:
        img_A: an N-by-M image of floating point greyscale values
        mask: an N-by-M boolean array
    Returns:
        fraction: the fraction of total grey in `img_A` under `mask`.
                  Fails if all elements of `img_A` are zero.
    
    NOTE: This function supports greyscale values.
    NOTE: If blackness represents the presence of something, you want `1-fraction`.
    '''
    
    assert img_A.shape == mask.shape
    assert len( img_A.shape ) == 2
    
    total = img_A.sum()
    masked = img_A[ mask ].sum()
    
    return masked/total

def make_thick_mask_from_image_boolean( img, kernel, pixel_diameter ):
    '''
    Thickens a boolean image for use as a mask to `find_fraction_of_image_inside_mask()`.
    For example, given an image with 1-pixel thick strokes, thickens the strokes to
    the given diameter.
    
    Given:
        img: an N-by-M numpy array of boolean values
        pixel_diameter: the amount to dilate `img` as a non-negative integer
    Returns:
        thick_mask: the True values of `img` are expanded via dilation to a thick mask.
    '''
    
    assert len( img.shape ) == 2
    
    ## Must be an integer.
    assert pixel_diameter == int( pixel_diameter )
    
    import cv2
    return cv2.dilate( img, kernel, iterations = pixel_diameter )

def find_closest_point_distances_for_images_greyscale( img_A, img_B ):
    '''
    Given:
        img_A: an N-by-M image of floating point greyscale values
        img_B: an N-by-M image of floating point greyscale values
    Returns:
        distances: for each black = 0.0 pixel in `img_A`, return the
                   distance (not squared) to the closest black = 0.0 pixel in `points_B`.
    '''
    
    assert img_A.shape == img_B.shape
    assert len( img_A ) == 2
    
    return find_closest_point_distances_for_images_boolean( img_A == 0.0, img_B == 0.0 )

def find_closest_point_distances_for_images_boolean( img_A, img_B ):
    '''
    Given:
        img_A: an N-by-M image of boolean values
        img_B: an N-by-M image array of boolean values
    Returns:
        distances: for each True pixel in `img_A`, return the distance (not squared) to
                   the closest True pixel in `points_B`.
    '''
    # size is represented as width height
    assert img_A.size == img_B.size
    ## `nonzero()` returns the transpose of the indices of img_A which are True.
    # points are represented as height width
    points_A, indices_A = image_to_points(img_A)
    points_B, _ = image_to_points(img_B)
    
    distances = find_closest_point_distances_for_point_sequences( points_A, points_B )
    
    # Create a masked array, so that no one can read values outside the mask.
    result = ma.array(zeros(img_A.size).T, mask = image_to_mask(img_A))    
    result.data[indices_A] = distances
    # a easy way to verify the result is generated right
    assert(np.max(result) == np.max(distances))
    return result

def polylines_to_dense_points( list_of_polylines ):
    '''
    Given:
        list_of_polylines: a list of sequences of ( boolean: closed, 2D array: points ).
    Returns:
        points: a sequence of 2D points
    '''
    
    ## Pick a parser. Convert to polylines sampled densely and evenly.
    ## Return a sequence of 2D points
    
    raise NotImplementedError

def load_svg_from_path_to_polylines( path ):
    '''
    Given:
        path: a path to an SVG file
    Returns:
        list_of_polylines: a list of sequences of ( boolean: closed, 2D array: points ).
    '''
    
    ## Pick a parser. Convert to polylines.
    ## Pick an output format. How about?
    ## List[ ( boolean: closed, 2D array: points ), ... ]
    
    raise NotImplementedError

def find_fraction_of_points_within_distance_of_points( points_A, points_B, distance ):
    '''
    Given:
        points_A: a sequence of 2D points
        points_B: a sequence of 2D points
        distance: a number representing the cutoff distance
    Returns:
        fraction: the fraction of points in A within `distance` of any point in B.
    '''
    
    distances = find_closest_point_distances_for_point_sequences( points_A, points_B )
    return ( distances <= distance ).sum() / len( distances ) 

def find_fraction_of_points_within_distance_of_polylines( points_A, list_of_polylines_B, distance ):
    '''
    Given:
        points_A: a sequence of 2D points
        list_of_polylines_B: a list of sequences of ( boolean: closed, 2D array: points )
        distance: a number representing the cutoff distance
    Returns:
        fraction: the fraction of points in A within `distance` of any point in B.
    '''
    
    distances = find_closest_point_distances_for_points_to_polylines( points_A, list_of_polylines_B )
    return ( distances <= distance ).count_nonzero() / len( distances )

def find_closest_point_distances_for_points_to_polylines( points_A, list_of_polylines_B ):
    '''
    Given:
        points_A: a sequence of 2D points
        list_of_polylineslylines_B: a list of sequences of ( boolean: closed, 2D array: points )
    Returns:
        distances: for each point in `points_A`, return the distance (not squared) to
                   the closest point on an edge of `list_of_polylines_B`.
    '''
    
    raise NotImplementedError

def find_closest_point_distances_for_point_sequences( points_A, points_B ):
    '''
    Given:
        points_A: a sequence of 2D points
        points_B: a sequence of 2D points
    Returns:
        distances: for each point in `points_A`, return the distance (not squared) to
                   the closest point in `points_B`.
    '''
    
    ## Get the index of the closest point in B to each point in A
    indices = find_closest_point_indices_for_point_sequences( points_A, points_B )
    assert len( indices ) == len( points_A )
    
    ## Subtract each point in A from its closest point in B.
    ## Each element in `deltas` is a little 2D vector.
    deltas = ( points_A - points_B[ indices ] )
    assert deltas.shape == points_A.shape
    
    ## Compute the length of each vector in `deltas`: sqrt( x^2 + y^2 )
    distances = sqrt( ( deltas**2 ).sum(-1) )
    ## The result should be a len(A) vector.
    assert len( distances ) == len( points_A )
    assert len( distances.shape ) == 1
    
    return distances

def find_closest_point_indices_for_point_sequences( points_A, points_B ):
    '''
    Given:
        points_A: a sequence of 2D points
        points_B: a sequence of 2D points
    Returns:
        indices: for each point in `points_A`, return the index of the closest point
                 in `points_B`.
    '''
    indices = []
    for i in tqdm(range(len(points_A)), ncols=50):
        distances = ((points_A[i] - points_B)**2).sum(-1)
        min_distance_indice = distances.argmin()
        indices.append(min_distance_indice)
    return indices

def compute_distance_transform_for_boolean_array( arr ):
    '''
    Given:
        arr: an N-by-M array of boolean values
    Returns:
        distances: an N-by-M array of floating point L2 distances (not squared) to
                   the closest True pixel in `img`.
    '''
    
    import cv2 as cv
    result = cv.distanceTransform( logical_not(arr).astype(uint8), cv.DIST_L2, cv.DIST_MASK_PRECISE )
    return result

def find_closest_point_distances_for_images_boolean_fast( bool_arr_A, bool_arr_B ):
    '''
    Given:
        bool_arr_A: an N-by-M array of boolean values
        bool_arr_B: an N-by-M array array of boolean values
    Returns:
        distances: for each True pixel in `bool_arr_A`, return the distance (not squared)
                   to the closest True pixel in `bool_arr_B` as a masked array.
    '''
    assert bool_arr_A.shape == bool_arr_B.shape
    
    distance_to_B = compute_distance_transform_for_boolean_array( bool_arr_B )
    
    # Create a masked array, so that no one can read values outside the mask.
    result = ma.array( distance_to_B, mask = logical_not(bool_arr_A) )
    # a easy way to verify the result is generated right
    assert(np.max(result) == np.max(distance_to_B[ bool_arr_A ]))
    return result
