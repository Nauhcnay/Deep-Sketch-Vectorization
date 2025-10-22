## Dual Contouring Algorithm on 2D UDF by Chuan Yan
# this script contains all necessary functions for testing this algorithm, all you need is one SVG file as input
# please also make sure the input SVG only contains simple line segments

import numpy as np
import cv2 
from tqdm import tqdm
from PIL import Image
from svgpathtools import Path, Line, wsvg
from scipy.interpolate import interpn, interp2d, interp1d, BivariateSpline
from skimage.morphology import skeletonize
from functools import partial

scaling = None
def DS(y, x):
    global scaling
    assert y % scaling == 0
    assert x % scaling == 0
    return (int(y / scaling), int(x / scaling))

def d2d3_is_local_minimum(d1, d2, d3, d4):
    return (d1 + d2) > (d2 + d3) and (d3 + d4) > (d2 + d3)

def d2tod5_all_small(d2, d3, d4, d5):
    return (d2 + d3) <= 1 and (d3 + d4) <= 1 and (d4 + d5) <= 1

def d2d3_is_local_maximum(d2, d3, d4, d5):
    return (d2 + d3) > (d3 + d4) and (d2 + d3) > (d4 + d5)

def is_parallel(udf, x, y, scaling, axis):
    h, w  = udf.shape
    x1 = (x - 1) * scaling; x2 = x * scaling; x3 = (x + 1) * scaling; x4 = (x + 2) * scaling
    y1 = (y - 1) * scaling; y2 = y * scaling; y3 = (y + 1) * scaling; y4 = (y + 2) * scaling
    if x4 >= w or y4 >= h: return False
    d11 = udf[y1, x1]; d12 = udf[y1, x2]; d13 = udf[y1, x3]; d14 = udf[y1, x4];
    d21 = udf[y2, x1]; d22 = udf[y2, x2]; d23 = udf[y2, x3]; d24 = udf[y2, x4];
    d31 = udf[y3, x1]; d32 = udf[y3, x2]; d33 = udf[y3, x3]; d34 = udf[y3, x4];
    d41 = udf[y4, x1]; d42 = udf[y4, x2]; d43 = udf[y4, x3]; d44 = udf[y4, x4];
    
    if axis == 0: # y-axis
        is_parallel_case1 = np.array([d22, d32, d42]).std() < 0.05
        all_left = (d21 + d22) <= 1 and (d31 + d32) <= 1 and (d41 + d42) <= 1 and \
            (d22 + d23) > 1 and (d32 + d33) > 1 and (d42 + d43) > 1
        all_right = (d21 + d22) > 1 and (d31 + d32) > 1 and (d41 + d42) > 1 and \
            (d22 + d23) <= 1 and (d32 + d33) <= 1 and (d42 + d43) <= 1
        is_parallel_case2 = all_left or all_right
    else:
        assert axis == 1 # x-axis
        is_parallel_case1 = np.array([d22, d23, d24]).std() < 0.05
        all_above = (d12 + d22) <= 1 and (d13 + d23) <= 1 and (d14 + d24) <= 1 and \
            (d32 + d22) > 1 and (d33 + d23) > 1 and (d34 + d24) > 1
        all_below = (d12 + d22) > 1 and (d13 + d23) > 1 and (d14 + d24) > 1 and \
            (d32 + d22) <= 1 and (d33 + d23) <= 1 and (d34 + d24) <= 1
        is_parallel_case2 = all_above or all_below
    return is_parallel_case1 or is_parallel_case2

def sovle_y(y, x, has_above, has_below, has_left, has_right, 
        has_upper_left, has_upper_right, has_lower_left, has_lower_right,
        has_left_below, has_left_above, has_right_below, has_right_above, 
        at_1st, at_2nd, at_3rd, at_4th):
    ext_above = None
    ext_left = None
    if has_left and (at_3rd and at_1st):
        if not has_right_below:
            ext_above = True
    elif has_left and (at_2nd and at_4th):
        if not has_right_above:
            ext_above = False
    elif has_right and (at_2nd and at_4th):
        if not has_left_below:
            ext_above = True
    elif has_right and (at_3rd and at_1st):
        if not has_left_above:
            ext_above = False
    elif has_right and has_upper_left and has_lower_left:
        pass
    elif has_left and has_upper_right and has_bottom_right:
        pass
    elif has_left and (at_3rd or at_1st):
        ext_above = True
    elif has_left and (at_2nd or at_4th):
        ext_above = False
    elif has_right and (at_2nd or at_4th):
        ext_above = True
    elif has_right and (at_3rd or at_1st):
        ext_above = False
    # if has_left and (at_3rd or at_1st) and has_above:
    #     ext_above = True
    # elif has_left and (at_2nd or at_4th) and has_below:
    #     ext_above = False
    # elif has_right and (at_2nd or at_4th) and has_above:
    #     ext_above = True
    # elif has_right and (at_3rd or at_1st) and has_below:
        # ext_above = False
    elif at_3rd and at_4th:
        ext_above = False
    elif at_2nd and at_1st:
        ext_above = True
    elif at_3rd or at_4th:
        ext_above = False
    elif at_2nd or at_1st:
        ext_above = True
    else:
        print("Warning:\tunsolvable ambiguity structures at (%d, %d), this usually suggest wrong stroke topolgy at dense stroke regions"%(y, x))
    return ext_above, ext_left

def sovle_x(y, x, has_above, has_below, has_left, has_right, 
        has_upper_left, has_upper_right, has_lower_left, has_lower_right, 
        has_left_below, has_left_above, has_right_below, has_right_above, 
        at_1st, at_2nd, at_3rd, at_4th):
    ext_above = None
    ext_left = None
    if has_above and (at_3rd and at_1st):
        if not has_lower_right:
            ext_left = True
    elif has_above and (at_2nd and at_4th):
        if not has_lower_left:
            ext_left = False
    elif has_below and (at_3rd and at_1st):
        if not has_upper_left:
            ext_left = False
    elif has_below and (at_2nd and at_4th):
        if not has_upper_right:
            ext_left = True
    elif has_above and has_left_below and has_right_below:
        pass
    elif has_below and has_left_above and has_right_above:
        pass
    elif has_above and (at_3rd or at_1st):
        ext_left = True
    elif has_above and (at_2nd or at_4th):
        ext_left = False
    elif has_below and (at_2nd or at_4th):
        ext_left = True
    elif has_below and (at_3rd or at_1st):
        ext_left = False
    # vertical case
    elif at_2nd and at_3rd:
        ext_left = True
    elif at_1st and at_4th:
        ext_left = False
    elif at_2nd or at_3rd:
        ext_left = True
    elif at_1st or at_4th:
        ext_left = False
    else:
        print("Warning:\tunsolvable ambiguity structures at (%d, %d), this usually suggest wrong stroke topolgy at dense stroke regions"%(y, x))
    return ext_above, ext_left

def solve_xy(at_1st, at_2nd, at_3rd, at_4th, interp_mask, x, y, udf, pt_matrix, scaling):
    '''
        Method 2:
        add check the stroke direction based on Yotam's suggestion
    '''
    '''
        There is one corner case, if the ambiguous point are consecutive along 
        x or y axis, and we searching the point from top to bottom, left to right
        then it is impossible to figure out the correct topology if we got a special 
        point group: 
        it starts with a x,y ambiguous pt and end with a y ambiguous pt on x-axis 
        (or x ambiguous pt on y-axis)
        so we need to sovle the ambiguity point on the RIGHT / BELOW first then continue
        the processing logic 
    '''
    ext_above = None
    ext_left = None
    # get coord index and distance valuce along y-axis, y2, x2 is the point location
    x0, x1, x2, x3, x4, x5, x6, y0, y1, y2, y3, y4, y5, y6,\
    d0, d1, d2, d3, d4, d5, d6, d00, d11, d22, d33, d44, d55, d66,\
    x00, x11, x22, x33, x44, x55, x66, y00, y11, y22, y33, y44, y55, y66,\
    xs1_, xs2_, xs3_, ys1_, ys2_, ys3_\
        = get_coords_dist_from_udf(x, y, udf, scaling, pt_matrix)

    at_1st_3rd = at_1st and at_3rd
    at_2nd_4th = at_2nd and at_4th
    has_above = interp_mask[y2+1:y3, x3].all()
    has_below = interp_mask[y3+1:y4, x3].all()
    has_left = interp_mask[y3, x2+1:x3].all()
    has_right = interp_mask[y3, x3+1:x4].all()
    has_upper_left = interp_mask[y2, x2+1:x3].all()
    has_upper_right = interp_mask[y2, x3+1:x4].all()
    has_lower_left = interp_mask[y4, x2+1:x3].all()
    has_lower_right = interp_mask[y4, x3+1:x4].all()
    has_left_above = interp_mask[y2+1:y3, x2].all()
    has_right_above = interp_mask[y2+1:y3, x4].all()
    has_left_below = interp_mask[y3+1:y4, x2].all()
    has_right_below = interp_mask[y3+1:y4, x4].all()
    # reminder: these flags don't consider if they are safe, that is why we need to re-compute them again
    has_flags_1st = has_right_above or has_upper_right
    has_flags_2nd = has_left_above or has_upper_left
    has_flags_3rd = has_left_below or has_lower_left
    has_flags_4th = has_right_below or has_lower_right

    if at_1st_3rd == at_2nd_4th:
        if at_1st and not (at_2nd or at_3rd or at_4th):
            # we need to decide the direction of the stroke because 
            # the current information still can't help us to make clear decision
            # if could have 3 possible cases:
            # 1. x-axis, 2.y-axis, 3. 45 degree (3rd to 1st)
            if has_flags_2nd and has_above:
                # case x axis
                interp_mask[y3, x2+1:x3] = False
                interp_mask[y3, x3+1:x4] = False
                interp_mask[y3+1:y4, x3] = False
                return ext_above, ext_left
            elif has_flags_4th and has_right:
                # case y axis
                interp_mask[y3, x2+1:x3] = False # left
                interp_mask[y2+1:y3, x3] = False # above
                interp_mask[y3+1:y4, x3] = False # below
                return ext_above, ext_left    
            else:
                # case 45 degree
                at_1st_3rd = True
        elif at_2nd and not (at_1st or at_3rd or at_4th):
            if has_flags_1st and has_above:
                # case x axis
                interp_mask[y3, x2+1:x3] = False
                interp_mask[y3, x3+1:x4] = False
                interp_mask[y3+1:y4, x3] = False
                return ext_above, ext_left
            elif has_flags_3rd and has_left:
                # case y axis
                interp_mask[y3, x3+1:x4] = False # right
                interp_mask[y2+1:y3, x3] = False # above
                interp_mask[y3+1:y4, x3] = False # below
                return ext_above, ext_left    
            else:
                # case -45 degree
                at_2nd_4th = True
        elif at_3rd and not (at_2nd or at_1st or at_4th):
            if has_flags_4th and has_below:
                interp_mask[y3, x2+1:x3] = False # left
                interp_mask[y3, x3+1:x4] = False # right
                interp_mask[y2+1:y3, x3] = False # above
                return ext_above, ext_left    
            elif has_flags_2nd and has_left:
                # case y axis
                interp_mask[y3, x3+1:x4] = False # right
                interp_mask[y2+1:y3, x3] = False # above
                interp_mask[y3+1:y4, x3] = False # below
                return ext_above, ext_left
            else:
                at_1st_3rd = True
        elif at_4th and not (at_2nd or at_3rd or at_1st):
            if has_flags_1st and has_right:
                interp_mask[y3, x2+1:x3] = False # left
                interp_mask[y2+1:y3, x3] = False # above
                interp_mask[y3+1:y4, x3] = False # below
                return ext_above, ext_left
            elif has_flags_3rd and has_below:
                interp_mask[y3, x2+1:x3] = False # left
                interp_mask[y3, x3+1:x4] = False # right
                interp_mask[y2+1:y3, x3] = False # above
                return ext_above, ext_left
            else:
                at_2nd_4th = True
        # this indicate a corner (sharp turn)
        # because the cross shape ambiguous region is a full connection region
        # it can connect stroke from any direction
        elif at_1st and at_2nd:
            interp_mask[y3, x2+1:x3] = False # left
            interp_mask[y3, x3+1:x4] = False # right
            interp_mask[y3+1:y4, x3] = False # below
            return ext_above, ext_left
        elif at_2nd and at_3rd:
            interp_mask[y3, x3+1:x4] = False # right
            interp_mask[y2+1:y3, x3] = False # above
            interp_mask[y3+1:y4, x3] = False # below
            return ext_above, ext_left
        elif at_3rd and at_4th:
            interp_mask[y3, x2+1:x3] = False # left
            interp_mask[y3, x3+1:x4] = False # right
            interp_mask[y2+1:y3, x3] = False # above
            return ext_above, ext_left
        elif at_4th and at_1st:
            interp_mask[y3, x2+1:x3] = False # left
            interp_mask[y2+1:y3, x3] = False # above
            interp_mask[y3+1:y4, x3] = False # below
            return ext_above, ext_left
        else:
            # discard this point
            interp_mask[y3, x2+1:x3] = False # left
            interp_mask[y3, x3+1:x4] = False # right
            interp_mask[y2+1:y3, x3] = False # above
            interp_mask[y3+1:y4, x3] = False # below
            return ext_above, ext_left
            
    # this case usually indicates there are two consecutive y, x ambiguous point along x or y axis
    # and its rigth / below ambiguous point has been resolved already, so we don't need to resolve this again
    # and we just need to infer the result
    if at_1st_3rd and (has_above == False or has_left == False):
        ext_above = False
        ext_left = False
    elif at_1st_3rd and (has_below == False or has_right == False):
        ext_above = True
        ext_left = True
    elif at_2nd_4th and (has_above == False or has_right == False):
        ext_above = False
        ext_left = True
    elif at_2nd_4th and (has_below == False or has_left == False):
        ext_above = True
        ext_left = False
    else:
        # the segement which has larger distance starndar deviation means the segment has larger gradient
        # which means that segment is more reliable
        if np.array([d1, d2, d4, d5]).std() > np.array([d11, d22, d44, d55]).std():
            x_first = True
        else:
            x_first = False
        x_intersec = find_intersection(x11, x22, d1, d2, x44, x55, d4, d5)
        y_intersec = find_intersection(y11, y22, d11, d22, y44, y55, d44, d55)
        if x_intersec <= x33:
            ext_left = True
        else:
            ext_left = False
        if y_intersec <= y33 :
            ext_above = True
        else:
            ext_above = False
        if x_first:
            if at_1st_3rd:
                if ext_left:
                    ext_above = True
                else:
                    ext_above = False
            if at_2nd_4th:
                if ext_left:
                    ext_above = False
                else:
                    ext_above = True
        else:
            if at_1st_3rd:
                if ext_above:
                    ext_left = True
                else:
                    ext_left = False
            if at_2nd_4th:
                if ext_above:
                    ext_left = False
                else:
                    ext_left = True
    return ext_above, ext_left

def find_consecutive_pts(ambiguous_pts, pt, axis, force_direction = False):
    assert pt in ambiguous_pts
    y, x = pt
    hit_list = []
    direction = 'y' if axis == 0 else 'x'
    while (y, x) in ambiguous_pts:
        hit = False
        if 'r' in ambiguous_pts[(y, x)]: break
        if force_direction:
            if (y, x) != pt and (direction in ambiguous_pts[(y, x)] or direction + "_plus" in ambiguous_pts[(y, x)]):
                hit_list.append((y, x))
                hit = True
        elif (y, x) != pt:
            hit_list.append((y, x))
            hit = True
        if axis == 0: 
            y += 1
        elif axis == 1:
            x += 1
        else:
            raise ValueError("Can't support dimensions greater than 2!")
        if not hit:
            break
    return hit_list

def resolve_ambiguou_pt(ext_above, ext_left, udf_interpolated, interp_mask, x, y, \
    udf, pt_matrix, scaling, ambiguous_pts, pt_x, pt_y):
    x0, x1, x2, x3, x4, x5, x6, y0, y1, y2, y3, y4, y5, y6,\
    d0, d1, d2, d3, d4, d5, d6, d00, d11, d22, d33, d44, d55, d66,\
    x00, x11, x22, x33, x44, x55, x66, y00, y11, y22, y33, y44, y55, y66,\
    xs1_, xs2_, xs3_, ys1_, ys2_, ys3_\
        = get_coords_dist_from_udf(x, y, udf, scaling, pt_matrix)
    pt = (y, x)

    # if ext_above == None:
    #     interp_mask[y3+1:y4, x3] = False
    #     interp_mask[y2+1:y3, x3] = False

    if ext_above == True: # stroke across y2, y3
        _, f0 = find_zeropt(y22, y33, d22, -d33)
        ds1 = f0(ys1_)
        # we need to make sure the gradient direction is correct
        # when the zero point is just next to y22
        if abs(ds1)[0] > udf_interpolated[y2, x3]:
            ds1[0] = udf_interpolated[y2, x3] / 2
        udf_interpolated[y2+1:y3, x3] = ds1
        interp_mask[y2+1:y3, x3] = True
        # assert (interp_mask[y2+1:y3, x3]).all() == True
        interp_mask[y3+1:y4, x3] = False

    elif ext_above == False: # stroke across y3, y4
        _, f0 = find_zeropt(y33, y44, d33, -d44)
        ds2 = f0(ys2_)
        if abs(ds2)[0] > udf_interpolated[y3, x3]:
            ds2[0] = udf_interpolated[y3, x3] / 2
        udf_interpolated[y3+1:y4, x3] = ds2
        interp_mask[y2+1:y3, x3] = False
        interp_mask[y3+1:y4, x3]= True
        # assert (interp_mask[y3+1:y4, x3]).all() == True

    # if ext_left == None:
    #     interp_mask[y3, x3+1:x4] = False
    #     interp_mask[y3, x2+1:x3] = False

    elif ext_left == True: # stroke across x2, x3
        _, f0 = find_zeropt(x22, x33, d2, -d3)
        ds1 = f0(xs1_)
        if abs(ds1)[0] > udf_interpolated[y3, x2]:
            ds1[0] = udf_interpolated[y3, x2] / 2
        udf_interpolated[y3, x2+1:x3] = ds1
        # assert(interp_mask[y3, x2+1:x3]).all() == True
        interp_mask[y3, x2+1:x3] = True
        interp_mask[y3, x3+1:x4] = False

    elif ext_left == False: # stroke across x3, x4
        _, f0 = find_zeropt(x33, x44, d3, -d4)
        ds2 = f0(xs2_)
        if abs(ds2)[0] > udf_interpolated[y3, x3]:
            ds2[0] = udf_interpolated[y3, x3] / 2
        udf_interpolated[y3, x3+1:x4] = ds2
        interp_mask[y3, x2+1:x3] = False
        # assert(interp_mask[y3, x3+1:x4]).all() == True
        interp_mask[y3, x3+1:x4] = True

    # corner case 1:
    # if the ambiguous edges are consecutive and more than 2, this is definitely something need to be solved
    # because we will only expect one single smooth stroke, so it is impossible for one stroke repeatly 
    if ext_left or ext_left == False:
        # find all consecutive ambiguous points along x-axis 
        ambi_pts_x = find_consecutive_pts(ambiguous_pts, pt, 1)
        last_is_skip = False
        for ptx in ambi_pts_x: 
            reset = False
            skip = False
            # we will tend to keep the connection for strokes along y-direction
            # so that will indicate a X junction there
            yx, xx = ptx
            yx2 = (yx - 1) * scaling; yx4 = (yx + 1) * scaling
            xx3 = xx * scaling; xx4 = (xx + 1) * scaling
            has_flag_above = interp_mask[yx2, xx3 + 1 : xx4].all()
            has_flag_below = interp_mask[yx4, xx3 + 1 : xx4].all()
            maybe_x_junction = has_flag_above and has_flag_below
            y, x = ptx
            xp = (x - 1) * scaling
            xn = (x + 1) * scaling
            y *= scaling
            x *= scaling
            if "x" in ambiguous_pts[ptx] and  "y" in ambiguous_pts[ptx]:
                # ambiguous_pts[ptx].remove('x')
                # reset = True
                interp_mask[y, xp+1:x] = True
                interp_mask[y, x+1:xn] = True
                skip = True
            elif "x" in ambiguous_pts[ptx]:
                ambiguous_pts[ptx].append("r")
                reset = True
            if reset and skip == False:
                if not last_is_skip:
                    interp_mask[y, xp+1:x] = False
                if not maybe_x_junction:
                    interp_mask[y, x+1:xn] = False
            if "x" in ambiguous_pts[ptx] and  "y" in ambiguous_pts[ptx]:
                last_is_skip = True
            else:
                last_is_skip = False
                
    if ext_above or ext_above == False:
        # find all consecutive ambiguous points along y-axis 
        ambi_pts_y = find_consecutive_pts(ambiguous_pts, pt, 0)
        last_is_skip = False
        for pty in ambi_pts_y:
            reset = False
            skip = False
            yy, xy = pty
            xy2 = (xy - 1) * scaling; xy4 = (xy + 1) * scaling
            yy3 = yy * scaling; yy4 = (yy + 1) * scaling
            has_flag_left = interp_mask[yy3 + 1 : yy4, xy2].all()
            has_flag_right = interp_mask[yy3 + 1 : yy4, xy4].all()
            maybe_x_junction = has_flag_left and has_flag_right
            y, x = pty
            yp = (y - 1) * scaling
            yn = (y + 1) * scaling
            y *= scaling
            x *= scaling
            if "x" in ambiguous_pts[pty] and  "y" in ambiguous_pts[pty]:
                # ambiguous_pts[pty].remove('y')
                # reset = True
                interp_mask[yp+1:y, x] = True
                interp_mask[y+1:yn, x] = True
                skip = True
            elif "y" in ambiguous_pts[pty]:
                ambiguous_pts[pty].append("r")
                reset = True
            if reset and skip == False:    
                if last_is_skip == False:
                    interp_mask[yp+1:y, x] = False
                if not maybe_x_junction:
                    interp_mask[y+1:yn, x] = False
            if "x" in ambiguous_pts[pty] and  "y" in ambiguous_pts[pty]:
                last_is_skip = True
            else:
                last_is_skip = False

    # corner case 2: if there is another stroke across x4, x5 and it has not been detected
    # we will need to pick them up at this very last stage
    if ext_left and is_in(pt_x, DS(y3, x4)) and d2d3_is_local_minimum(d3, d4, d5, d6):
        # infact, we don't need to interpolate on our current work flow...
        _, f0 = find_zeropt(x44, x55, d4, -d5)
        ds3 = f0(xs3_)
        if abs(ds3)[0] > udf_interpolated[y3, x4]:
            ds3[0] = udf_interpolated[y3, x4] / 2
        udf_interpolated[y3, x4+1:x5] = ds3
        interp_mask[y3, x4+1:x5] = True

    if ext_above and is_in(pt_y, DS(y4, x3)) and d2d3_is_local_minimum(d3, d4, d5, d6):
        _, f0 = find_zeropt(y44, y55, d44, -d55)
        ds3 = f0(ys3_)
        if abs(ds3)[0] > udf_interpolated[y4, x3]:
            ds3[0] = udf_interpolated[y4, x3] / 2
        udf_interpolated[y4+1:y5, x3] = ds3
        interp_mask[y4+1:y5, x3] = True

def get_coords_dist_from_udf(x, y, udf, scaling, pt_matrix):
    y0 = y - 3; y1 = y - 2; y2 = y - 1;
    y3 = y; y4 = y + 1; y5 = y + 2; y6 = y + 3
    d00 = udf[(y0, x)]; d11 = udf[(y1, x)]; d22 = udf[(y2  , x)]
    d33 = udf[(y3, x)]; d44 = udf[(y4, x)]; d55 = udf[(y5  , x)]
    d66 =udf[(y6, x)]
    y0 *= scaling; y1 *= scaling; y2 *= scaling; y3 *= scaling; y4 *= scaling; y5 *= scaling;
    y6 *= scaling

    # get coord index and distance valuce along x-axis
    x0 = x - 3; x1 = x - 2; x2 = x - 1
    x3 = x; x4 = x + 1; x5 = x + 2; x6 = x + 3
    d0 = udf[(y, x0)]; d1 = udf[(y, x1)]; d2 = udf[(y, x2)]
    d3 = udf[(y, x3)]; d4 = udf[(y, x4)]; d5 = udf[(y, x5)]
    d6 = udf[(y, x6)]
    x0 *= scaling; x1 *= scaling; x2 *= scaling; x3 *= scaling; x4 *= scaling; x5 *= scaling;
    x6 *= scaling
    # get real xy coordinates
    x00 = pt_matrix[y3, x0][1]; x11 = pt_matrix[y3, x1][1]; x22 = pt_matrix[y3, x2][1]; 
    x33 = pt_matrix[y3, x3][1]; x44 = pt_matrix[y3, x4][1]; x55 = pt_matrix[y3, x5][1];
    x66 = pt_matrix[y3, x3][1]
    y00 = pt_matrix[y0, x3][0]; y11 = pt_matrix[y1, x3][0]; y22 = pt_matrix[y2, x3][0]; 
    y33 = pt_matrix[y3, x3][0]; y44 = pt_matrix[y4, x3][0]; y55 = pt_matrix[y5, x3][0];
    y66 = pt_matrix[y6, x3][0]

    xs1_ = pt_matrix[y3, x2+1:x3, ...][:,1]
    xs2_ = pt_matrix[y3, x3+1:x4, ...][:,1]
    xs3_ = pt_matrix[y3, x4+1:x5, ...][:,1]
    ys1_ = pt_matrix[y2+1:y3, x3, ...][:,0]
    ys2_ = pt_matrix[y3+1:y4, x3, ...][:,0]
    ys3_ = pt_matrix[y4+1:y5, x3, ...][:,0]

    return x0, x1, x2, x3, x4, x5, x6, y0, y1, y2, y3, y4, y5, y6,\
        d0, d1, d2, d3, d4, d5, d6, d00, d11, d22, d33, d44, d55, d66,\
        x00, x11, x22, x33, x44, x55, x66, y00, y11, y22, y33, y44, y55, y66,\
        xs1_, xs2_, xs3_, ys1_, ys2_, ys3_

def edge_not_safe(y, x, ambiguous_pts, axis):
    pos_not_safe = axis in ambiguous_pts[(y, x)] and 'r' not in ambiguous_pts[(y, x)]
    stroke_not_safe = (axis + '_plus') in ambiguous_pts[(y, x)]
    return pos_not_safe or stroke_not_safe

def get_ambiguouspt_flags(x, y, scaling, interp_mask, ambiguous_pts, reverse_direction = False, pos_mode = True):
    '''
        Given: 
            (x, y), integer as the ambiguous point location
            scaling, the factor to multiply if interpolation is applied
            reverse_drection, boolean to indicate the search direction, the 
                default direction is from top to bottom and left to right.
                if reversed, it will search from bottom to top, right to left
        Return:
            all edge flags that used to tell if there exists strokes around this ambiguous point

    '''
    # get coord index and distance valuce along y-axis, y2, x2 is the point location
    y0 = y - 3; y1 = y - 2; y2 = y - 1;
    y3 = y; y4 = y + 1; y5 = y + 2; y6 = y + 3
    y0 *= scaling; y1 *= scaling; y2 *= scaling; y3 *= scaling; y4 *= scaling; y5 *= scaling;
    y6 *= scaling

    # get coord index and distance valuce along x-axis
    x0 = x - 3; x1 = x - 2; x2 = x - 1
    x3 = x; x4 = x + 1; x5 = x + 2; x6 = x + 3
    x0 *= scaling; x1 *= scaling; x2 *= scaling; x3 *= scaling; x4 *= scaling; x5 *= scaling;
    x6 *= scaling

    # get edge flags
    above_is_safe1 = True
    above_is_safe2 = True
    if (y - 1, x) in ambiguous_pts:
        if edge_not_safe(y - 1, x, ambiguous_pts, 'y') and pos_mode:
            above_is_safe1 = False
    # if 'y' in ambiguous_pts[(y, x)]:
    #      above_is_safe2 = False
    if reverse_direction:
        has_above = (interp_mask[y2+1:y3, x3]).all() and above_is_safe1 and above_is_safe2
    else:
        has_above = (interp_mask[y2+1:y3, x3]).all() and above_is_safe2

    below_is_safe1 = True
    below_is_safe2 = True
    if (y + 1, x) in ambiguous_pts:
        if edge_not_safe(y + 1, x, ambiguous_pts, 'y') and pos_mode:
            below_is_safe1 = False
    # if 'y' in ambiguous_pts[(y, x)]:
    #      below_is_safe2 = False
    if reverse_direction:
        has_below = (interp_mask[y3+1:y4, x3]).all() and below_is_safe2
    else:
        has_below = (interp_mask[y3+1:y4, x3]).all() and below_is_safe1 and below_is_safe2

    left_is_safe1 = True
    left_is_safe2 = True
    if (y, x - 1) in ambiguous_pts:
        if edge_not_safe(y, x - 1, ambiguous_pts, 'x') and pos_mode:
            left_is_safe1 = False
    # if 'x' in ambiguous_pts[(y, x)]:
    #      left_is_safe2 = False
    if reverse_direction:
        has_left = (interp_mask[y3, x2+1:x3]).all() and left_is_safe1 and left_is_safe2
    else:
        has_left = (interp_mask[y3, x2+1:x3]).all() and left_is_safe2

    right_is_safe1 = True
    right_is_safe2 = True
    if (y, x + 1) in ambiguous_pts:
        if edge_not_safe(y, x + 1, ambiguous_pts, 'x') and pos_mode:
            right_is_safe1 = False
    # if 'x' in ambiguous_pts[(y, x)]:
    #      right_is_safe2 = False
    if reverse_direction:
        has_right = (interp_mask[y3, x3+1:x4]).all() and right_is_safe2
    else:
        has_right = (interp_mask[y3, x3+1:x4]).all() and right_is_safe1 and right_is_safe2

    upper_left_is_safe = True
    upper_right_is_safe = True
    if (y - 1, x - 1) in ambiguous_pts:
        if edge_not_safe(y - 1, x - 1, ambiguous_pts, 'x') and pos_mode:
            upper_left_is_safe = False
    if (y - 1, x) in ambiguous_pts:
        if edge_not_safe(y - 1, x, ambiguous_pts, 'x') and pos_mode:
            upper_left_is_safe = False
            upper_right_is_safe = False
    if (y - 1, x + 1) in ambiguous_pts:
        if edge_not_safe(y - 1, x + 1, ambiguous_pts, 'x') and pos_mode:
            upper_right_is_safe = False
    if reverse_direction:
        has_upper_left = (interp_mask[y2, x2+1:x3]).all() and upper_left_is_safe
        has_upper_right = (interp_mask[y2, x3+1:x4]).all() and upper_right_is_safe
    else:
        has_upper_left = (interp_mask[y2, x2+1:x3]).all()
        has_upper_right = (interp_mask[y2, x3+1:x4]).all()

    lower_left_is_safe = True
    lower_right_is_safe = True
    if (y + 1, x - 1) in ambiguous_pts:
        if edge_not_safe(y + 1, x - 1, ambiguous_pts, 'x') and pos_mode:
            lower_left_is_safe = False
    if (y + 1, x) in ambiguous_pts:
        if edge_not_safe(y + 1, x, ambiguous_pts, 'x') and pos_mode:
            lower_left_is_safe = False
            lower_right_is_safe = False
    if (y + 1, x + 1) in ambiguous_pts:
        if edge_not_safe(y + 1, x + 1, ambiguous_pts, 'x') and pos_mode:
            lower_right_is_safe = False
    if reverse_direction:
        has_lower_left = (interp_mask[y4, x2+1:x3]).all()
        has_lower_right = (interp_mask[y4, x3+1:x4]).all()
    else:
        has_lower_left = (interp_mask[y4, x2+1:x3]).all() and lower_left_is_safe
        has_lower_right = (interp_mask[y4, x3+1:x4]).all() and lower_right_is_safe

    left_above_is_safe = True
    left_below_is_safe = True
    if (y - 1, x - 1) in ambiguous_pts:
        if edge_not_safe(y - 1, x - 1, ambiguous_pts, 'y') and pos_mode:
            left_above_is_safe = False
    if (y, x - 1) in ambiguous_pts:
        if edge_not_safe(y, x - 1, ambiguous_pts, 'y') and pos_mode:
            left_above_is_safe = False
            left_below_is_safe = False
    if (y + 1, x - 1) in ambiguous_pts:
        if edge_not_safe(y + 1, x - 1, ambiguous_pts, 'y') and pos_mode:
            left_below_is_safe = False
    if reverse_direction:
        has_left_above = (interp_mask[y2+1:y3, x2]).all() and left_above_is_safe
        has_left_below = (interp_mask[y3+1:y4, x2]).all() and left_above_is_safe
    else:
        has_left_above = (interp_mask[y2+1:y3, x2]).all()
        has_left_below = (interp_mask[y3+1:y4, x2]).all() and left_below_is_safe

    right_above_is_safe = True
    right_below_is_safe = True
    if (y - 1, x + 1) in ambiguous_pts:
        if edge_not_safe(y - 1, x + 1, ambiguous_pts, 'y') and pos_mode:
            right_above_is_safe = False
    if (y, x + 1) in ambiguous_pts:
        if edge_not_safe(y, x + 1, ambiguous_pts, 'y') and pos_mode:
            right_above_is_safe = False
            right_below_is_safe = False
    if (y + 1, x + 1) in ambiguous_pts:
        if edge_not_safe(y + 1, x + 1, ambiguous_pts, 'y') and pos_mode:
            right_below_is_safe = False
    if reverse_direction:
        has_right_above = (interp_mask[y2+1:y3, x4]).all() and right_above_is_safe
        has_right_below = (interp_mask[y3+1:y4, x4]).all()
    else:
        has_right_above = (interp_mask[y2+1:y3, x4]).all() and right_above_is_safe
        has_right_below = (interp_mask[y3+1:y4, x4]).all() and right_below_is_safe

    at_1st = has_upper_right or has_right_above
    at_2nd = has_upper_left or has_left_above
    at_3rd = has_lower_left or has_left_below
    at_4th = has_lower_right or has_right_below

    return has_above, has_below, has_left, has_right, has_upper_left, has_upper_right,\
        has_lower_left, has_lower_right, has_left_above, has_right_above,\
        has_left_below, has_right_below, at_1st, at_2nd, at_3rd, at_4th

def check_stroke_per_grid(interp_mask, x1, x2, x3, x4, y1, y2, y3, y4, clip = False):
    # check if it really connects to its top
    has_left_top = interp_mask[y1+1:y2, x2].all()
    has_right_top = interp_mask[y1+1:y2, x3].all()
    has_top_top = interp_mask[y1, x2+1:x3].all()
    has_top_left_top = interp_mask[y1, x1+1:x2].all()
    has_left_left_top = interp_mask[y1+1:y2, x1].all()
    if clip:
        has_top_stroke = (has_left_top or has_right_top or has_top_top or has_top_left_top or has_left_left_top)
    else:
        has_top_stroke = (has_left_top or has_right_top or has_top_top)
    
    # bottom
    has_left_bottom = interp_mask[y3+1:y4, x2].all()
    has_right_bottom = interp_mask[y3+1:y4, x3].all()
    has_bottom_bottom = interp_mask[y4, x2+1:x3].all()
    has_bottom_right_bottom = interp_mask[y4, x3+1:x4].all()
    has_right_right_bottom = interp_mask[y3+1:y4, x4].all()
    if clip:
        has_bottom_stroke = (has_left_bottom or has_right_bottom or has_bottom_bottom or has_bottom_right_bottom or has_right_right_bottom)
    else:
        has_bottom_stroke = (has_left_bottom or has_right_bottom or has_bottom_bottom)
    
    # left
    has_top_left = interp_mask[y2, x1+1:x2].all()
    has_bottom_left = interp_mask[y3, x1+1:x2].all()
    has_left_left = interp_mask[y2+1:y3, x1].all()
    has_bottom_left_bottom = interp_mask[y4, x1+1:x2].all()
    has_bottom_bottom_left = interp_mask[y3+1:y4, x1].all()
    if clip:
        has_left_stroke = (has_top_left or has_bottom_left or has_left_left or has_bottom_left_bottom or has_bottom_bottom_left)
    else:
        has_left_stroke = (has_top_left or has_bottom_left or has_left_left)
    
    # right
    has_top_right = interp_mask[y2, x3+1:x4].all()
    has_bottom_right = interp_mask[y3, x3+1:x4].all()
    has_right_right = interp_mask[y2+1:y3, x4].all()
    has_top_right_top = interp_mask[y1, x3+1:x4].all()
    has_right_right_top = interp_mask[y1+1:y2, x1].all()
    if clip:
        has_right_stroke = (has_top_right or has_bottom_right or has_right_right or has_top_right_top or has_right_right_top)
    else:
        has_right_stroke = (has_top_right or has_bottom_right or has_right_right)

    return has_left_top, has_right_top, has_top_top, has_top_stroke, \
        has_left_bottom, has_right_bottom, has_bottom_bottom, has_bottom_stroke,\
        has_bottom_stroke, has_bottom_left, has_left_left, has_left_stroke,\
        has_top_right, has_bottom_right, has_right_right, has_right_stroke

def cleanup_invalid_junction(udf, scaling, interp_mask, keep_connection = False, debug = False):
    # for each 2 by 2 grid, search for the t-junction 
    h, w = udf.shape
    hit_false_t_junction = False
    for i in range(1, h - 2):
        for j in range(1, w - 2):
            '''
            For debug
            '''
            # if i == 117 and j == 263 and keep_connection == False:
            #     import pdb
            #     pdb.set_trace()
            x1 = (j - 1) * scaling; x2 = j * scaling; x3 = (j + 1) * scaling; x4 = (j + 2) * scaling
            y1 = (i - 1) * scaling; y2 = i * scaling; y3 = (i + 1) * scaling; y4 = (i + 2) * scaling;
            has_top = interp_mask[y2, x2+1:x3].all()
            has_bottom = interp_mask[y3, x2+1:x3].all()
            has_left = interp_mask[y2+1:y3, x2].all()
            has_right = interp_mask[y2+1:y3, x3].all()

            if np.array([has_top, has_bottom, has_left, has_right]).sum() < 3: continue
            need_cut = np.array([has_top, has_bottom, has_left, has_right]).sum() == 4

            has_left_top, has_right_top, has_top_top, has_top_stroke, \
            has_left_bottom, has_right_bottom, has_bottom_bottom, has_bottom_stroke,\
            has_bottom_stroke, has_bottom_left, has_left_left, has_left_stroke,\
            has_top_right, has_bottom_right, has_right_right, has_right_stroke = \
                check_stroke_per_grid(interp_mask, x1, x2, x3, x4, y1, y2, y3, y4, True)

            if has_top and not keep_connection and not has_bottom and not has_top_stroke:    
                interp_mask[y1+1:y2, x2] = False

            if has_bottom and not keep_connection and not has_top and not has_bottom_stroke:
                interp_mask[y3+1:y4, x2] = False

            if has_left and not keep_connection and not has_right and not has_left_stroke:
                interp_mask[y2, x1+1:x2] = False

            if has_right and not keep_connection and not has_left and not has_right_stroke:
                interp_mask[y2, x3+1:x4] = False

            has_left_top, has_right_top, has_top_top, has_top_stroke, \
            has_left_bottom, has_right_bottom, has_bottom_bottom, has_bottom_stroke,\
            has_bottom_stroke, has_bottom_left, has_left_left, has_left_stroke,\
            has_top_right, has_bottom_right, has_right_right, has_right_stroke = \
                check_stroke_per_grid(interp_mask, x1, x2, x3, x4, y1, y2, y3, y4)

            strokes = np.array([has_top_stroke, has_bottom_stroke, has_left_stroke, has_right_stroke]).sum()
            no_stroke = strokes == 0
            one_stroke = strokes == 1
            two_stroke = strokes == 2
            three_stroke = strokes == 3

            skip_bottom = False
            if has_top:    
                if not has_top_stroke:
                    interp_mask[y2, x2+1:x3] = False
                    hit_false_t_junction = True
                elif one_stroke:
                    skip_bottom = True
            
            if has_bottom and not skip_bottom:
                if not has_bottom_stroke:
                    interp_mask[y3, x2+1:x3] = False
                    hit_false_t_junction = True
                elif has_top and keep_connection:
                    interp_mask[y2, x2+1:x3] = True

            skip_right = False
            if has_left:
                if not has_left_stroke:
                    interp_mask[y2+1:y3, x2] = False
                    hit_false_t_junction = True
                elif one_stroke:
                    skip_right = True

            if has_right and not skip_right:
                if not has_right_stroke:
                    interp_mask[y2+1:y3, x3] = False
                    hit_false_t_junction = True
                elif has_left and keep_connection:
                    interp_mask[y2+1:y3, x2] = True
            if debug and hit_false_t_junction:
                import pdb
                pdb.set_trace()
    return hit_false_t_junction

def crop_by_size(img, size):
    '''
    Given,
        img, a numpy array as the input image
        size, a tuple as the target cropping size (height, width)
    Action,
        crop the image by drop the right and bottom region of the image, the drop size depends on the given size
    '''
    assert len(img.shape) == 2
    h, w = size
    return img[:h, :w]

def pad_by_window_size(img, size):
    # todo: may be we should not padding, and we could change the windows size if necessary
    assert len(img.shape) == 2
    h, w = img.shape
    # this should be different!
    # the determinant for wether pad the image should be:
    # (h - size) % (size - 1) == 0
    # (w - size) % (size - 1) == 0
    # need to update the code here!
    h_round = h // size * size + size if h % size != 0 else h
    w_round = w // size * size + size if w % size != 0 else w
    if h_round != h or w_round != w:
        # create new image
        img_padded = np.zeros((h_round, w_round))
        img_padded[:h, :w] = img
        # pad rows
        if h_round != h:
            for h_delta in range(h_round - h):
                img_padded[h+h_delta, :w] = img_padded[h-h_delta-1, :w]
        if w_round != w:
            for w_delta in range(w_round - w):
                img_padded[:h_round, w+w_delta] = img_padded[:h_round, w-w_delta-1]
        return img_padded
    else:
        return img

def dual_contouring(udf, win_size = 5, interpolation = -1):
    '''
    Given,
        udf, a 2d array as unsigned distance field
        win_size, a int as the grid size
        interpolation, please turn on this option if the input udf is computed 
            from pixel maps by funcstions like cv2.distanceTransform, this function
            will try to interpolate the gradient before sampling the grid, which could 
            give correct result at the sharp and narrow corners.
    Return,
        lines_pt, 
        lines_idx, 
        grids

    Some notes:
        Okay I think get this algorithm work at the theoretically correct is really diffiuclt, we should not waste more time on imporving it.
        But, having a deterministic version DC algorithm working on sketches is still interesting if we can push the algorithm to the resolution 
        limite of the unsigned distance field.
        And writing a complex algorithm without a comprehensive think ahead is really a bad idea. The problem grows faster than your coding logic
        and this quickly becomes very difficult to track your logic when you debug.
    '''

    ## init grid from the udf
    edge_h = (udf.shape[0] - win_size) // (win_size - 1) + 1
    edge_w = (udf.shape[1] - win_size) // (win_size - 1) + 1
    interpolation = 0 if interpolation <= 0 else interpolation
    scaling = interpolation + 1
    '''
        init edge maps, the edge map will be like:
        A     D

              ↑ edge_map_r
        B   ← C 
         edge_map_d
    '''
    edge_map_r = np.zeros((edge_h, edge_w, 1), dtype = bool) # right
    edge_map_d = np.zeros((edge_h, edge_w, 1), dtype = bool) # down
    # create grid from UDF
    UDF_ZERO = 0.5 / scaling
    
    grids, udf_grad, grids_size, pt_matrix, interp_mask = \
        initial_grid(udf, win_size, udf_zero = UDF_ZERO, debug = False, interpolation = interpolation)
    assert edge_h == len(grids)
    assert edge_w == len(grids[0])
    assert interp_mask.shape == udf_grad.shape[:2]
    ## iterate over each grid and record the right
    lines_pt = [] # stroe line segments
    lines_idx = []
    '''
        the grid setting is:
        A  AD  D
        ________
        |      |
      AB|      |CD
        |______|
        B  BC  C
    '''
    # reconstruction version 1
    if interpolation == 0:
        for i in range(grids_size[0]): # height
            for j in range(grids_size[1]): # width
                
                '''for debug'''
                # if i == 85 and j == 12:
                #     import pdb
                #     pdb.set_trace()
                #     udf_debug = get_grid_from_udf(udf_, grids[i][j]['coord_index'])
                '''for debug'''
                
                # check right line
                if j == grids_size[1] - 1: # at the right most
                    has_r = False
                else:
                    has_r = detect_right(i, j, grids, interp_mask, udf_grad, UDF_ZERO)
                    cen_pt_r = grids[i][j + 1]['center_point']
            
                # check bottom line
                if i == grids_size[0] - 1:
                    has_b = False
                else:
                    has_b = detect_bottom(i, j, grids, interp_mask, udf_grad, UDF_ZERO)
                    cen_pt_b = grids[i + 1][j]['center_point']
                
                if has_r and (grids[i][j]['center_point'], cen_pt_r) not in lines_pt:
                    lines_pt.append((grids[i][j]['center_point'], cen_pt_r))
                    lines_idx.append(((i, j),(i, j + 1)))
                    edge_map_r[i][j] = True
            
                if has_b and (grids[i][j]['center_point'], cen_pt_b) not in lines_pt:
                    lines_pt.append((grids[i][j]['center_point'], cen_pt_b))
                    lines_idx.append(((i, j),(i + 1, j)))
                    edge_map_d[i][j] = True

            
        # seems return line index seems 
        edge_maps = np.concatenate((edge_map_r, edge_map_d), axis = -1)
    else:
        for i in range(grids_size[0]): # height
            for j in range(grids_size[1]): # width
                '''
                For debug
                '''
                # if i == 17 and j == 25:
                #     import pdb
                #     pdb.set_trace()
                sub_grids = grids[i][j]['perpixel_grid']
                if sub_grids is None: continue
                h, w = sub_grids.shape[0], sub_grids.shape[1]
                for m in range(h):
                    for n in range(w):
                        stroke_on_edges = sub_grids[m][n]
                        has_strokes = stroke_on_edges[:, 0] != -1
                        assert (has_strokes == (stroke_on_edges[:, 1] != -1)).all()
                        hit_num = has_strokes.sum()
                        if hit_num == 1: continue
                        if hit_num == 2:
                            # connect the two stroke point
                            start_pt = tuple(stroke_on_edges[has_strokes][0])
                            end_pt = tuple(stroke_on_edges[has_strokes][1])
                            lines_pt.append((start_pt, end_pt))
                        if hit_num == 3:
                            idxs = np.where(has_strokes)[0]
                            # has next but not previous
                            if idxs[1] == (idxs[0] + 1) % 4 and idxs[0] != (idxs[-1] + 1) % 4:
                                start_idx = 0 
                            # has ext and previous
                            elif idxs[1] == (idxs[0] + 1) % 4 and idxs[0] == (idxs[-1] + 1) % 4:
                                start_idx = 2
                            else:
                                # assert don't have next
                                assert idxs[1] != (idxs[0] + 1) % 4
                                start_idx = 1
                            start_pt1 = tuple(stroke_on_edges[idxs[start_idx]])
                            end_pt1 = tuple(stroke_on_edges[idxs[(start_idx + 1) % 3]])
                            start_pt2 = tuple(stroke_on_edges[idxs[(start_idx + 1) % 3]])
                            end_pt2 = tuple(stroke_on_edges[idxs[(start_idx + 2) % 3]])
                            lines_pt.append((start_pt1, end_pt1))
                            lines_pt.append((start_pt2, end_pt2))
                        if hit_num == 4: # x-junctions, but we will not reconstruct this topology!
                            pt0 = tuple(stroke_on_edges[0])
                            pt1 = tuple(stroke_on_edges[1])
                            pt2 = tuple(stroke_on_edges[2])
                            pt3 = tuple(stroke_on_edges[3])
                            line1 = (pt1, pt3)
                            line2 = (pt0, pt2)
                            lines_pt.append(line1)
                            lines_pt.append(line2)
        edge_maps = None # we don't consider the edge map for now, but what could we do for that?

    return lines_pt, edge_maps, grids, udf_grad, pt_matrix

def dist(pt1, pt2):
    return abs(complex(*pt1) - complex(*pt2))

def get_edge_range(ha, wa, hb, wb, no_endpoint = False):
    # convert edge coord to index that could be used slicing 
    if ha == hb: # if got the vertical edge
        hs = ha # height start
        he = ha + 1 # height end
        ws = wa if wa < wb else wb # width start
        we = wa if wa > wb else wb # width end
        if no_endpoint:
            ws += 1
        else:
            we += 1
    if wa == wb: # if got the horizontal edge
        hs = ha if ha < hb else hb # height start
        he = ha if ha > hb else hb # height end
        if no_endpoint:
            hs += 1
        else:
            he += 1
        ws = wa # width start
        we = wa + 1 # width end
    return hs, he, ws, we

def detect_stroke_on_edge(ver_a, ver_b, interp_mask, udf_grad, udf_zero = 0):
    '''tell if there exists a stroke across the edge ver_a to ver_b'''
    # get coordinations of the edge
    h, w = interp_mask.shape[0], interp_mask.shape[1]
    ha, wa = ver_a # height a, width a 
    hb, wb = ver_b # height b, width b
    # if we have reached to the boundary of the image, it is no necessary to detect edge
    if ha == hb and (ha == h-1 or hb == h-1): return False, False
    if wa == wb and (wa == w-1 or wb == w-1): return False, False
    
    # get all gradients and distance values along the edge    
    hs, he, ws, we = get_edge_range(ha, wa, hb, wb)
    grads = np.squeeze(udf_grad[hs:he, ws:we])
    hs, he, ws, we = get_edge_range(ha, wa, hb, wb, no_endpoint = True)
    has_stroke = np.squeeze(interp_mask[hs:he, ws:we]).any()

    # record if there exists gradient direction flipping along this edge
    flip_counter = 0
    '''
        record current direction relation, True is the same and False is the opposite
        this status should start as True, so it produces result that always wouldn't be 
        worse than compare the gradient at two grid vertices directly!
    '''
    dir_status = True 
    for i in range(1, len(grads)):
        dp = np.dot(grads[0], grads[i])
        assert dp != 0
        if (dp > 0 and dir_status == False) or \
            (dp < 0 and dir_status == True):
            dir_status = not dir_status
            flip_counter += 1
    return flip_counter > 0 and has_stroke

def detect_stroke_on_edge_(ver_a, ver_b, udf, udf_grad, direction, udf_zero = 0):
    '''tell if there exists a stroke across the edge ver_a to ver_b'''
    # only compare x or y direction seems not good
    # if direction == 'x':
    #     grad_a = udf_grad[ver_a][1]
    #     grad_b = udf_grad[ver_b][1]
    # if direction == 'y':
    #     grad_a = udf_grad[ver_a][0]
    #     grad_b = udf_grad[ver_b][0]
    # return (grad_a * grad_b) < 0

    grad_a = udf_grad[ver_a]
    grad_b = udf_grad[ver_b]
    return np.dot(grad_a, grad_b) < 0


def detect_right(i, j, grids, interp_mask, udf_grad, udf_zero):
    # detect if there is a stroke to the right neighbour grid
    A, B, C, D = grids[i][j]['coord_index']
    has_r = detect_stroke_on_edge(C, D, interp_mask, udf_grad, udf_zero)
    return has_r

def detect_bottom(i, j, grids, interp_mask, udf_grad, udf_zero):
    # get corner coordination of current grid
    A, B, C, D = grids[i][j]['coord_index']
    has_b = detect_stroke_on_edge(B, C, interp_mask, udf_grad, udf_zero)
    return has_b

def average_pts(pts, pt_matrix):
    x, y = np.split(np.array(pts), 2, axis = 1)
    pts = pt_matrix[(x, y)].squeeze()
    h = pts[:, 0]
    w = pts[:, 1]
    return h.mean(), w.mean()

def get_grid_from_udf(udf, pos):
    tl, _, br, _ = pos
    return udf[tl[0]:br[0]+1, tl[1]:br[1]+1]

def ud_to_sd(ud_values, ud_grad_dirs, idx):
    udv = ud_values[idx]
    ud_dir_anchor = ud_grad_dirs[0]
    ud_dir = ud_grad_dirs[idx]
    return sgn(dot(ud_dir_anchor, ud_dir)) * udv

def sgn(x):
    if x > 0: 
        return 1
    else:
        return -1
    
def dot(vec1, vec2):
    assert len(vec1) == 2
    assert len(vec2) == 2
    return vec1[0] * vec2[0] + vec1[1] * vec2[1]

def grid_to_svg(grid, svg_size):
    paths = Path()
    for row in grid:
        for sqaure in row:
            if sqaure[3] and len(sqaure[2]) > 0:
                for pts in sqaure[2]:
                    start = complex(*(pts[0][1], pts[0][0]))
                    end = complex(*(pts[1][1], pts[1][0]))
                    paths.append(Line(start, end))
    wsvg(paths, colors = 'r'*len(paths), stroke_widths = [0.5]*len(paths), dimensions = (svg_size[1], svg_size[0]), filename = "output.svg")
    # wsvg(paths, filename = "output.svg")

def seg_to_svg(seg, svg_size, save_path, grids = None, gradient = None, pt_matrix = None):
    '''
    Given:
        seg, n x 2 x 2 list as the segment start and end points
        svg_size, the canvas size of svg
        save_path, the save path to the svg file
        grids, the grids generated from dual contouring function, 
            if presented, this function will draw red grids upon the svg image,
            which will be convinient for debug
    Action:
        generate svg file from the given parameters
    '''
    P = []
    D = []
    C = ['green']
    W = [0.5]
    paths = Path()
    G = Path()
    for s in seg:
        start = complex(*T(s[0]))
        end = complex(*T(s[1]))
        paths.append(Line(start, end))
    P.append(paths)
    if grids is not None:
        cell_lines = Path()
        # draw grid lines
        for i in range(len(grids)):
            # get the start point of the this
            start = complex(*T(pt_matrix[grids[i][0]['coord_index'][0]])) # A
            end = complex(*T(pt_matrix[grids[i][-1]['coord_index'][-1]])) + complex(1, 0) # D
            cell_lines.append(Line(start, end))
            if i == len(grids) - 1:
                start = complex(*T(pt_matrix[grids[i][0]['coord_index'][1]])) + complex(0, 1) # B
                end = complex(*T(pt_matrix[grids[i][-1]['coord_index'][2]])) + complex(1, 1) # C
                cell_lines.append(Line(start, end))
        for j in range(len(grids[0])):

            start = complex(*(T(pt_matrix[grids[0][j]['coord_index'][0]]))) # A
            end = complex(*(T(pt_matrix[grids[-1][j]['coord_index'][1]]))) + complex(0, 1)
            cell_lines.append(Line(start, end))
            if j == len(grids[0]) - 1:
                start = complex(*(T(pt_matrix[grids[0][j]['coord_index'][-1]]))) + complex(1, 0)
                end = complex(*(T(pt_matrix[grids[-1][j]['coord_index'][2]]))) + complex(1, 1)
                cell_lines.append(Line(start, end))

        # add point and gradient direction
        for i in range(len(grids)):
            for j in range(len(grids[0])):
                # draw grid center points
                D.append(complex(*T(grids[i][j]['center_point'])))
                ver_c_idx = grids[i][j]['coord_index'][2]
                ver_c = T(pt_matrix[ver_c_idx])
                grad = norm(gradient[ver_c_idx])
                start = complex(*ver_c)
                end = start + complex(*T(grad))
                G.append(Line(start = start, end = end))
        P.append(cell_lines)
        C.append('red')
        W.append(0.1)
        P.append(G)
        C.append('cyan')
        W.append(0.1)

    wsvg(P, colors = C, stroke_widths = W, dimensions = (svg_size[1], svg_size[0]), 
            nodes = D, node_colors = ['purple'] * len(D), node_radii = [0.1] * len(D),  filename = save_path)

def T(pt):
    return pt[1], pt[0]

def norm(grad):
    return grad / np.linalg.norm(grad)

def roll(M, shift, axis):
    assert len(M.shape) == 2 # assert M is a 2D matrix
    h, w = M.shape
    ws = 0
    we = w
    hs = 0
    he = h
    xs = np.linspace(0, w - 1, w) # i
    ys = np.linspace(0, h - 1, h) # j
    points = (ys, xs)
    values = M
    # generate new values
    assert axis == 0 or axis == 1
    assert shift != 0
    if axis == 1:
        if shift < 0:
            xs_new = np.linspace(0, w - 1 + abs(shift), w + abs(shift))
        else:
            xs_new = np.linspace(-shift, w - 1, w + shift)
            ws = shift
            we = w + shift
        ys_new = ys
    else:
        if shift < 0:
            ys_new = np.linspace(0, h - 1 + abs(shift), h + abs(shift))
        else:
            ys_new = np.linspace(-shift, h - 1, h + shift)
            hs = shift
            he = h + shift
        xs_new = xs
    point = np.stack(np.meshgrid(ys_new, xs_new, indexing = 'ij'), axis = -1)
    M_new = interpn(points, values, point.reshape((-1, 2)), bounds_error = False, fill_value = None).reshape((point.shape[0], point.shape[1]))
    M_new = np.roll(M_new, shift = shift, axis = axis)[hs:he, ws:we]
    return M_new

# compute the gradient of UDF
def gradient(udf, remove_zeors = True):
    # x_roll = np.roll(udf, shift = -1, axis = 1)
    # x_roll[:, -1] = udf[:, -1]
    x_roll = roll(udf, -1, 1)
    # y_roll = np.roll(udf, shift = -1, axis = 0)
    # y_roll[-1, :] = udf[-1, :]
    y_roll = roll(udf, -1, 0)
    x_grad = x_roll - udf
    y_grad = y_roll - udf
    grad = np.stack((y_grad, x_grad), axis = -1)
    zero_mask = (grad == 0) #.astype(float)
    # remove all zero gradient 
    if zero_mask.sum() > 0 and remove_zeors:
        h, w = udf.shape
        eps = np.arange(0, (h+1) * (w+1)).reshape((h+1, w+1))
        eps_x_roll = np.roll(eps, shift = -1, axis = 1)
        eps_y_roll = np.roll(eps, shift = -1, axis = 0)
        eps_x_grad = eps_x_roll[:h, :w] - eps[:h, :w]
        eps_y_grad = eps_y_roll[:h, :w] - eps[:h, :w]
        eps_grad = np.stack((eps_y_grad, eps_x_grad), axis = -1)
        grad[ zero_mask ] = eps_grad[ zero_mask ] # grad * (1 - zero_mask) + eps_grad * zero_mask 
    return grad

def interp_picewise(x, f1, f2, intersect_pt):
    xs1 = x[x <= intersect_pt]
    xs2 = x[x > intersect_pt]
    if len(xs1) == 0:
        return f2(xs2)
    elif len(xs2) == 0:
        return f1(xs1)
    else:
        return np.concatenate((f1(xs1), f2(xs2)), axis = 0)


def initial_grid(udf, size, key_pts = None, udf_zero = 0.5, debug = False, interpolation = 0, merge_size = 5):
    '''
    Given:
        udf, 2d array with shape (h, w) as the input unsigned distance field
        size, int as the grid shape (size, size), but the grid size at the last row/column may change
        udf_zero, float as the stroke distance threshold, any position that has distance lower than this threshold means there is a stroke passing around
        debug, will force the center point as the key point coords for each grid if turned on, this will be useful when debugging this code
        interpolation, the interpolation level
        merge_size, int as the merge threshold, if the rest grid size at the last row/column is smaller than merge_size, then merge it to the second last row/column grid
    Return:
        grids, a 2d list of grids, each item in this list means a grid for both marching cube or dual contouring

    '''
    global scaling
    # init
    eps = 0.1
    grids = []
    h, w = udf.shape # record the original size
    y = np.linspace(0, h - 1, h) + 0.5 # row index (i)
    x = np.linspace(0, w - 1, w) + 0.5 # column indexn (j)
    
    def push_ambiguity_pts(y, x, ambiguous_pts, axis):
        if DS(y, x) not in ambiguous_pts:
            ambiguous_pts[DS(y, x)] = [axis]
        else:
            if axis not in ambiguous_pts[DS(y, x)]:
                ambiguous_pts[DS(y, x)].append(axis)
    # interpolate the udf if we need to do so
    if interpolation > 0:
        interpolation = int(interpolation)   
        grid_axis = (y, x)
        '''
            by interpolating as this, we could make sure we won't change the index at level 0 no matter 
            how many points are added.
            This will gives us the perfect fit for the scaling for all following operations
        '''
        h_rescal = h + interpolation * (h - 1)
        ys = np.linspace(0, h - 1, h_rescal) + 0.5 # row index
        w_rescal = w + interpolation * (w - 1)
        xs = np.linspace(0, w - 1, w_rescal) + 0.5 # column index
        ys, xs = np.meshgrid(ys, xs, indexing = 'ij')
        pt_matrix = np.stack((ys, xs), axis =  -1)
        udf_interpolated = interpn(grid_axis, udf, pt_matrix.reshape((-1, 2)), bounds_error = False, fill_value = None).reshape((h_rescal, w_rescal))
    else:
        interpolation = 0 # set to level 0
        ys, xs = np.meshgrid(y, x, indexing = 'ij') # so the format in pt matrix is (y, x), remember to reverse it before use it
        pt_matrix = np.stack((ys, xs), axis = -1)
    
    scaling = interpolation + 1 # the scaling factor
    # find the last row / column that could exactly fit the grid
    last_row = h - (h - size) % (size - 1)
    last_col = w - (w - size) % (size - 1)
    
    # compute the gradient on the original udf, which will be used next
    udf_grad = gradient(udf)
    udf_grad_ = udf_grad.copy()

    '''
    the sequence of 4 corners: A, B, C, D
    A     D
    _______
    |     |
    |     |
    |_____|
    B     C
    '''
    # init each grid, 
    h_rest = (h - 1) % (size - 1)
    merge_h = h_rest < merge_size
    h_end = h - h_rest if merge_h else h

    w_rest = (w - 1) % (size - 1)
    merge_w = w_rest < merge_size
    w_end = w - w_rest if merge_w else w

    # 1st pass init grids
    for i in np.arange(0, h_end, size -1):
        if i == h - 1: break # we have reached the last row
        if i == h_end -1 and merge_h: break # we have merged the current grid to its upper row
        if i == h_end - size and merge_h:
            is_last_row = True
        else:
            is_last_row = False
        grids_row = []
        for j in np.arange(0, w_end, size - 1):
            if j == w - 1: break # we have reached the last column
            if j == w_end - 1 and merge_w: break
            if j == w_end - size and merge_w:
                is_last_col = True
            else:
                is_last_col = False
            grid = {}

            # top left coord
            A = (i * scaling, j * scaling)
            A_org = (i, j)
            # revise bottom and right coordinates
            if is_last_row: # merge and at the last row
                i_bt = h - 1
            else:
                i_bt = i + size - 1 # not merge / merge and not at the last row
                if i_bt >= h: # not merge and at the last row, now all 4 possible cases are covered
                        i_bt = h -1
            
            if is_last_col:
                j_rt = w - 1
            else:
                j_rt = j + size - 1
                if j_rt >= w:
                    j_rt = w -1
            # bottom left
            B = (i_bt * scaling, j * scaling)
            B_org = (i_bt, j) 
            # bottom right
            C = (i_bt * scaling, j_rt * scaling)
            C_org = (i_bt, j_rt)
            # top right
            D = (i * scaling, j_rt * scaling)
            D_org = (i, j_rt)
            # append position
            grid['coord_index'] = (A, B, C, D)
            grid['coord_index_org'] = (A_org, B_org, C_org, D_org)
            # push each grid into grid row
            grids_row.append(grid)

        grids.append(grids_row)
    
    # get grid size
    grids_h = len(grids)
    grids_w = len(grids[0])
    
    # 2nd pass, find the edge that has the gradient direction change, then
    # extrapolate the value along that edge and update the gradient 
    if interpolation > 0:
        udf_interpolated_  = udf_interpolated.copy()
        interp_mask = np.zeros(udf_interpolated.shape).astype(bool) # might be not useful
        inter_task_list = []
        # record the pixel location that has the gradient changes
        pt_y, pt_x = find_flip_loc(udf_grad, udf) # this line is correct
        ambiguous_pts = {}
        # extrapolate the pixel value at each edge along y axis
        skip_list = []
        for pt in pt_y:
            y, x = pt
            if y < 2 or y > h - 7: continue # skip points that too close to top or bottom
            if (y, x) in skip_list: continue
            '''
            For Debug
            '''
            # if y == 185 and x == 94:
            #     import pdb
            #     pdb.set_trace()
            is_marked = False
            y0 = y - 2; y1 = y - 1; y2 = y; y3 = y + 1; y4 = y + 2; y5 = y + 3
            y6 = y + 4; y7 = y + 5 
            d0 = udf[(y0, x)]; d1 = udf[(y1, x)]; d2 = udf[(y2  , x)]
            d3 = udf[(y3, x)]; d4 = udf[(y4, x)]; d5 = udf[(y5  , x)]
            d6 = udf[(y6, x)]; d7 = udf[(y7, x)]
            y0 *= scaling; y1 *= scaling; y2 *= scaling; y3 *= scaling; y4 *= scaling; y5 *= scaling;
            y6 *= scaling; y7 *= scaling
            x2 = x * scaling
            y00 = pt_matrix[y0, x2][0]; y11 = pt_matrix[y1, x2][0]; 
            y22 = pt_matrix[y2, x2][0]; y33 = pt_matrix[y3, x2][0]; 
            y44 = pt_matrix[y4, x2][0]; y55 = pt_matrix[y5, x2][0]; 
            y66 = pt_matrix[y6, x2][0]; y77 = pt_matrix[y7, x2][0]
            
            has_been_marked = (interp_mask[y2+1:y3, x2]).all()
            above_been_marked = (interp_mask[y1+1:y2, x2]).all()
            f1_is_safe = False
            # we will skip no matter if this edge has stroke or not, this exceed the algorithm resolution
            if (above_been_marked or has_been_marked) and not is_in(pt_y, DS(y3, x2)) and d3 < d4:
                f1 = interp1d([y33, y44], [d3, d4], kind = 'linear', fill_value = 'extrapolate')
                f1_is_safe = True
            elif (above_been_marked or has_been_marked):
                # the f1 should not be used
                if has_been_marked and not above_been_marked and d1 > d2:
                    f1 = interp1d([y11, y22], [d1, d2], kind = 'linear', fill_value = 'extrapolate')
                    f1_is_safe = True
                elif above_been_marked:
                    if d2 < d3:
                        f1 = interp1d([y22, y33], [d2, d3], kind = 'linear', fill_value = 'extrapolate')
                    else:
                        y_zero, _ = find_zeropt(y11, y22, -d1, d2)
                        if abs(y11 - y_zero) > abs(y_zero - y22):
                            f1 = interp1d([y11, y_zero], [-d1, 0], kind = 'linear', fill_value = 'extrapolate')
                        else:
                            f1 = interp1d([y_zero, y22], [0, d2], kind = 'linear', fill_value = 'extrapolate')
            else:
                f1 = interp1d([y11, y22], [d1, d2], kind = 'linear', fill_value = 'extrapolate')
                f1_is_safe = True

            f2 = interp1d([y33, y44, y55], [d3, d4, d5], kind = 'linear', fill_value = 'extrapolate')
            f3 = interp1d([y44, y55], [d4, d5], kind = 'linear', fill_value = 'extrapolate')
            ys1 = pt_matrix[y2:y3+1, x2, ...][:,0]
            ys1_ = pt_matrix[y2+1:y3, x2, ...][:,0]
            ys2 = pt_matrix[y3:y4+1, x2, ...][:,0]
            ys2_ = pt_matrix[y3+1:y4, x2, ...][:,0]
            ys3_ = pt_matrix[y4+1:y5, x2, ...][:,0]
            ys4_ = pt_matrix[y5+1:y6, x2, ...][:,0]

            ext_f1ys1 = f1(ys1)
            ext_f1ys2 = f1(ys2)

            ext_f1ys1_may_wrong = d3 < eps
            at_ys1 = (ext_f1ys1 < 0).any() or ((d2 + d3) <= 1 and (not d2d3_is_local_maximum(d2, d3, d4, d5) or d2tod5_all_small(d2, d3, d4, d5)))

            parallel = is_parallel(udf_interpolated, x, y, scaling, 0)
            if not f1_is_safe and (ext_f1ys1 > 0).all() and (ext_f1ys2 > 0).all() and not d2d3_is_local_minimum(d1, d2, d3, d4) and not parallel: continue 
            elif has_been_marked: pass   
            elif ext_f1ys1_may_wrong: 
                if DS(y3, x2) not in ambiguous_pts:
                    ambiguous_pts[DS(y3, x2)] = ['y']
                else:
                    ambiguous_pts[DS(y3, x2)].append('y')
                udf_interpolated[y2+1:y3, x2] = udf_interpolated_[y2+1:y3, x2]
                udf_interpolated[y3+1:y4, x2] = udf_interpolated_[y3+1:y4, x2]
                interp_mask[y2+1:y3, x2] = True
                interp_mask[y3+1:y4, x2] = True
                continue
            elif not f1_is_safe and (ext_f1ys1 > 0).all() and (ext_f1ys2 < 0).any() and (ext_f1ys2 > 0).any() and (d3 + d4) <= 1 and \
                (d2+d3) > (d3+d4):
                ds1 = f1(ys2_)
                udf_interpolated[y3+1:y4, x2] = ds1
                if (ds1 < 0).all():
                    udf_interpolated[y3+1:y4, x2] = ds1
                    udf_interpolated[y3+1, x2] = eps / 2
                else:
                    udf_interpolated[y3+1:y4, x2] = ds1
                interp_mask[y3+1:y4, x2] = True
                # however, this could also be wrong if this flag is between two very close strokes
                if (d2 + d3) <= 1 and (d1 + d2) <= 1:
                    interp_mask[y2+1:y3, x2] = True
                    push_ambiguity_pts(y3, x2, ambiguous_pts, 'y')
            elif at_ys1:
                _, f0 = find_zeropt(ys1[0], ys1[-1], d2, -d3)
                ds1 = f0(ys1_)
                if (ds1 < 0).all():
                    udf_interpolated[y2+1:y3, x2] = ds1
                    udf_interpolated[y2+1, x2] = eps / 2
                else:
                    udf_interpolated[y2+1:y3, x2] = ds1
                interp_mask[y2+1:y3, x2] = True
                if (d2 + d3) <= 1 and (d1 + d2) <= 1 and above_been_marked and d3 < d4:
                    if d2 < eps:
                        push_ambiguity_pts(y2, x2, ambiguous_pts, 'y')
                    else:
                        push_ambiguity_pts(y2, x2, ambiguous_pts, 'y_plus')
                if (d2 + d3) <= 1 and (d3 + d4) <= 1 and d4 < d5:
                    if d3 < eps:
                        push_ambiguity_pts(y3, x2, ambiguous_pts, 'y')
                    else:
                        push_ambiguity_pts(y3, x2, ambiguous_pts, 'y_plus')
                    interp_mask[y3+1:y4, x2] = True
                is_marked = True

            # if (y2, y3) has been identified as containing stroke in previous step, 
            if has_been_marked or is_marked:
                if is_marked and has_been_marked == False:
                    y_zero1, _ = find_zeropt(y22, y33, -d2, d3)
                    if abs(y22 - y_zero1) > abs(y_zero1 - y33):
                        f1 = interp1d([y22, y_zero1], [-d2, 0], kind = 'linear', fill_value = 'extrapolate')
                    else:
                        f1 = interp1d([y_zero1, y33], [0, d3], kind = 'linear', fill_value = 'extrapolate')

                # case 1: stroke is in y3, y4
                if not is_in(pt_y, DS(y4, x2)) and is_in(pt_y, DS(y3, x2)) and (d4 < d5) and (d3 + d4) <= 1:
                    f0 = interp1d([y44, y55], [d4, d5], kind = 'linear', fill_value = 'extrapolate')
                    y3y4 = f0(pt_matrix[y3:y4, x2][:, 0])
                    if (y3y4 > 0).all() and d3 > eps: continue
                    y_zero2, _ = find_zeropt(y44, y55, d4, d5)
                    # if y_zero2 > y33 or d3 < eps:
                    #     f2 = interp1d([y_zero2, y44], [0, -d4], kind = 'linear', fill_value = 'extrapolate')
                    #     intersect_pt = find_intersection(y22, y_zero1, -d2, 0, y_zero2, y44, 0, -d4)
                    #     f3 = partial(interp_picewise, f1 = f1, f2 = f2, intersect_pt = intersect_pt)
                    #     udf_interpolated[y2+1:y3, x2] = f3(ys1_)
                    #     udf_interpolated[y3+1:y4, x2] = f3(ys2_)
                    interp_mask[y3+1:y4, x2] = True
                    '''
                        Corner case, if we detect two consecutive strokes in this way, there will be two ambiguity:
                        1. is this really indicate two strokes which every close to each other? or 
                        2. just one stroke but we don't know which side it passes?
                        so we just record this case as "y_plus" here and resovle it later
                    '''
                    if d3 < eps:
                        push_ambiguity_pts(y3, x2, ambiguous_pts, 'y')
                    else:
                        push_ambiguity_pts(y3, x2, ambiguous_pts, 'y_plus')
                        
                # case 2: stroke in y4, y5
                elif is_in(pt_y, DS(y3, x2)) and is_in(pt_y, DS(y4, x2)) and (not is_in(pt_y, DS(y5, x2)) or d5 + d6 < 1) and d5 < d6 and (d2 + d3) <= 1 and (d4 + d5) <= 1 and d3 > eps and d4 > eps:
                    y_zero2, _ = find_zeropt(y55, y66, d5, d6)
                    # if y_zero2 > y44:
                    #     f2 = interp1d([y_zero2, y55], [0, -d5], kind = 'linear', fill_value = 'extrapolate')
                    #     intersect_pt = find_intersection(y22, y_zero1, -d2, 0, y_zero2, y55, 0, -d5)
                    #     f3 = partial(interp_picewise, f1 = f1, f2 = f2, intersect_pt = intersect_pt)
                    #     udf_interpolated[y2+1:y3, x2] = f3(ys1_)
                    #     udf_interpolated[y3+1:y4, x2] = f3(ys2_)
                    #     udf_interpolated[y4+1:y5, x2] = f3(ys3_)
                    interp_mask[y4+1:y5, x2] = True
                    if interp_mask[y3+1:y4, x2].all():
                        if d4 < eps:
                            push_ambiguity_pts(y4, x2, ambiguous_pts, 'y')
                        else:
                            push_ambiguity_pts(y4, x2, ambiguous_pts, 'y_plus')
                    skip_list.append((y+1, x))
                
                # case 3: stroke in y5, y6
                elif is_in(pt_y, DS(y5, x2)) and (d5 + d6) <= 1 and not is_in(pt_y, DS(y6, x2)) and (is_in(pt_y, DS(y3, x2)) or is_in(pt_y, DS(y4, x2))) and d6 < d7 and d4 > d3 and d4 > d5:
                    y_zero2, _ = find_zeropt(y66, y77, d6, d7)
                    # if y_zero2 > y55:
                    #     f2 = interp1d([y_zero2, y66], [0, -d6], kind = 'linear', fill_value = 'extrapolate')
                    #     intersect_pt = find_intersection(y22, y_zero1, -d2, 0, y_zero2, y66, 0, -d6)
                    #     f3 = partial(interp_picewise, f1 = f1, f2 = f2, intersect_pt = intersect_pt)
                    #     udf_interpolated[y2+1:y3, x2] = f3(ys1_)
                    #     udf_interpolated[y3+1:y4, x2] = f3(ys2_)
                    #     udf_interpolated[y4+1:y5, x2] = f3(ys3_)
                    #     udf_interpolated[y5+1:y6, x2] = f3(ys4_)
                    interp_mask[y5+1:y6, x2] = True
                    skip_list.append((y+1, x))
                    skip_list.append((y+2, x))

        # extrapolate the pixel value at each edge along x axis
        skip_list = []
        for pt in pt_x:
            y, x = pt
            if x < 2 or x > w - 6: continue
            if (y, x) in skip_list: continue
            '''
            For Debug
            '''
            # if y == 185 and x == 94:
            #     import pdb
            #     pdb.set_trace()
            is_marked = False
            # get coord index and distance valuce along x-axis
            y2 = y * scaling
            x0 = x - 2; x1 = x - 1; x2 = x; x3 = x + 1; x4 = x + 2; x5 = x + 3
            x6 = x + 4; x7 = x + 5
            d0 = udf[(y, x0)]; d1 = udf[(y, x1)]; d2 = udf[(y, x2)]
            d3 = udf[(y, x3)]; d4 = udf[(y, x4)]; d5 = udf[(y, x5)]
            d6 = udf[(y, x6)]; d7 = udf[(y, x7)];
            x0 *= scaling; x1 *= scaling; x2 *= scaling; x3 *= scaling; x4 *= scaling; x5 *= scaling
            x6 *= scaling; x7 *= scaling
            
            # get real xy coordinates
            x00 = pt_matrix[y2, x0][1]; x11 = pt_matrix[y2, x1][1]; x22 = pt_matrix[y2, x2][1]; 
            x33 = pt_matrix[y2, x3][1]; x44 = pt_matrix[y2, x4][1]; x55 = pt_matrix[y2, x5][1];
            x66 = pt_matrix[y2, x6][1]; x77 = pt_matrix[y2, x7][1]

            # we need addtional check to tell if it is safe to continue the extrapolation
            # if it's left neigbour edge is also included but the distance value at the start point is NOT zero
            # then this mean it is not safe to do any extrapolation! and we should skip this case!
            left_been_marked = (interp_mask[y2, x1+1:x2]).all()
            has_been_marked = (interp_mask[y2, x2+1:x3]).all()
            
            # if we can't give the right extrapolation result, then let's give a definitly wrong result
            # so that will not affect the following logic
            # if the right size is safe, then use the interpolation from the right side
            f1_is_safe = False
            if (left_been_marked or has_been_marked) and not is_in(pt_x, DS(y2, x3)) and d3 < d4:
                f1 = interp1d([x33, x44], [d3, d4], kind = 'linear', fill_value = 'extrapolate')
                f1_is_safe = True
            elif (left_been_marked or has_been_marked):
                if has_been_marked and not left_been_marked and d1 > d2:
                    f1 = interp1d([x11, x22], [d1, d2], kind = 'linear', fill_value = 'extrapolate')
                    f1_is_safe = True
                elif left_been_marked:
                    if d2 < d3:
                        f1 = interp1d([x22, x33], [d2, d3], kind = 'linear', fill_value = 'extrapolate')
                    else:
                        x_zero, _ = find_zeropt(x11, x22, -d1, d2)
                        if abs(x11 - x_zero) > abs(x_zero - x22):
                            f1 = interp1d([x11, x_zero], [-d1, 0], kind = 'linear', fill_value = 'extrapolate')
                        else:
                            f1 = interp1d([x_zero, x22], [0, d2], kind = 'linear', fill_value = 'extrapolate')
            else:
                f1 = interp1d([x11, x22], [d1, d2], kind = 'linear', fill_value = 'extrapolate')
                f1_is_safe = True
            
            xs1 = pt_matrix[y2, x2:x3+1, ...][:,1]
            xs2 = pt_matrix[y2, x3:x4+1, ...][:,1]
            xs1_ = pt_matrix[y2, x2+1:x3, ...][:,1]
            xs2_ = pt_matrix[y2, x3+1:x4, ...][:,1]
            xs3_ = pt_matrix[y2, x4+1:x5, ...][:,1]
            xs4_ = pt_matrix[y2, x5+1:x6, ...][:,1]

            # interpolation results along x-axis
            ext_f1xs1 = f1(xs1)
            ext_f1xs2 = f1(xs2)

            # distance value shows there might be too small to be correct (ambiguous point)
            ext_f1xs1_may_wrong = d3 < eps

            parallel = is_parallel(udf_interpolated, x, y, scaling, 1)
            at_xs1 = (ext_f1xs1 < 0).any() or ((d2 + d3) <= 1 and (not d2d3_is_local_maximum(d2, d3, d4, d5) or d2tod5_all_small(d2, d3, d4, d5)))
            
            # case 1: very flat gradient which indicate no local minimum is found
            if f1_is_safe and (ext_f1xs1 > 0).all() and (ext_f1xs2 > 0).all() and not d2d3_is_local_minimum(d1, d2, d3, d4) and not parallel: continue # skip if the extrapolate result doesn't show a direction flipping
            # case 2: has been marked in previous round, directly jump to next logic stage
            elif has_been_marked: pass
            # case 3: ambiguous point
            elif ext_f1xs1_may_wrong:
                if DS(y2, x3) not in ambiguous_pts:
                    ambiguous_pts[DS(y2, x3)] = ['x']
                else:
                    ambiguous_pts[DS(y2, x3)].append('x')
                # reset values on the edges if there were extrapolated already      
                udf_interpolated[y2, x2+1:x3] = udf_interpolated_[y2, x2+1:x3]
                udf_interpolated[y2, x3+1:x4] = udf_interpolated_[y2, x3+1:x4]
                interp_mask[y2, x2+1:x3] = True
                interp_mask[y2, x3+1:x4] = True
                continue
            
            # case 4: stroke at next x edge
            elif f1_is_safe and (ext_f1xs1 > 0).all() and (ext_f1xs2 < 0).any() and (ext_f1xs2 > 0).any() and (d3 + d4) <= 1 and \
                (d2+d3) > (d3+d4):
                # extrapolate xs2
                ds1 = f1(xs2_)
                if (ds1 < 0).all():
                    udf_interpolated[y2, x3+1:x4] = ds1
                    udf_interpolated[y2, x3+1] = eps /2
                else:
                    udf_interpolated[y2, x3+1:x4] = ds1
                interp_mask[y2, x3+1:x4] = True
                # there should has no ambiguity on either left or right side because
                # the left edge at x3 be identified as false and the right edge at x4
                # hasn't been explored 
                # if (d2 + d3) <= 1 and (d1 + d2) <= 1 and d4 < d5:
                #     interp_mask[y2, x2+1:x3] = True
                #     push_ambiguity_pts(y2, x3, ambiguous_pts, 'x')

            # case 5: stroke at current edge
            elif at_xs1:
                _, f0 = find_zeropt(xs1[0], xs1[-1], d2, -d3)
                ds1 = f0(xs1_)
                if (ds1 < 0).all():
                    udf_interpolated[y2, x2+1:x3] = ds1
                    udf_interpolated[y2, x2+1] = eps / 2
                else:
                    udf_interpolated[y2, x2+1:x3] = ds1
                interp_mask[y2, x2+1:x3] = True
                # that's why we need to add more branches here 
                if (d2 + d3) <= 1 and (d1 + d2) <= 1 and left_been_marked and d3 < d4:
                    if d2 < eps:
                        push_ambiguity_pts(y2, x2, ambiguous_pts, 'x')
                    else:
                        push_ambiguity_pts(y2, x2, ambiguous_pts, 'x_plus')
                if (d2 + d3) <= 1 and (d3 + d4) <= 1 and d4 < d5:
                    if d3 < eps:
                        push_ambiguity_pts(y2, x3, ambiguous_pts, 'x')
                    else:
                        push_ambiguity_pts(y2, x3, ambiguous_pts, 'x_plus')
                    interp_mask[y2, x3+1:x4] = True
                is_marked = True

            # find if there is any stroke very close to the current one
            if has_been_marked or is_marked:
                if is_marked and has_been_marked == False:
                    x_zero1, _ = find_zeropt(x22, x33, -d1, d2)
                    if abs(x22 - x_zero1) > abs(x_zero1 - x33):
                        f1 = interp1d([x22, x_zero1], [-d2, 0], kind = 'linear', fill_value = 'extrapolate')
                    else:
                        f1 = interp1d([x_zero1, x33], [0, d3], kind = 'linear', fill_value = 'extrapolate')
                
                # case 1: stroke is in x3, x4
                if not is_in(pt_x, DS(y2, x4)) and is_in(pt_x, DS(y2, x3)) and (d4 < d5) and (d3 + d4) <= 1:
                    f0 = interp1d([x44, x55], [d4, d5], kind = 'linear', fill_value = 'extrapolate')
                    x3x4 = f0(pt_matrix[y2, x3: x4][:, 1])
                    if (x3x4 > 0).all() and d3 > eps: continue
                    x_zero2, _ = find_zeropt(x44, x55, d4, d5)
                    # if x_zero2 > x33 or d3 < eps:
                    #     f2 = interp1d([x_zero2, x44], [0, -d4], kind = 'linear', fill_value = 'extrapolate')
                    #     intersect_pt = find_intersection(x22, x_zero1, -d2, 0, x_zero2, x44, 0, -d4)
                    #     f3 = partial(interp_picewise, f1 = f1, f2 = f2, intersect_pt = intersect_pt)
                    #     udf_interpolated[y2, x2+1:x3] = f3(xs1_)
                    #     udf_interpolated[y2, x3+1:x4] = f3(xs2_)
                    interp_mask[y2, x3+1:x4] = True
                    if d3 < eps:
                        push_ambiguity_pts(y2, x3, ambiguous_pts, 'x')
                    else:
                        push_ambiguity_pts(y2, x3, ambiguous_pts, 'x_plus')

                # case 2: stroke in x4, x5
                elif is_in(pt_x, DS(y2, x3)) and is_in(pt_x, DS(y2, x4)) and (not is_in(pt_x, DS(y2, x5)) or d5 + d6 < 1) and d5 < d6\
                    and (d2 + d3) <= 1 and (d4 + d5) <= 1 and d3 > eps and d4 > eps:
                    x_zero2, _ = find_zeropt(x55, x66, d5, d6)
                    # if x_zero2 > x44:
                    #     f2 = interp1d([x_zero2, x55], [0, -d5], kind = 'linear', fill_value = 'extrapolate')
                    #     intersect_pt = find_intersection(x22, x_zero1, -d2, 0, x_zero2, x55, 0, -d5)
                    #     f3 = partial(interp_picewise, f1 = f1, f2 = f2, intersect_pt = intersect_pt)
                    #     udf_interpolated[y2, x2+1:x3] = f3(xs1_)
                    #     udf_interpolated[y2, x3+1:x4] = f3(xs2_)
                    #     udf_interpolated[y2, x4+1:x5] = f3(xs3_)
                    interp_mask[y2, x4+1:x5] = True
                    if interp_mask[y2, x3+1:x4].all():
                        if d4 < eps:
                            push_ambiguity_pts(y2, x4, ambiguous_pts, 'x')
                        else:
                            push_ambiguity_pts(y2, x4, ambiguous_pts, 'x_plus')
                    skip_list.append((y, x+1))

                # case 3: stroke in x5, x6
                elif is_in(pt_x, DS(y2, x5)) and (d5 + d6) <= 1 and is_in(pt_x, DS(y2, x6)) == False and\
                    (is_in(pt_x, DS(y2, x3)) or is_in(pt_x, DS(y2, x4))) and d6 < d7 and\
                    d4 > d3 and d4 > d5:
                    x_zero2, _ = find_zeropt(x66, x77, d6, d7)
                    # if x_zero2 > x55:
                    #     f2 = interp1d([x_zero2, x66], [0, -d6], kind = 'linear', fill_value = 'extrapolate')
                    #     intersect_pt = find_intersection(x22, x_zero1, -d2, 0, x_zero2, x66, 0, -d6)
                    #     f3 = partial(interp_picewise, f1 = f1, f2 = f2, intersect_pt = intersect_pt)
                    #     udf_interpolated[y2, x2+1:x3] = f3(xs1_)
                    #     udf_interpolated[y2, x3+1:x4] = f3(xs2_)
                    #     udf_interpolated[y2, x4+1:x5] = f3(xs3_)
                    #     udf_interpolated[y2, x5+1:x6] = f3(xs4_)
                    interp_mask[y2, x5+1:x6] = True
                    skip_list.append((y, x+1))
                    skip_list.append((y, x+2))


        '''
        For debug
        '''
        Image.fromarray(interp_mask).save("../experiments/03.DC/edge.png")
        import pdb
        pdb.set_trace()

        # solve ambiguous distance values
        ambi_pt_keys = list(ambiguous_pts.keys())
        ambi_pt_keys.sort()
        for pt in ambi_pt_keys:
            y, x = pt
            '''
            For Debug
            '''
            # if y == 182 and x == 94:
            #     Image.fromarray(interp_mask).save("../experiments/03.DC/edge.png")
            #     import pdb
            #     pdb.set_trace()
                
            if 'r' in ambiguous_pts[pt]: continue

            # solve the first ambiguity: one stroke or two stroke?
            if 'y_plus' in ambiguous_pts[pt]:
                sovle_y_plus(y, x, udf, scaling, pt_matrix, interp_mask, ambiguous_pts)

            if 'x_plus' in ambiguous_pts[pt]:
                sovle_x_plus(y, x, udf, scaling, pt_matrix, interp_mask, ambiguous_pts)
            
            if len(ambiguous_pts[pt]) == 0: continue

            has_above, has_below, has_left, has_right,\
            has_upper_left, has_upper_right,\
            has_lower_left, has_lower_right,\
            has_left_above, has_right_above,\
            has_left_below, has_right_below,\
            at_1st, at_2nd, at_3rd, at_4th\
                = get_ambiguouspt_flags(x, y, scaling, interp_mask, ambiguous_pts, pos_mode = False)
            # solve the second ambiguity: where is the stroke?
            if 'y' in ambiguous_pts[pt] and 'x' not in ambiguous_pts[pt]:
                ext_above, ext_left = sovle_y(y, x, has_above, has_below, has_left, has_right, 
                    has_upper_left, has_upper_right, has_lower_left, has_lower_right,
                    has_left_below, has_left_above, has_right_below, has_right_above, 
                    at_1st, at_2nd, at_3rd, at_4th)

            elif 'y' not in ambiguous_pts[pt] and 'x' in ambiguous_pts[pt]:
                ext_above, ext_left = sovle_x(y, x, has_above, has_below, has_left, has_right, 
                    has_upper_left, has_upper_right, has_lower_left, has_lower_right, 
                    has_left_below, has_left_above, has_right_below, has_right_above,
                    at_1st, at_2nd, at_3rd, at_4th)
                
            # this would be the most complex part!
            elif 'y' in ambiguous_pts[pt] and 'x' in ambiguous_pts[pt]:
                pt_list = find_consecutive_pts(ambiguous_pts, pt, 1, force_direction = True)
                for i in range(len(pt_list) - 1, -1, -1):
                    yx, xx = pt_list[i]
                    ext_a = None
                    ext_l = None
                    # if the pt's right / below neighbour is also ambiguous point, we need to sovle the neghbour points first
                    has_above, has_below, has_left, has_right,\
                    has_upper_left, has_upper_right,\
                    has_lower_left, has_lower_right,\
                    has_left_above, has_right_above,\
                    has_left_below, has_right_below,\
                    at_1st, at_2nd, at_3rd, at_4th\
                        = get_ambiguouspt_flags(xx, yx, scaling, interp_mask, ambiguous_pts, reverse_direction = True)
                    if 'x' not in ambiguous_pts[(yx, xx)]:
                        if 'y_plus' in ambiguous_pts[(yx, xx)]:
                            sovle_y_plus(yx, xx, udf, scaling, pt_matrix, interp_mask, ambiguous_pts, True, True)
                        if 'y' in ambiguous_pts[(yx, xx)]:
                            ext_a, ext_l = sovle_y(yx, xx, has_above, has_below, has_left, has_right, 
                                has_upper_left, has_upper_right, has_lower_left, has_lower_right, 
                                has_left_below, has_left_above, has_right_below, has_right_above, 
                                at_1st, at_2nd, at_3rd, at_4th)
                    elif 'x' in ambiguous_pts[(yx, xx)]:
                        if 'y_plus' in ambiguous_pts[(yx, xx)]:
                            sovle_y_plus(yx, xx, udf, scaling, pt_matrix, interp_mask, ambiguous_pts, True, True)
                        if 'y' in ambiguous_pts[(yx, xx)]:
                            ext_a, ext_l = solve_xy(at_1st, at_2nd, at_3rd, at_4th, interp_mask, xx, yx, udf, pt_matrix, scaling)        
                    else:
                        raise ValueError("Impossible structure detected, please debug the code!")
                    resolve_ambiguou_pt(ext_a, ext_l, udf_interpolated, interp_mask,\
                        xx, yx, udf, pt_matrix, scaling, ambiguous_pts, pt_x, pt_y)
                    ambiguous_pts[(yx, xx)].append('r')
                    # if 'x' in ambiguous_pts[(yx, xx - 1)] and ext_l == True: 
                    #     interp_mask[y3, x2+1:x3] = False
                    #     ambiguous_pts[(yx, xx - 1)].remove('x')
                
                pt_list = find_consecutive_pts(ambiguous_pts, pt, 0, force_direction = True)
                for i in range(len(pt_list) - 1, -1, -1):
                    yy, xy = pt_list[i]
                    ext_a = None
                    ext_l = None
                    has_above, has_below, has_left, has_right,\
                    has_upper_left, has_upper_right,\
                    has_lower_left, has_lower_right,\
                    has_left_above, has_right_above,\
                    has_left_below, has_right_below,\
                    at_1st, at_2nd, at_3rd, at_4th\
                        = get_ambiguouspt_flags(xy, yy, scaling, interp_mask, ambiguous_pts, reverse_direction = True)
                    if 'y' not in ambiguous_pts[(yy, xy)]:
                        if 'x_plus' in ambiguous_pts[(yy, xy)]:
                            sovle_x_plus(yy, xy, udf, scaling, pt_matrix, interp_mask, ambiguous_pts, True, True)
                        if 'x' in ambiguous_pts[(yy, xy)]:
                            ext_a, ext_l = sovle_x(yy, xy, has_above, has_below, has_left, has_right, 
                                has_upper_left, has_upper_right, has_lower_left, has_lower_right, 
                                has_left_below, has_left_above, has_right_below, has_right_above,
                                at_1st, at_2nd, at_3rd, at_4th)
                    elif 'y' in ambiguous_pts[(yy, xy)]:
                        if 'x_plus' in ambiguous_pts[(yy, xy)]:
                            sovle_x_plus(yy, xy, udf, scaling, pt_matrix, interp_mask, ambiguous_pts, True, True)
                        if 'x' in ambiguous_pts[(yy, xy)]:
                            ext_a, ext_l = solve_xy(at_1st, at_2nd, at_3rd, at_4th, interp_mask, xy, yy, udf, pt_matrix, scaling)
                    resolve_ambiguou_pt(ext_a, ext_l, udf_interpolated, interp_mask,\
                        xy, yy, udf, pt_matrix, scaling, ambiguous_pts, pt_x, pt_y)
                    ambiguous_pts[(yy, xy)].append('r')
                    # if 'y' in ambiguous_pts[(yy - 1, xy)]: ambiguous_pts[(yy - 1, xy)].remove('y')

                # the flags here should be updated again since the surrounding edges have been updated
                has_above, has_below, has_left, has_right,\
                has_upper_left, has_upper_right,\
                has_lower_left, has_lower_right,\
                has_left_above, has_right_above,\
                has_left_below, has_right_below,\
                at_1st, at_2nd, at_3rd, at_4th\
                    = get_ambiguouspt_flags(x, y, scaling, interp_mask, ambiguous_pts, pos_mode = True)
                ext_above, ext_left = solve_xy(at_1st, at_2nd, at_3rd, at_4th, interp_mask, x, y, udf, pt_matrix, scaling)

            resolve_ambiguou_pt(ext_above, ext_left, udf_interpolated, interp_mask, x, y, udf, pt_matrix, scaling, ambiguous_pts, pt_x, pt_y)
            if "y_plus" in ambiguous_pts[pt]:
                ambiguous_pts[pt].remove("y_plus")
                
            if "x_plus" in ambiguous_pts[pt]:
                ambiguous_pts[pt].remove("x_plus")
            ambiguous_pts[pt].append('r')

        # make sure the distance field in the following step is unsigned
        udf_interpolated = np.abs(udf_interpolated)
        # although this gradient will not be accurate at most places, the gradient at the grid vertices will be accurate
        udf_grad = gradient(udf_interpolated)

        '''
        For debug
        '''
        Image.fromarray(interp_mask).save("../experiments/03.DC/edge.png")
        import pdb
        pdb.set_trace()

        # 3rd pass, fix the t-junction structures
        cleanup_invalid_junction(udf, scaling, interp_mask, keep_connection = True)
        '''
        For debug
        '''
        # Image.fromarray(interp_mask).save("../experiments/03.DC/edge.png")
        # import pdb
        # pdb.set_trace()
        # we need to do the second pass but there should have no 
        # invalid t-junctions theoretically after the second pass
        cleanup_invalid_junction(udf, scaling, interp_mask)
        cleanup_invalid_junction(udf, scaling, interp_mask)
        '''
        For debug
        '''
        Image.fromarray(interp_mask).save("../experiments/03.DC/edge.png")
        import pdb
        pdb.set_trace()

        # 4th pass, compute the center point of each grid
        for i in tqdm(range(grids_h)):
            for j in range(grids_w):
                grid_mask = np.zeros(udf_interpolated.shape).astype(bool)
                A, B, C, D = grids[i][j]['coord_index']
                t, l = A; b, r = C
                AO, BO, CO, DO = grids[i][j]['coord_index_org']
                to, lo = AO; bo, ro = CO
                grid_mask[t:b+1, l:r+1] = True
                '''
                    for debug
                '''
                # if i == 11 and j == 30:
                #     import pdb
                #     pdb.set_trace()
                grid_mask_org = np.zeros(udf.shape).astype(bool)
                grid_mask_org[to:bo+1, lo:ro+1] = True
                '''
                    The sub-grid records the stroke point of each sub-edge (if detected)
                    if there is no stroke, the stroke point will be the impossible position (-1, -1)
                '''
                
                zero_pts = []
                hit_mask = np.logical_and(grid_mask, interp_mask)
                if hit_mask.any():
                    sub_grids = np.ones((bo - to, ro - lo, 4, 2)) * -1
                    # interpolation here should be re-writed
                    for m in range(to, bo):
                        for n in range(lo, ro):
                            y0 = m * scaling
                            y1 = ( m + 1 ) * scaling
                            x0 = n * scaling
                            x1 = ( n + 1 ) * scaling
                            # check all edges of each sub grid
                            
                            up_edge_f = (interp_mask[ m * scaling, n * scaling : ( n + 1 ) * scaling ]).any() # x-axis
                            left_edge_f = (interp_mask[ m * scaling : ( m + 1 ) * scaling, n * scaling ]).any() # y-axis
                            bottom_edge_f = (interp_mask[ ( m + 1 ) * scaling, n * scaling : ( n + 1 ) * scaling ]).any() # x-axis
                            right_edge_f = (interp_mask[ m * scaling : ( m + 1 ) * scaling, (n + 1) * scaling ]).any() # axis
                            
                            # if they are labeled during the last step, re-compute the gradient and the zero point of that sub-grid
                            if up_edge_f: 
                                d0 = udf_interpolated[y0, x0]
                                d1 = udf_interpolated[y0, x1]
                                if d0 == d1: continue
                                _, x0_ = pt_matrix[y0, x0]
                                y0_, x1_ = pt_matrix[y0, x1]
                                zero_pt, _ = find_zeropt(x0_, x1_, d0, -d1)
                                zero_pt = (y0_, zero_pt)
                                sub_grids[m - to, n - lo, 0] = zero_pt
                                # we only need to consider the top edge onece
                                if m == to and zero_pt not in zero_pts:   
                                    zero_pts.append(zero_pt)
                                

                            if left_edge_f:
                                
                                d0 = udf_interpolated[y0, x0]
                                d1 = udf_interpolated[y1, x0]
                                if d0 == d1: continue
                                y0_, _ = pt_matrix[y0, x0]
                                y1_, x0_ = pt_matrix[y1, x0]
                                zero_pt, _ = find_zeropt(y0_, y1_, d0, -d1)
                                zero_pt = (zero_pt, x0_)
                                sub_grids[m - to, n - lo, 1] = zero_pt
                                if n == lo and zero_pt not in zero_pts:
                                    zero_pts.append(zero_pt)
                                
                            if bottom_edge_f:
                                d0 = udf_interpolated[y1, x0]
                                d1 = udf_interpolated[y1, x1]
                                if d0 == d1: continue
                                _, x0_ = pt_matrix[y1, x0]
                                y1_, x1_ = pt_matrix[y1, x1]
                                zero_pt, _ = find_zeropt(x0_, x1_, d0, -d1)
                                zero_pt = (y1_, zero_pt)
                                if zero_pt not in zero_pts:
                                    zero_pts.append(zero_pt)
                                sub_grids[m - to, n - lo, 2] = zero_pt
                            
                            if right_edge_f:
                                d0 = udf_interpolated[y0, x1]
                                d1 = udf_interpolated[y1, x1]
                                if d0 == d1: continue
                                y0_, _ = pt_matrix[y0, x1]
                                y1_, x1_ = pt_matrix[y1, x1]
                                zero_pt, _ = find_zeropt(y0_, y1_, d0, -d1)
                                zero_pt = (zero_pt, x1_)
                                if zero_pt not in zero_pts:
                                    zero_pts.append(zero_pt)
                                sub_grids[m - to, n - lo, 3] = zero_pt
                else:
                    sub_grids = None
                # compute the average of the zero points as the center point of current grid
                if len(zero_pts) == 0:
                    zero_pts = [pt_matrix[A], pt_matrix[B], pt_matrix[C], pt_matrix[D]]
                zero_pts = np.array(zero_pts)
                center_pt = zero_pts.mean(axis = 0)
                grids[i][j]['center_point'] = tuple(center_pt)
                grids[i][j]['perpixel_grid'] = sub_grids
        udf = udf_interpolated
    
    return grids, udf_grad, (grids_h, grids_w), pt_matrix, interp_mask

def sovle_x_plus(y, x, udf, scaling, pt_matrix, interp_mask, ambiguous_pts, pos_mode = False, reverse_direction = False):
    has_above, has_below, has_left, has_right,\
    has_upper_left, has_upper_right,\
    has_lower_left, has_lower_right,\
    has_left_above, has_right_above,\
    has_left_below, has_right_below,\
    at_1st, at_2nd, at_3rd, at_4th\
        = get_ambiguouspt_flags(x, y, scaling, interp_mask, ambiguous_pts, pos_mode = pos_mode, reverse_direction = reverse_direction)
    x0, x1, x2, x3, x4, x5, x6, y0, y1, y2, y3, y4, y5, y6,\
    d0, d1, d2, d3, d4, d5, d6, d00, d11, d22, d33, d44, d55, d66,\
    x00, x11, x22, x33, x44, x55, x66, y00, y11, y22, y33, y44, y55, y66,\
    xs1_, xs2_, xs3_, ys1_, ys2_, ys3_\
        = get_coords_dist_from_udf(x, y, udf, scaling, pt_matrix)
    connect_left_up = has_above or has_upper_left or has_left_above
    connect_right_up = has_above or has_upper_right or has_right_above
    connect_left_down = has_below or has_lower_left or has_left_below
    connect_right_down = has_below or has_lower_right or has_right_below
    # both have stroke
    if np.array([connect_right_up, connect_left_up, connect_right_down, connect_left_down]).sum() >= 3:
        ambiguous_pts[(y, x)].remove('x_plus')
        if len(ambiguous_pts[(y, x)]) == 0: return
    # only one right stroke
    elif connect_right_up and not connect_left_up and connect_right_down and not connect_left_down:
        interp_mask[y3, x2+1:x3] = False
        ambiguous_pts[(y, x)].remove('x_plus')
        if len(ambiguous_pts[(y, x)]) == 0: return
    # only one left stroke
    elif not connect_right_up and connect_left_up and not connect_right_down and connect_left_down:
        interp_mask[y3, x3+1:x4] = False
        ambiguous_pts[(y, x)].remove('x_plus')
        if len(ambiguous_pts[(y, x)]) == 0: return
    # the rest case 
    else:
        ambiguous_pts[(y, x)].append('x')
        ambiguous_pts[(y, x)].remove('x_plus')

def sovle_y_plus(y, x, udf, scaling, pt_matrix, interp_mask, ambiguous_pts, pos_mode = False, reverse_direction = False):
    has_above, has_below, has_left, has_right,\
    has_upper_left, has_upper_right,\
    has_lower_left, has_lower_right,\
    has_left_above, has_right_above,\
    has_left_below, has_right_below,\
    at_1st, at_2nd, at_3rd, at_4th\
        = get_ambiguouspt_flags(x, y, scaling, interp_mask, ambiguous_pts, pos_mode = pos_mode, reverse_direction = reverse_direction)
    x0, x1, x2, x3, x4, x5, x6, y0, y1, y2, y3, y4, y5, y6,\
    d0, d1, d2, d3, d4, d5, d6, d00, d11, d22, d33, d44, d55, d66,\
    x00, x11, x22, x33, x44, x55, x66, y00, y11, y22, y33, y44, y55, y66,\
    xs1_, xs2_, xs3_, ys1_, ys2_, ys3_\
        = get_coords_dist_from_udf(x, y, udf, scaling, pt_matrix)
    connect_up_left = has_left or has_upper_left or has_left_above
    connect_up_right = has_right or has_upper_right or has_right_above
    connect_down_left = has_left or has_lower_left or has_left_below
    connect_down_right = has_right or has_lower_right or has_right_below
    # both have stroke
    if np.array([connect_up_right, connect_up_left, connect_down_right, connect_down_left]).sum() >= 3:
        ambiguous_pts[(y, x)].remove('y_plus')
        if len(ambiguous_pts[(y, x)]) == 0: return
    # only one upper stroke
    elif connect_up_right and connect_up_left and not connect_down_right and not connect_down_left:
        interp_mask[y3+1:y4, x3] = False
        ambiguous_pts[(y, x)].remove('y_plus')
        if len(ambiguous_pts[(y, x)]) == 0: return
    # only one lower stroke
    elif not connect_up_right and not connect_up_left and connect_down_right and connect_down_left:
        interp_mask[y3+1:y4, x3] = False
        ambiguous_pts[(y, x)].remove('y_plus')
        if len(ambiguous_pts[(y, x)]) == 0: return
    # the rest case 
    else:
        ambiguous_pts[(y, x)].append('y')
        ambiguous_pts[(y, x)].remove('y_plus')

def is_in(pts, pt):
    res1 = pts[:, 0] == pt[0]
    res2 = pts[:, 1] == pt[1]
    return np.logical_and(res1, res2).any()

def find_zeropt(x1, x2, d1, d2):
    assert d1 != d2
    assert x1 != x2
    a = (d1 - d2) / (x1 - x2) 
    b = d1 - a*x1
    def f(x, a, b):
        return a*x +b
    return -(b/a), partial(f, a = a, b = b)

def find_intersection(x1, x2, d1, d2, x3, x4, d3, d4):
    assert d1 != d2
    assert x1 != x2
    assert d3 != d4
    assert x3 != x4
    a1 = (d1 - d2) / (x1 - x2) 
    b1 = d1 - a1*x1
    
    a2 = (d3 - d4) / (x3 - x4) 
    b2 = d3 - a2*x3

    return (b2 - b1) / (a1 - a2)

def to_real_coords(xs, ys, pt_matrix):
    pt_yx = pt_matrix[(ys, xs)]
    ys = pt_yx[:,0]
    xs = pt_yx[:,1]
    y_axis = np.unique(ys)
    y_axis.sort()
    x_axis = np.unique(xs)
    x_axis.sort()
    return ys, xs, y_axis, x_axis

def find_flip_loc(udf_grad, udf):
    udf_grady = udf_grad[:, :, 0]
    mask_y_grad = (udf_grady * roll(udf_grady, shift = -1, axis = 0)) < 0
    mask_y_dist = (udf + roll(udf, shift = -1, axis = 0)) <= 1
    mask_y = np.logical_or(mask_y_grad, mask_y_dist)
    
    udf_gradx = udf_grad[:, :, 1]
    mask_x_grad = (udf_gradx * roll(udf_gradx, shift = -1, axis = 1)) < 0
    mask_x_dist = (udf + roll(udf, shift = -1, axis = 1)) <= 1
    mask_x = np.logical_or(mask_x_grad, mask_x_dist)
    
    pt_y = np.array(np.where(mask_y)).T
    pt_x = np.array(np.where(mask_x)).T
    return pt_y, pt_x

def get_center_point(udf, pos, udf_zero, pt_matrix, debug = False):
    '''
    Given:
        udf, a 2D array as the full unsigned distance filed
        pos, a 4 x 2 array as the coordination of current cell's 4 corners
        udf_zero, a float indicate the relative 0 level
        debug, always return the center point as average(pos, axis = 1) if True.
    Return:
        the center point of each cell
    '''
    tl, bl, br, tr = pos
    center_pt = np.array(average_pts(pos, pt_matrix))
    dis_val = udf[tl[0]:br[0]+1, tl[1]:br[1]+1]
    pt_matrix_ = pt_matrix[tl[0]:br[0]+1, tl[1]:br[1]+1]
    # update the udf_zero to the smallest distance value
    zero_pt = pt_matrix_[np.where(dis_val <= udf_zero)]
    if len(zero_pt) == 0 or debug:
        return tuple(center_pt)
    dis_pt = np.linalg.norm(zero_pt - center_pt, axis = 1)
    closest_pt = zero_pt[np.argsort(dis_pt)[0]]
    return tuple(closest_pt)

def open_img(path_to_udf):
    udf = np.array(Image.open(path_to_udf).convert('L'))
    return np.squeeze(udf)

def cleanup_stray_stroke(lines_pt, lines_idx, grids):
    '''remove all stray lines'''
    assert len(lines_pt) == len(lines_idx)
    for i in range(len(lines_pt) - 1, -1, -1):
        a, b = lines_idx[i][0]
        c, d = lines_idx[i][1]
        if not grids[a][b][2] or not grids[c][d][2]:
            lines_pt.pop(i)


if __name__ == "__main__":
    # we should start at a simple test case
    ''' test stage 1 '''
    # open UDF
    # sketch = open_img("../experiments/test/09.png")
    # _, sketch = cv2.threshold(sketch,200,255,cv2.THRESH_BINARY)
    # sketch = 255 - sketch
    # sketch = 255 - skeletonize(sketch/255).astype(np.uint8) * 255
    # # vis_add_grid_lines(sketch, show = True)
    # # get diagnal length
    # h, w = sketch.shape
    # diag_len = np.sqrt(h**2 + w**2)
    # # compute the UDF
    # udf = cv2.distanceTransform(sketch, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    # # normalize UDF
    # udf = udf / diag_len
    # # extract topology
    # lines_pt, lines_idx, grids = dual_contouring(udf, 5)
    # # write to file
    # seg_to_svg(lines_pt, udf.shape, "../experiments/test/output.svg")
    # print("Done")

    '''test stage 3'''
    # udf = open_img("../experiments/diffusion denoise/7-1-2022/udfs/GT_0042528.png")
    # sketch = np.copy(udf)
    # sketch[udf == 255] = 0
    # sketch[udf != 255] = 255
    # Image.fromarray(sketch).show()
    # lines_pt, lines_idx, grids = dual_contouring(255 - udf, 3)
    # seg_to_svg(lines_pt, udf.shape, "../experiments/diffusion denoise/7-1-2022/udfs/output.svg")
    # # vis_add_grid_lines(sketch, 5, show = True)

    '''test case 3'''
    udf = "../experiments/03.DC/udf/0005541.npz"
    dist1 = np.load(udf)['udf']
    Image.fromarray(dist1 > 0.5).save(udf.replace(".npz", "_skel.png").replace("udf/", ''))
    lines_pt, lines_idx, grids, grad, pt_matrix = dual_contouring(dist1, 3, interpolation = 4)
    seg_to_svg(lines_pt, dist1.shape, udf.replace(".npz", "_dc.svg").replace("udf/", ''), grids, grad, pt_matrix)


    print("Done")