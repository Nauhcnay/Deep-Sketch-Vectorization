from pathlib import Path as P
import sys
directory = P(__file__)
if directory not in sys.path:
    d1 = str(directory.parent)
    if d1 not in sys.path:
        sys.path.append(d1)
    d2 = str(directory.parent.parent)
    if d2 not in sys.path:
        sys.path.append(str(directory.parent.parent))
import numpy as np
import time
import xml.etree.ElementTree as ET
import copy


from os.path import split, join, exists
from svgpathtools import Path, Line, wsvg, svg2paths, CubicBezier, Document, QuadraticBezier
from svgpathtools.document import flattened_paths
# https://stackoverflow.com/questions/30447355/speed-of-k-nearest-neighbour-build-search-with-scikit-learn-and-scipy
# we should use KDTree in sklearn since it is implemented by Cython
# from scipy.spatial import KDTree
from sklearn.neighbors import KDTree
from random import randint, shuffle, sample, choice
from fitCurves import fitCurve
# try:
#     from FitCurves import FitCurvesC
#     fitCurve = FitCurvesC.fitCurve
#     print('Using fast Bezier curve fitting')
# except ImportError:
#     print('Using slow Bezier curve fitting')
from rdp import rdp
from scipy.sparse.csgraph import shortest_path

def get_hw_offset(bbox):
    h_min, w_min, h_max, w_max = bbox
    h = h_max - h_min
    w = w_max - w_min
    h_offset = h_min + h / 2
    w_offset = w_min + w / 2
    return h_offset, w_offset

def get_rotate_matrix(ang):
    '''assume ang are degrees'''
    # convert ang from degree to radian
    assert abs(ang) <= 360
    ang = (ang / 360 ) * np.pi *+ 2
    return np.array([[np.cos(ang), -np.sin(ang)],[np.sin(ang), np.cos(ang)]])

def get_line_approximation(paths, sample_dist = None, subdivide = 4, mode = 'line'):
    '''
    Given:
        paths, list of svg elements with length of N
        subdivide, a int to decide use how many lines to approximate each curve.
    Return:
        paths_arrpox, list of svg Lines, the approximation of input svg graph with length of M
        invert_idx, M x 1 array, mapping approximate line index to svg path index
        inver_type, M x 1 array, mapping approximate line type back to svg path element type
        edges, M x 2 list, record all start and end point of each line
    '''
    paths_approx = []
    invert_idx = []
    invert_type = []
    edges = []
    pts = []
    for i, p in enumerate(paths):
        if mode == 'line':
            if type(p) is Line:
                paths_approx.append(p)
                invert_idx.append(i)
                edges.append(((p.start.real, p.start.imag), (p.end.real, p.end.imag)))
                invert_type.append("Line")
            # evently split the curve into 4 pieces
            elif type(p) is CubicBezier or isinstance(p, QuadraticBezier):
                if sample_dist is not None:
                    subdivide = int(p.length() / sample_dist)
                    subdivide = 1 if subdivide == 0 else subdivide
                else:
                    assert isinstance(subdivide, int)
                ts = np.linspace(0, 1, num = subdivide)
                for j in range(0, len(ts) - 1):
                    start = p.point(ts[j])
                    end = p.point(ts[j + 1])
                    paths_approx.append(Line(start, end))
                    invert_idx.append(i)
                    edges.append(((start.real, start.imag), (end.real, end.imag)))
                    invert_type.append("CubicBezier")
            else:
                raise ValueError("Unsupported path type %s"%str(type(p)))
            
        elif mode == 'pt':
            assert sample_dist is not None
            subdivide = int(p.length() / sample_dist)
            subdivide = 1 if subdivide == 0 else subdivide
            ts = np.linspace(0, 1, num = subdivide)
            for j in range(0, len(ts)):
                pts.append(p.point(ts[j]))
        else:
            raise ValueError("Unsupported mode %s"%mode)

    if mode == 'line':
        return paths_approx, np.array(invert_idx), np.array(invert_type), edges
    elif mode == 'pt':
        return pts

def open_svg_flatten(path_to_svg):
    filepath, svg = split(path_to_svg)

    # the svgpathtools has one function for flatting the paths but it only support xml root as the input
    # hopefully this could remove all transformations
    doc = ET.parse(path_to_svg)
    root = doc.getroot()

    # get paths with base coordinates (transforms applied)
    try:
        flatpaths = flattened_paths(root)
    except:
        flatpaths, _ = svg2paths(path_to_svg)
    paths = []
    
    # remove nested path structrue
    for p in flatpaths:
        for s in p:
            paths.append(s)

    # clean up invalid path
    for i in range(len(paths) - 1, -1, -1):
        if paths[i].start == None or paths[i].end == None:         
            paths.pop(i)
        elif paths[i].length() == 0: 
            paths.pop(i)

    # read canvas size
    try:
        h = int(root.attrib['height'].strip("px"))
        w = int(root.attrib['width'].strip("px"))
    except:
        h = int(float(root.attrib['viewBox'].split(" ")[3].strip("px")))
        w = int(float(root.attrib['viewBox'].split(" ")[2].strip("px")))

    return paths, (h, w)

def path_to_pts(paths):
    seg_idx = 0
    nplist = []
    path_index = []
    for i, p in enumerate(paths):
        if type(p) is Line:
            nplist.append((paths[i].start.real, paths[i].start.imag))
            nplist.append((paths[i].end.real, paths[i].end.imag))
            seg_idx += 2
            path_index.append((seg_idx, "Line"))
        elif type(p) is CubicBezier:
            nplist.append((paths[i].start.real, paths[i].start.imag))
            nplist.append((paths[i].control1.real, paths[i].control1.imag))
            nplist.append((paths[i].control2.real, paths[i].control2.imag))
            nplist.append((paths[i].end.real, paths[i].end.imag))
            seg_idx += 4
            path_index.append((seg_idx, 'CubicBezier'))
        else:
            continue
            # raise ValueError("Unsupported path type %s"%str(type(p)))
        
    return np.array(nplist), path_index

def pts_to_path(pts, path_index = None):
    paths = []
    path_start_idx = 0
    if path_index is None:
        assert len(pts) % 2 == 0
        for i in range(0, len(pts), 2):
            start = complex(*pts[i])
            end = complex(*pts[i + 1])
            paths.append(Line(start, end))
    else:
        for path_end_idx, path_type in path_index:
            path_list = pts[path_start_idx: path_end_idx, :]
            if path_type == "Line":
                assert len(path_list) == 2
                start = complex(*path_list[0])
                end = complex(*path_list[1])
                paths.append(Line(start, end))
            elif path_type == "CubicBezier":
                assert len(path_list) == 4
                start = complex(*path_list[0])
                ctrl1 = complex(*path_list[1])
                ctrl2 = complex(*path_list[2])
                end = complex(*path_list[3])
                paths.append(CubicBezier(start, ctrl1, ctrl2, end))
            else:
                raise ValueError("Unsupported path type %s"%path_type)
            path_start_idx = path_end_idx
    return paths

def resize_path(paths, re_scale, offset = 0):
    '''
    Given,
        paths, list contains all svg paths
        re_scale, list with size (1, 2) which contains the re-scaling factor for x and y axis
    Return,
        paths, list contains re-scaled svg paths
    '''
    assert len(re_scale) == 2
    if re_scale[0] != 1 or re_scale[1] != 1:
        pts, path_index = path_to_pts(paths)
        pts = resize_pts(pts, re_scale[0], re_scale[1], offset)
        res = pts_to_path(pts, path_index)
    else:
        res = paths
    return res

def flip_path(pts, axis, canvas_size, index = 'xy'):
    h, w = canvas_size
    h_offset, w_offset = get_hw_offset((0, 0, h, w))

    if index == 'xy':
        offset = np.array([[w_offset, h_offset]])
    elif index == 'ij':
        offset = np.array([[h_offset, w_offset]])
    else:
        raise ValueError("index mode should only be xy or ij")
    if len(pts) == 0:
        return pts
    else:
        res = pts - offset
        if (axis == 'x' and index == 'ij') or (axis == 'y' and index == 'xy'):
            res[:, 0] = res[:, 0] * -1
        elif (axis == 'y' and index == 'ij') or (axis == 'x' and index == 'xy'):
            res[:, 1] = res[:, 1] * -1
        else:
            raise ValueError("unsupported axis %s!"%axis)
        res += offset
        return res

def resize_pts(pts, x_scale, y_scale, offset = 0):
    M = np.array([[x_scale, 0], [0, y_scale]])
    pts = np.matmul(pts, M) + offset
    return pts

def rotate_pts(pts, canvas_size, ang, index = 'xy'):
    M = get_rotate_matrix(ang)
    h, w = canvas_size
    h_offset, w_offset = get_hw_offset((0, 0, h, w))
    offset = np.array([[w_offset, h_offset]])
    if index == 'ij':
        pts[:, [0, 1]] = pts[:, [1, 0]]
    else:
        if index != 'xy':
            raise ValueError("index mode should only be xy or ij")
    if len(pts) == 0:
        return pts
    else:
        res = pts - offset
        res = np.matmul(M, res.T).T
        res = res + offset
        if index == 'ij':
            res[:, [0, 1]] = res[:, [1, 0]]
        return res

def crop_path(pts, bbox, canvas_size, mode = 'keypt', index = 'xy'):
    top, left, bottom, right = bbox
    h, w = canvas_size
    if len(pts) == 0:
        return pts
    if bottom > h or right > w:
    
        raise ValueError("invalid bounding box")
    if mode == 'keypt':
        eps = 1e-3
        if index == 'xy':
            mask1 = np.logical_and(pts[:, 0] >= left - eps, pts[:, 0] <= right + eps)
            mask2 = np.logical_and(pts[:, 1] >= top - eps, pts[:, 1] <= bottom + eps)
            offset = np.array([[left, top]])
        elif index == 'ij':
            mask1 = np.logical_and(pts[:, 1] >= left - eps, pts[:, 1] <= right + eps)
            mask2 = np.logical_and(pts[:, 0] >= top - eps, pts[:, 0] <= bottom + eps)
            offset = np.array([[top, left]])
        else:
            raise ValueError("index mode should only be xy or ij")
        mask = np.logical_and(mask1, mask2)
        pts_cropped = pts[mask]
        if len(pts_cropped) > 0:
            return pts_cropped - offset
        else:
            return pts_cropped
    elif mode == 'path':
        # this function will only support Line
        offset = np.array([[left, top]])
        pts_crop = []
        end_pts = []
        for i in range(0, len(pts), 2):
            start_pt = pts[i]
            end_pt = pts[i + 1]
            if index == 'ij':
                pts[:, [0, 1]] = pts[:, [1, 0]]
            if index == 'xy' or index == 'ij':
                # pick up the point out of the bounding box
                f1 = pts[i][0] < left
                f2 = pts[i][0] > right
                f3 = pts[i][1] < top
                f4 = pts[i][1] > bottom
                assert (f1 and f2) == False
                assert (f3 and f4) == False
                mask_x1 = np.logical_or(f1, f2)
                mask_y1 = np.logical_or(f3, f4)
                f1 = pts[i + 1][0] < left
                f2 = pts[i + 1][0] > right
                f3 = pts[i + 1][1] < top
                f4 = pts[i + 1][1] > bottom
                assert (f1 and f2) == False
                assert (f3 and f4) == False
                mask_x2 = np.logical_or(f1, f2)
                mask_y2 = np.logical_or(f3, f4)
            else:
                raise ValueError("index mode should only be xy or ij")
            mask_pt1 = mask_x1 or mask_y1
            mask_pt2 = mask_x2 or mask_y2

            # if all two points out of bbox, test if the whole path is also out of bbox
            keep = False
            if mask_pt1 or mask_pt2:
                if (pts[i] == pts[i + 1]).all(): continue
                line0 = Line(complex(*pts[i]), complex(*pts[i+1]))
                line_da = Line(complex(*(left, top)), complex(*(right, top)))
                line_ab = Line(complex(*(left, top)), complex(*(left, bottom)))
                line_bc = Line(complex(*(left, bottom)), complex(*(right, bottom)))
                line_cd = Line(complex(*(right, bottom)), complex(*(right, top)))
                status0 = np.array([len(line0.intersect(line_da)), len(line0.intersect(line_ab)), len(line0.intersect(line_bc)), len(line0.intersect(line_cd))])
                status0_sum = status0.sum()
                status1 = np.array([line0.intersect(line_da), line0.intersect(line_ab), line0.intersect(line_bc), line0.intersect(line_cd)], dtype = object)
                if status0_sum == 2:
                    keep = True
                    new_endpt_t = status1[status0 != 0]
                    start_new = line0.point(new_endpt_t[0][0][0])
                    end_new = line0.point(new_endpt_t[1][0][0])
                    pts[i] = [start_new.real, start_new.imag]
                    pts[i + 1] = [end_new.real, end_new.imag]
                    end_pts.append((start_new.real, start_new.imag))
                    end_pts.append((end_new.real, end_new.imag))
                if status0_sum == 1:
                    keep = True
                    new_endpt_t = status1[status0 != 0]
                    new_pt = line0.point(new_endpt_t[0][0][0])
                    if mask_pt1:
                        pts[i] = [new_pt.real, new_pt.imag]
                    else:
                        pts[i + 1] = [new_pt.real, new_pt.imag]
                    end_pts.append((new_pt.real, new_pt.imag))
            # if only one points or no points out of bbox, keep this path to the crop list
            else:
                keep = True
                if mask_pt1:
                    end_pts.append(pts[i])
                if mask_pt2:
                    end_pts.append(pts[i + 1])

            # record current path if it is labeled
            if keep:
                pts_crop.append(pts[i])
                pts_crop.append(pts[i + 1])

        pts_crop = np.array(pts_crop)
        end_pts = np.array(end_pts)

        if len(pts_crop) > 0:
            pts_crop = pts_crop - offset
            if index == 'ij':
                pts_crop[:, [0, 1]] = pts_crop[:, [1, 0]]
        
        return pts_crop, end_pts
    else:
        raise ValueError("unsupported mode %s!"%mode)

def dot(A, B):
    return A.real * B.real + A.imag * B.imag

def simplify(paths, keypts = None):
    

    def build_search_array(paths):
        paths_array = np.array(paths)
        paths_start = paths_array[:, 0]
        paths_end = paths_array[:, 1]
        return paths_start, paths_end
    
    def unify_direction(line):
        if (line.start.real > line.end.real) or (line.start.real == line.end.real and line.start.imag > line.end.imag):
            return line.reversed()
        else:
            return line

    # make sure all segments have the same direction
    for i in range(len(paths)):
        paths[i] = unify_direction(paths[i])

    # register all keypoints onto strokes and also refine keypoints 
    paths_start, paths_end = build_search_array(paths)

    # construct each lines
    res = []
    while(len(paths) > 0):
        remove_idx = []
        paths_start, paths_end = build_search_array(paths)
        
        # random pick up a line
        start_idx = randint(0, len(paths) - 1)
        remove_idx.append(start_idx)
        tangent_idx = paths[start_idx].unit_tangent()
        
        # search along start side
        start = paths[start_idx].start
        hit_idxs = np.where(paths_end == start)[0]   
        new_path = Path(paths[start_idx])
        while len(hit_idxs) > 0:
            if len(hit_idxs) > 1:
                hit_idx = None
                max_tangent_dot = -1
                for idx in hit_idxs:
                    tangent_hit = paths[idx].unit_tangent()
                    dot_idx = abs(dot(tangent_idx, tangent_hit))
                    if max_tangent_dot < dot_idx:
                        hit_idx = idx
                        max_tangent_dot = dot_idx
                # add current segment to new path
                new_path.append(paths[hit_idx])
                # update the path list and path point list
                remove_idx.append(hit_idx)
                paths_end[hit_idx] = -1
                # update the search point
                start = paths[hit_idx].start
                hit_idxs = np.where(paths_end == start)[0]

            elif len(hit_idxs) == 1:
                hit_idx = hit_idxs[0]
                # add current segment to new path
                new_path.append(paths[hit_idx])
                # update the path list and path point list
                remove_idx.append(hit_idx)
                paths_end[hit_idx] = -1
                # update the search point
                start = paths[hit_idx].start
                hit_idxs = np.where(paths_end == start)[0]
            tangent_idx = paths[hit_idx].unit_tangent()
        new_path.reverse()

        # search along end side
        end = paths[start_idx].end
        hit_idxs = np.where(paths_start == end)[0]
        while len(hit_idxs) > 0:
            if len(hit_idxs) > 1:
                hit_idx = None
                max_tangent_dot = -1
                for idx in hit_idxs:
                    tangent_hit = paths[idx].unit_tangent()
                    dot_idx = abs(dot(tangent_idx, tangent_hit))
                    if max_tangent_dot < dot_idx:
                        hit_idx = idx
                        max_tangent_dot = dot_idx
                # add current segment to new path
                new_path.append(paths[hit_idx])
                # update the path list and path point list
                remove_idx.append(hit_idx)
                paths_end[hit_idx] = -1
                # update the search point
                end = paths[hit_idx].end
                hit_idxs = np.where(paths_start == end)[0]
            elif len(hit_idxs) == 1:
                hit_idx = hit_idxs[0]
                # add current segment to new path
                new_path.append(paths[hit_idx])
                # update the path list and path point list
                remove_idx.append(hit_idx)
                paths_start[hit_idx] = -1
                # update the search point
                end = paths[hit_idx].end
                hit_idxs = np.where(paths_start == end)[0]
        res.append(new_path)
        # remove allocated segements from original sets
        remove_idx.sort(reverse = True)
        for idx in remove_idx:
            paths.pop(idx)
    return res

def flatten_paths_to_pts_array(paths, unique = False):
    all_pts = np.asarray(paths)
    start_pts = np.stack((all_pts[:, 0].real, all_pts[:, 0].imag), axis = 1)
    end_pts = np.stack((all_pts[:, 1].real, all_pts[:, 1].imag), axis = 1)
    assert start_pts.shape == end_pts.shape
    lines_array = np.concatenate((start_pts, end_pts), axis = 1)
    pts_array = lines_array.reshape(-1, 2)
    if unique:
        # we can't simply use np.unique to cleanup redundant points cause this 
        # function will also re-order the remained points.
        map_to_clean = [0]
        for i in range(1, len(pts_array)):
            if (pts_array[i - 1] == pts_array[i]).all(): continue
            map_to_clean.append(i)
        map_to_clean = np.array(map_to_clean)
        pts_array = pts_array[map_to_clean]
    return pts_array

def flatten_paths_known_to_be_polyline_to_pts_array(paths):
    ## Since `s` is a polyline, we can just take the start point of each line segment and append the last endpoint.
    paths = np.asarray(paths)
    ## Assert that it's a polyline.
    assert ( paths[1:,0] == paths[:-1,1] ).all()
    pts = np.zeros( ( len(paths) + 1, 2 ) )
    pts[:-1,0] = paths[:,0].real
    pts[:-1,1] = paths[:,0].imag
    pts[-1,0] = paths[-1,1].real
    pts[-1,1] = paths[-1,1].imag
    return pts

def build_edge(pts_array, inv = None):
    ## Yotam: numpy.unique() can replace this entire function.
    ## I don't have a way to test the `inv` parameter, but I believe it can be handled with `return_index = True`.
    if inv is None:
        unique_pts, edges = np.unique( pts_array, axis = 0, return_inverse = True )
        edges = edges.reshape(-1,2)
        return edges, unique_pts
        # edges2 = edges

    # initial edge graphs
    assert len(pts_array) % 2 == 0
    edges = np.arange(len(pts_array)).reshape(-1, 2)
    # edges = np.insert((np.arange(2, len(pts_array)+1)//2), 0, 0).reshape(-1, 2)
    visited_pts = []
    zero_dist = 1e-5
    for i in range(len(pts_array)):
        pt = pts_array[i]
        if (pt == -1).all(): continue
        dist = np.sqrt((pts_array - pt[np.newaxis, :])**2).sum(axis = -1)
        # search connected points
        cpt_idx = np.where(dist < zero_dist)[0]
        cpt_idx = np.delete(cpt_idx, np.where(cpt_idx == i))
        # if found connections 
        if len(cpt_idx) > 0 and i not in visited_pts:
            # remove duplicate connected points
            pts_array[cpt_idx] = -1
            for e in cpt_idx:
                edges[edges == e] = i
        visited_pts.append(i)
    
    # cleanup graphs, we could let the code run faster if skip this step
    pt_idx_old_to_new = np.ones(len(pts_array)).astype(int) * -1
    new_idx_counter = 0
    for i in range(len(pts_array)):
        if (pts_array[i] != -1).any():
            pt_idx_old_to_new[i] = new_idx_counter
            new_idx_counter += 1
    mask = pts_array != -1
    pts_array = pts_array[mask].reshape(-1, 2)
    edges = pt_idx_old_to_new[edges]
    if inv is not None:
        assert (mask[:, 0] == mask[:, 1]).all()
        return edges, pts_array, inv[mask[:, 0]]
    return edges, pts_array

def build_connectivity(pts_array, edges):
    cn_dict = {} # connected neighbour dictionary
    fedge = edges.flatten()
    ptidx = np.arange(len(fedge))
    valence = np.zeros(len(pts_array))
    for i in range(len(pts_array)):
        if (pts_array[i] == -1).all(): continue
        # find the idx of the point which is identical to the ith pt
        idp = np.where(fedge == i)[0]
        # find the idx of the point which is the neighbour of the ith pt
        sp_mask = idp % 2 == 0
        ep_mask = idp % 2 == 1
        inn = idp.copy()
        if sp_mask.any():
            inn[sp_mask] = fedge[inn[sp_mask] + 1]
        if ep_mask.any():
            inn[ep_mask] = fedge[inn[ep_mask] - 1]
        inn = np.unique(inn)
        # update the connectivity dict and valence list
        if len(inn) == 0:
            cn_dict[i] = None
        else:
            cn_dict[i] = inn
            valence[i] = len(inn)
    return cn_dict, valence

def update_unvisited_pt(valence, unvisited_pts, idx):
    assert valence[idx] >= 0
    if valence[idx] == 0:
        return False
    else:
        valence[idx] = valence[idx] - 1
        if valence[idx] == 0: unvisited_pts.remove(idx)
        return True

def compute_dist_M(pt1, pt2):
    return np.sqrt(((pt1[:,np.newaxis, :] - pt2[np.newaxis, :, :])**2).sum(axis = -1).astype(float))

def find_alignment(pts_array, val_pt_idx, keypt, val, valence, pt_type, thres = 1.5):
    '''
    Given:
        pts_array, N x 2 array, all DC vertices
        val_pt_idx, M x 1 array, indices of all DC vertices that has the same valence, the number of valence is defined by val
        keypt, K x 2 array, detected key points we want to aligned with.
        val, a int, indicating which type of the DC vertices and keypoints we want to align
        valence, N x 1 array, the valence of each DC vertices
        pt_type, N x 1 array, point alginment result and this variable will be inplace updated, therefore there is NO return results
        thres, a float, distance threshold for point alignment
    Action:
        align all predicted key point to its nearest DC vertex if the distance between them is less than thres, otherwise drop the keypoint as false positive.
    '''
    # do nothing if no dual contouring vertices or keypoint is found
    if len(val_pt_idx[0]) == 0 or len(keypt)== 0: return None
    
    # compute distance matrix between to point groups
    M_dist = compute_dist_M(pts_array[val_pt_idx], keypt)
    
    # for each DC vertex, find the closest keypoint
    M_idx = np.argsort(M_dist, axis = 0)
    
    # if the distance is less than given threshold, then we find a sucessful alginment
    aligned_idx = np.where(M_dist[M_idx[0,:], np.arange(M_dist.shape[1])] <= thres)
    found_idx = M_idx[0,:][aligned_idx]

    # update result
    for idx in found_idx:
        # # if the valence is 2, check if the current vertex is really a sharp turn
        # if val == 2:
        #     edges = connectivity[val_pt_idx[0][idx]]
        #     # compute the dot product of two segment's unit tangnet
        #     ang = dot(Line(start = complex(*pts_array[edges[0]]), end = complex(*pts_array[val_pt_idx[0][idx]])).unit_tangent(), 
        #         Line(start = complex(*pts_array[val_pt_idx[0][idx]]), end = complex(*pts_array[edges[1]])).unit_tangent())
        #     if ang > 0.6: continue
        # # if multiple key points are allocated to a small range 
        pt_type[val_pt_idx[0][idx]] = val
    return None

def find_alignment_all(keypts, valence, pts_array, connectivity, pt_type = None, thre_v1 = 1.5, thre_v2 = 1.5, thre_vx = 5, sec_pass = False):
    if pt_type is None:
        pt_type = np.ones(len(pts_array)) * - 1
    v0_pt_idx = np.where(valence == 0)
    v1_pt_idx = np.where(valence == 1)
    v2_pt_idx = np.where(valence == 2)
    v3_pt_idx = np.where(valence >= 3)
    end_pt = keypts['end_point']
    sharp_pt = keypts['sharp_turn']
    junc_pt = keypts['junc']
    # end point
    find_alignment(pts_array, v1_pt_idx, end_pt, 1, valence, pt_type, thres = thre_v1)
    # sharp turn
    find_alignment(pts_array, v2_pt_idx, sharp_pt, 2, valence, pt_type, thres = thre_v2)
    # high valence junction
    if sec_pass == False:
        find_alignment(pts_array, v3_pt_idx, junc_pt, 3, valence, pt_type, thres = thre_vx)

    v1 = pts_array[np.where(pt_type == 1)]
    v1 = [complex(*pt) for pt in v1]
    v2 = pts_array[np.where(pt_type == 2)]
    v2 = [complex(*pt) for pt in v2]
    if sec_pass == False:
        v3 = pts_array[np.where(pt_type >= 3)]
        v3 = [complex(*pt) for pt in v3]
        vx = v1+v2+v3
    else:
        vx = v1+v2

    return vx, pt_type, pts_array[np.where(pt_type >= 3)]

def update_valence(path_by_idx, valence, connectivity, unvisited_edges, visited_vertex, end_pts, st_pts, loop_clean = False):
    for i in range(len(path_by_idx)):
        p = path_by_idx[i]
        v_new = 0
        remove = True
        for next_p in connectivity[p]:
            if set([p, next_p]) in unvisited_edges: 
                v_new += 1
                remove = False
        if remove:
            visited_vertex[p] = True
            if p in end_pts: end_pts.remove(p)
            if p in st_pts: st_pts.remove(p)
        valence[p] = v_new

    if loop_clean:
        for n in connectivity[path_by_idx[0]]:
            if valence[n] > 0:
                valence[n] = valence[n] - 1
            else:
                ed = set([n, path_by_idx[-1]])
                if ed in unvisited_edges: unvisited_edges.remove(ed)

def bfs_all_pairs(s, ends, connectivity, pts_array, visited_vertex):
    # prepare bfs search
    min_len = float("inf")
    min_e = None
    s_shortest = None
    min_pidx = None
    for e in ends:
        stroke, pidx = bfs_single_pair(s, e, connectivity, pts_array, visited_vertex)
        if stroke is None: continue # skip if no connection found
        if len(stroke) < min_len:
            s_shortest = stroke
            min_len = len(stroke)
            min_e = e
            min_pidx = pidx
    return s_shortest, min_e, min_pidx


def bfs_all_pairs_scipy(s, ends, edges, pts_array, visited_vertex, pres = None):
    # construct graph matrix
    # but construct such a matrix is really time consuming...
    pt_num = len(pts_array)
    csgraph = np.zeros((pt_num, pt_num))
    for eg in edges:
        i, j = list(eg)
        if i == j: continue
        if visited_vertex[i] == False and visited_vertex[j] == False:    
            csgraph[i][j] = 1
            csgraph[j][i] = 1
    _, predecessors = shortest_path(csgraph, method='auto', directed=False, return_predecessors=True, unweighted=False, overwrite=False, indices=None)

    # reconstruct path from s to all ends

    return s_shortest, min_e, min_pidx


def bfs_single_pair(s, e, neighbours, pts_array, visited):
    def solve(s, neighbours, n, visited):
        # s, idx for start point
        # neighbours, dictionary, neighours of each vertex
        # n, number of vetices of the graph
        q = []
        q.append(s)
        visited = visited.copy()
        visited[s] = True
        prev = (np.ones(n) * -1).astype(int)

        while len(q)!=0:
            node = q.pop()
            nghbs = neighbours[node]
            if len(nghbs) > 0:
                for next_node in nghbs:
                    if visited[next_node]: continue
                    q.append(next_node)
                    visited[next_node] = True
                    prev[next_node] = node
        return prev

    

    def recon_path(s, e, prev, pts_array):
        res = Path()
        at = e
        p = []
        while prev[at] != -1:
            p.append(at)
            spt = complex(*pts_array[at])
            ept = complex(*pts_array[prev[at]])
            res.append(Line(start = spt, end = ept))
            at = prev[at]
        p.append(at)
        if p[-1] == s:
            p.reverse()
            return res.reversed(), p
        else:
            return None, None

    prev = solve(s, neighbours, len(pts_array), visited)
    if prev is not None:
        res, p = recon_path(s, e, prev, pts_array)
        return res, p
    else:
        return None, None

def simplify_graph(paths, keypts = None, mode = 'hybird', skip_len = 4):

    pts_array = flatten_paths_to_pts_array(paths)
    edges, pts_array = build_edge(pts_array)
    connectivity, valence = build_connectivity(pts_array, edges)
    valence_ = valence.copy()
    unvisited_pts = set(edges.flatten())

    # align keypoints onto the strokes
    # types definition: not keypoint(-1), end point(0), sharp turn(1), T junction(2), X junction(3), Star junction(4)
    pt_type = np.ones(len(pts_array)) * - 1
    nodes = None
    if keypts is not None:
        nodes = find_alignment_all(keypts, valence, pts_array, connectivity, pt_type, thre_v1 = 1.5, thre_v2 = 1.5, thre_vx = 5)

    def traversal_shortest_path(pts_array, pt_type, unvisited_edges, connectivity, valence, scipy = False):
        # collect all end points
        st_pts = set(np.where(pt_type == 1)[0])
        end_pts = set(np.where(pt_type > 1)[0])
        v1_pts = set(np.where(valence == 1)[0])
        # random pick one end point
        # compute all shortest path to other end points, select the shortest one among them
        # remove all vertices invovled from graph, repeat this process until all end point has been used up
        res = []
        visited_vertex = (np.zeros(len(pts_array))).astype(bool)
        imp_s_e = []
        cc = 0
        bfs_time = 0
        update_time = 0
        while len(st_pts) > 0 or len(v1_pts) > 1:
            # pick up the start point
            if len(st_pts) > 0:
                s = sample(sorted(st_pts), 1)[0]
                assert valence[s] == 1
            else:
                s = sample(sorted(v1_pts), 1)[0]

            # prepare the end point candidates
            ends = st_pts.union(end_pts) - set([s])
            v1s = v1_pts - set([s]) - st_pts
            
            # 1st pass, find shortest path between start point to each keypoint
            start_time = time.time()
            if scipy:
                s_shortest, predecessors, min_pidx = bfs_all_pairs_scipy(s, ends, unvisited_edges, pts_array, visited_vertex)
            else:
                s_shortest, min_e, min_pidx = bfs_all_pairs(s, ends, connectivity, pts_array, visited_vertex)
                
            # 2nd pass, find shortest path between start point to all valence 1 vertices
            if s_shortest is None:
                if scipy:
                    s_shortest, predecessors, min_pidx = bfs_all_pairs_scipy(s, v1s, connectivity, pts_array, visited_vertex, predecessors)   
                else:
                    s_shortest, min_e, min_pidx = bfs_all_pairs(s, v1s, connectivity, pts_array, visited_vertex)
            end_time = time.time()
            bfs_time = bfs_time + (end_time - start_time)

            # record result if find one shortest path
            start_time = time.time()
            if s_shortest is not None:
                might_loop = True
                # remove edges, first test if we could safely remove all edges
                for i in range(len(min_pidx) - 1):
                    eg = set([min_pidx[i], min_pidx[i+1]])
                    if eg in unvisited_edges:
                        unvisited_edges.remove(eg)
                        might_loop = False
                    # update valence and visited_vertex
                    update_valence(min_pidx, valence, connectivity, unvisited_edges, visited_vertex, end_pts, st_pts)
                    if might_loop:
                        print("log:\tfind loop, remove the start point and restart the reaserching")
                        for n in connectivity[s]:
                            ed = set([s, n])
                            if ed in unvisited_edges:unvisited_edges.remove(ed)
                        update_valence([s], valence, connectivity, unvisited_edges, visited_vertex, end_pts, st_pts, True)
                        assert s not in end_pts
                        assert s not in st_pts
                if len(s_shortest) > 1:     
                    res.append(s_shortest)
            else:
                # this is a dirty fix, but I still don't know why no path found is possible
                update_valence([s], valence, connectivity, unvisited_edges, visited_vertex, end_pts, st_pts, True)
            end_time = time.time()
            update_time = update_time + (end_time - start_time)
            v1_pts = set(np.where(valence == 1)[0])
            
        print("\nlog:\tbfd time %ds, update time %ds"%(bfs_time, update_time))
        assert (valence >= 0).all()
        # assert len(unvisited_edges) == 0 or len(unvisited_edges) == 1
        return res

    if mode.lower() == 'bfs':
        edges = [set(e) for e in edges]
        strokes = traversal_shortest_path(pts_array, pt_type, edges, connectivity, valence, False)
    elif mode.lower() == 'hybird':
        ## reconstruct stroke topology
        def find_next_hop_graph(cur_pt_idx, pre_pt_idx, unvisited_pts, connectivity, valence, res_stroke, pts_array, pt_type, dfs = False, pre_dfs = False):
            def find_next_hop_dfs(cur_pt_idx, pre_pt_idx, unvisited_pts, connectivity, res_stroke, pts_array, valence, pre_dfs):
                # if pt_type[cur_pt_idx] > 1:
                #     return cur_pt_idx, pre_pt_idx, False, False
                # find tangent of current stroke
                pre_tan = res_stroke[-1].unit_tangent()
                max_dot = -2
                max_tan_idx = None
                next_stroke = [] # stroe all strokes in next hop
                # we don't search backward
                next_pt = set(connectivity[cur_pt_idx]).intersection(unvisited_pts) - set([pre_pt_idx])
                if len(next_pt) == 0:
                    return cur_pt_idx, pre_pt_idx, False, False
                if len(next_pt) == 1:
                    if pre_dfs:
                        # if reaches to the end point
                        npt = next_pt.pop()
                        res_stroke.append(Line(start = complex(*pts_array[cur_pt_idx]), end = complex(*pts_array[npt])))
                        update_unvisited_pt(valence, unvisited_pts, npt)
                        return npt, cur_pt_idx, False, False
                    else:
                        npt = next_pt.pop()
                        res_stroke.append(Line(start = complex(*pts_array[cur_pt_idx]), end = complex(*pts_array[npt])))
                        update_unvisited_pt(valence, unvisited_pts, npt)
                        return npt, cur_pt_idx, True, True
                for i in next_pt:
                    next_stroke.append(Line(start = complex(*pts_array[cur_pt_idx]), end = complex(*pts_array[i])))
                    cur_tan = next_stroke[-1].unit_tangent()
                    cur_dot = dot(cur_tan, pre_tan)
                    if max_dot < cur_dot:
                        max_dot = cur_dot
                        max_tan_idx = i
                res_stroke.append(Line(start = complex(*pts_array[cur_pt_idx]), end = complex(*pts_array[max_tan_idx])))
                update_unvisited_pt(valence, unvisited_pts, max_tan_idx)
                return max_tan_idx, cur_pt_idx, True, False

            def add_pt_to_removal(pt_idxs, valence):
                pt_idxs_ = copy.deepcopy(pt_idxs)
                for idx in pt_idxs:
                    if valence[idx] > 0:
                        valence[idx] = valence[idx] - 1
                        if valence[idx] > 0:
                            pt_idxs_.remove(idx)
                    else:
                        # if we found the current pt's valence is 0 already
                        # then we should skip this point
                        assert valence[idx] == 0
                        pt_idxs_.remove(idx)
                return pt_idxs_

            def sync_val(old, new):
                assert (new >= 0).all()
                assert len(old) == len(new)
                # for i in range(len(old)): old[i] = new[i]
                old[:] = new[:]
                assert (old == new).all()

            # if pt_type[cur_pt_idx] >= 1:
            #     return cur_pt_idx, pre_pt_idx, False, False
            # if dfs flag is true, run dfs mode directly
            if dfs:
                if update_unvisited_pt(valence, unvisited_pts, cur_pt_idx):
                    return find_next_hop_dfs(cur_pt_idx, pre_pt_idx, unvisited_pts, connectivity, res_stroke, pts_array, valence, pre_dfs)
                else:
                    return cur_pt_idx, pre_pt_idx, False, False
            # else start bfs mode
            nodes = [set(), set(), set()]
            wbrpt = set() # will be removed points
            ## bfs search lv1
            valence_ = valence.copy()
            neighbours = set(connectivity[cur_pt_idx]).intersection(unvisited_pts) - set([pre_pt_idx])
            nodes[0] = neighbours
            if len(neighbours) == 0:
                # if reaches to the endpoint, return directly
                return cur_pt_idx, pre_pt_idx, False, False
            else:
                wbrpt = add_pt_to_removal(neighbours, valence_)
                for pt in neighbours: 
                    # if pt_type[pt] > 2:
                    #     return pt, cur_pt_idx, False, False       
                    wbrpt = wbrpt.union(add_pt_to_removal(set([cur_pt_idx]), valence_))
            ## bfs search lv2
            rm_lv1 = []
            loop1_ = []
            for pt in nodes[0]:
                # if pt in wbrpt: continue
                # neighbours at lv2
                neighbours = set(connectivity[pt]).intersection(unvisited_pts - wbrpt) - set([cur_pt_idx, pre_pt_idx])
                loop1_.append(set(connectivity[pt]).intersection(unvisited_pts) - set([cur_pt_idx, pre_pt_idx]))
                wbrpt = wbrpt.union(add_pt_to_removal(neighbours, valence_))
                if len(neighbours) == 0:
                    rm_lv1.append(pt)
                    continue
                for pt_ in neighbours:
                    ptn = add_pt_to_removal([pt], valence_)
                    wbrpt = wbrpt.union(ptn)
                    nodes[1].add(pt_)
            if len(rm_lv1) > 0 and len(nodes[1]) > 0:
                for i in rm_lv1: nodes[0].remove(i)
            # check if inner loop case 1 is detected
            if len(loop1_) > 1:
                loop1 = loop1_[0]
                for i in range(2, len(loop1_)):
                    loop1 = loop1.intersection(loop1_[i])
            else:
                loop1 = set()
            # check if inner loop case 2 is detected
            loop2 = nodes[0].intersection(nodes[1])
            if len(loop2) == 1:
                nodes[0] = loop2
                nodes[1] = nodes[1] - loop2
            ## tell if there matches any cases that could be proceed.
            if DEBUG:
                import pdb
                pdb.set_trace()
            if len(nodes[0]) == 1 and len(nodes[1]) == 1:
                n0 = nodes[0].pop()
                n1 = nodes[1].pop()
                res_stroke.append(Line(start = complex(*pts_array[cur_pt_idx]), end = complex(*pts_array[n0])))
                res_stroke.append(Line(start = complex(*pts_array[n0]), end = complex(*pts_array[n1])))
                for pt in wbrpt: unvisited_pts.remove(pt)
                sync_val(valence, valence_)
                return n1, n0, False, False
            elif len(nodes[0]) == 1 and len(nodes[1]) == 0:
                n0 = nodes[0].pop()
                res_stroke.append(Line(start = complex(*pts_array[cur_pt_idx]), end = complex(*pts_array[n0])))
                for pt in wbrpt: unvisited_pts.remove(pt)
                sync_val(valence, valence_)
                return n0, cur_pt_idx, False, False
            elif len(nodes[0]) > 1 and len(nodes[1]) > 1:
                if update_unvisited_pt(valence, unvisited_pts, cur_pt_idx):
                    return find_next_hop_dfs(cur_pt_idx, pre_pt_idx, unvisited_pts, connectivity, res_stroke, pts_array, valence, pre_dfs)
                else:
                    # break searh here and start a new search if next two levels contains branches
                    return cur_pt_idx, pre_pt_idx, False, False    
                # return cur_pt_idx, pre_pt_idx, False, False
            
            ## bfs search lv3
            rm_lv2 = []
            for pt in nodes[1]:
                # if pt in wbrpt: continue
                neighbours = set(connectivity[pt]).intersection(unvisited_pts - wbrpt)  - nodes[0] - set([cur_pt_idx, pre_pt_idx])
                wbrpt = wbrpt.union(add_pt_to_removal(neighbours, valence_))
                if len(neighbours) == 0:
                    rm_lv2.append(pt)
                    continue
                for pt_ in neighbours:
                    ptn = add_pt_to_removal([pt], valence_)
                    wbrpt = wbrpt.union(ptn)
                    nodes[2].add(pt_)
            if len(rm_lv2) > 0 and len(nodes[2]) > 0:
                for i in rm_lv2: nodes[1].remove(i)
            if DEBUG:
                import pdb
                pdb.set_trace()
            # if find inner loop between lv0 and lv2, build up the stroke and return new index for next hoop
            if len(nodes[0]) >= 1 and len(nodes[1]) == 1 and len(nodes[2]) == 1:
                n0 = nodes[0].pop()
                n1 = nodes[1].pop()
                n2 = nodes[2].pop()
                res_stroke.append(Line(start = complex(*pts_array[cur_pt_idx]), end = complex(*pts_array[n0])))
                res_stroke.append(Line(start = complex(*pts_array[n0]), end = complex(*pts_array[n1])))
                res_stroke.append(Line(start = complex(*pts_array[n1]), end = complex(*pts_array[n2])))
                for pt in wbrpt: unvisited_pts.remove(pt)
                sync_val(valence, valence_)
                return n2, n1, False, False
            # if find multiple branch, fall back to dfs search
            elif len(nodes[0]) > 1:
                if update_unvisited_pt(valence, unvisited_pts, cur_pt_idx):
                    return find_next_hop_dfs(cur_pt_idx, pre_pt_idx, unvisited_pts, connectivity, res_stroke, pts_array, valence, pre_dfs)
                else:
                    return cur_pt_idx, pre_pt_idx, False, False
            # if find branch at next 2 hop, move to the next hop only
            elif len(nodes[0]) == 1 and (len(nodes[1]) > 1 or len(nodes[2]) > 1):
                n0 = nodes[0].pop()
                res_stroke.append(Line(start = complex(*pts_array[cur_pt_idx]), end = complex(*pts_array[n0])))
                update_unvisited_pt(valence, unvisited_pts, n0)
                update_unvisited_pt(valence, unvisited_pts, cur_pt_idx)
                return n0, cur_pt_idx, False, False
            else:
                return cur_pt_idx, pre_pt_idx, False, False

        DEBUG = False
        first_iter = False
        strokes = []
        # while there still exists unallocated pts
        while len(unvisited_pts) != 0:
            ed = set(np.where(pt_type == 1)[0]).intersection(unvisited_pts)
            if len(ed) > 0 and DEBUG == False:
                sp = sample(sorted(ed), 1)[0]
            else:
                sp = sample(sorted(unvisited_pts), 1)[0]
            if DEBUG and first_iter:
                sp = 19
                first_iter = False
            if DEBUG: print("log:\tsq = %d"%sp)
            eps = set(connectivity[sp]).intersection(unvisited_pts)
            assert(update_unvisited_pt(valence, unvisited_pts, sp))
            # if we hit an endpoint, then skip half of the searching
            skip_sp = False
            if len(eps) == 1:
                skip_sp = True
            if len(eps) == 0:
                if (valence == 0).all():
                    break
                else:
                    continue    
            ep = eps.pop()
            if DEBUG and first_iter:
                ep = 17
            if DEBUG: print("log:\teq = %d"%ep)
            assert(update_unvisited_pt(valence, unvisited_pts, ep))
            cur_pidx = sp
            pre_pidx = ep
            if skip_sp == False:
                res_stroke = Path(Line(start = complex(*pts_array[sp]), end = complex(*pts_array[ep])).reversed())
            else:
                res_stroke = Path(Line(start = complex(*pts_array[sp]), end = complex(*pts_array[ep])))
            dfs = False
            pre_dfs = False
            # search along start point
            if skip_sp == False:
                while True:
                    # maybe need to rewrite the logic here, like summerize everything into a function
                    cur_pidx_new, pre_pidx_new, dfs, pre_dfs = find_next_hop_graph(
                        cur_pidx, 
                        pre_pidx, 
                        unvisited_pts, 
                        connectivity, 
                        valence, 
                        res_stroke, 
                        pts_array,
                        pt_type,
                        dfs, 
                        pre_dfs)
                    if cur_pidx_new != cur_pidx or pre_pidx_new != pre_pidx:
                        pre_pidx = pre_pidx_new
                        cur_pidx = cur_pidx_new
                    else:
                        break
                res_stroke = res_stroke.reversed()
            
            # search along end point
            cur_pidx = ep
            pre_pidx = sp
            dfs = False
            pre_dfs = False
            while True and len(unvisited_pts) > 0:
                cur_pidx_new, pre_pidx_new, dfs, pre_dfs = find_next_hop_graph(
                    cur_pidx, 
                    pre_pidx, 
                    unvisited_pts, 
                    connectivity, 
                    valence, 
                    res_stroke, 
                    pts_array, 
                    pt_type,
                    dfs, 
                    pre_dfs)
                if cur_pidx_new != cur_pidx or pre_pidx_new != pre_pidx:
                    pre_pidx = pre_pidx_new
                    cur_pidx = cur_pidx_new
                else:
                    break

            # there could have inner loop which length greater than 1
            if len(res_stroke) > skip_len: 
                strokes.append(res_stroke)
            # strokes.append(res_stroke)
    else:
        raise ValueError("unsupported running mode, the mode should only be one in the set (bfs, hybird)")
    return strokes, nodes

def fitBezier(strokes):
    res = []
    for s in strokes:
        # pts = flatten_paths_to_pts_array(s, True)
        pts = flatten_paths_known_to_be_polyline_to_pts_array(s)
        bezier = fitCurve(pts, 0.5)
        ## This test passes, so I replace fitCurve on import
        # bezier2 = FitCurves.FitCurve( pts, 0.5 )
        # assert np.abs( np.asarray(bezier) - bezier2 ).max() < 1e-10
        st = Path()
        for b in bezier:
            b = [complex(*arr) for arr in b]
            b = CubicBezier(b[0],b[1],b[2],b[3])
            # skip really bad fitting results
            # if b.length() <= 1.1 * s.length():
            st.append(CubicBezier(b[0],b[1],b[2],b[3]))
        res.append(st)
    return res

def ramerDouglas(strokes, epsilon):
    from rdp import rdp
    # Ramer-Douglas-Peucker simplify
    res = []
    for s in strokes:
        sim = rdp(flatten_paths_to_pts_array(s, True), epsilon = epsilon)
        p_new = Path()
        for i in range(1, len(sim)):
            p_new.append(Line(start = complex(*sim[i-1]), end = complex(*sim[i])))
        res.append(p_new)
    return res

def refine_topology_2nd(strokes, keypts, juncs):
    # flat strokes back to paths
    vertices = []
    poly_to_st = []
    sidx = 0
    
    # flat strokes back to paths
    for st in strokes:
        vertices = vertices + list(st)
        poly_to_st = poly_to_st + [sidx] * (len(st)  * 2)
        sidx += 1
    poly_to_st = np.array(poly_to_st)

    # build up graphs of poly lines
    pts_poly = flatten_paths_to_pts_array(vertices)
    edges_poly, pts_poly, poly_to_st = build_edge(pts_poly, poly_to_st)
    con_poly, val_poly = build_connectivity(pts_poly, edges_poly)


    # re-align keypoints to the refined strokes
    vx_poly, pt_type_poly, _ = find_alignment_all(keypts, val_poly, pts_poly, con_poly, thre_v1 = 1.5, thre_v2 = 1.5, thre_vx = 5, sec_pass = True)
    juncs_ = [complex(*pt) for pt in juncs]
    vx_poly =  vx_poly + juncs_

    ## re-register predicted sharp turn and aligned junction pts onto the refined strokes
    v2 = pts_poly[pt_type_poly == 2]
    vx = np.concatenate((v2, juncs), axis = 0)
    # compute distance matrix between to point groups
    M_dist = compute_dist_M(pts_poly, vx)
    found_idx = []
    for i in range(M_dist.shape[1]):
        fd_idx = np.where(M_dist[:, i] < 1e-2)[0]
        if len(fd_idx) > 0:
            found_idx.append(fd_idx)
    found_idx = np.concatenate(found_idx, axis = 0)

    ## map keypoints back to strokes
    st_to_pt = [None] * len(strokes)
    for pi in found_idx: # pi, point index
        si = poly_to_st[pi] # si, stroke index
        ptss = np.where(poly_to_st == si)[0]# ptss, all points in a given stroke
        # if the keypoint is in the middle of a stroke, record it
        if pi == ptss[0] or pi == ptss[-1]: 
            aa = 0
            continue
        if st_to_pt[si] is None:
            st_to_pt[si] = [pi]
        else:
            st_to_pt[si].append(pi)

    ## split strokes if necessary
    # add new splitted strokes
    removed_st_idx = []
    for i in range(len(st_to_pt)):
        if st_to_pt[i] is None: continue
        removed_st_idx.append(i)
        st_to_pt[i].sort()
        # this need to be updated 
        ptss = np.where(poly_to_st == i)[0]
        # we might drop the end point when get junctions, this is a dirty fix and we should find something more elegant to solve this...
        add_front = None
        add_end = None
        if len(ptss) - 1 != len(strokes[i]):
            if complex(*pts_poly[ptss[0]]) != strokes[i][0].start:
                # assert complex(*pts_poly[ptss[0]]) == strokes[i][0].end
                add_front = np.array([[strokes[i][0].start.real, strokes[i][0].start.imag]])
            if complex(*pts_poly[ptss[-1]]) != strokes[i][-1].end:
                # assert complex(*pts_poly[ptss[-1]]) == strokes[i][-1].start
                add_end = np.array([[strokes[i][-1].end.real, strokes[i][-1].end.imag]])
        st = ptss[0] # start point
        ed = ptss[-1] # end point
        keypt_list = [st] + st_to_pt[i] + [ed]
        sl = 0 # stroke lenght
        for j in range(len(keypt_list) - 1):
            pts = pts_poly[keypt_list[j] : keypt_list[j+1]+1]
            if j == 0 and add_front is not None:
                pts = np.concatenate((add_front, pts), axis = 0)
            if j == len(keypt_list) - 2 and add_end is not None:
                pts = np.concatenate((pts, add_end), axis = 0)
            s = Path() # stroke
            for k in range(len(pts) - 1):
                s.append(Line(complex(*pts[k]), complex(*pts[k+1])))
            sl += len(s)
            if len(s) > 0:
                strokes.append(s)
        # assert len(strokes[i]) == sl
    # remove old strokes that have been splitted
    removed_st_idx.sort()

    for i in range(len(removed_st_idx) - 1, -1 , -1):
        strokes.pop(removed_st_idx[i])

    return strokes, (vx_poly, pt_type_poly, juncs)

if __name__ == "__main__":
    import os
    PATH_TO_SVG = '../experiments/08.exp_compairison/00.debug_simplified'
    PATH_TO_RES = '../experiments/08.exp_compairison/00.debug_simplified'
    VIS = True
    for svg in os.listdir(PATH_TO_SVG):
        if "svg" not in svg or "keypt" in svg:continue
        if exists(join(PATH_TO_RES, svg.replace('.svg', '_sim.svg'))):continue
        print("Log:\tprocessing %s ..."%svg, end='')
        start_time = time.time()
        paths, (h, w) = open_svg_flatten(join(PATH_TO_SVG, svg))
        try:
            keypts = np.load(join(PATH_TO_SVG, svg.replace("_refine.svg", "_keypt.npz").replace(".svg", "_keypt.npz")))
        except:
            keypts = None
        # return grouped polylines
        strokes, nodes = simplify_graph(paths, keypts, mode = "hybird")
        
        # refine strokes and detected keypoints
        strokes, nodes = refine_topology_2nd(strokes, keypts, nodes[-1])
        
        # fit bezier curves on the polyline strokes
        # strokes = fitBezier(strokes)
        
        # visualize search results
        if VIS:
            import random
            colors = []
            for _ in range(len(strokes)):
                colors.append((random.randint(0, 255), random.randint(0, 255), random.randint(0, 255)))
            
            if nodes is not None:
                vx, pt_type, juncs = nodes
                n_color = []
                n_color = n_color + ['red'] * (pt_type == 1).sum()
                n_color = n_color + ['green'] * (pt_type == 2).sum()
                n_color = n_color + ['purple'] * len(juncs)
                wsvg(strokes, stroke_widths = [0.5]*len(strokes), colors = colors, 
                    nodes = vx, node_colors = n_color,  node_radii= [0.5]*len(n_color),
                    dimensions = (w, h), filename = join(PATH_TO_RES, svg.replace('.svg', '_sim.svg')))
            else:
                wsvg(strokes, stroke_widths = [0.5]*len(strokes), colors = colors, dimensions = (w, h), filename = join(PATH_TO_RES, svg.replace('.svg', '_sim.svg')))
        else:
            wsvg(strokes, stroke_widths = [0.5]*len(strokes), dimensions = (w, h), filename = join(PATH_TO_RES, svg.replace('.svg', '_sim.svg')))

        end_time = time.time()
        print("finished with %d mins (%d seconds)"%((end_time - start_time) / 60, end_time - start_time))
        start_time = end_time

