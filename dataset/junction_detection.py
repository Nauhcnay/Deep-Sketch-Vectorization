# Based on code from https://github.com/Nauhcnay/A-Benchmark-for-Rough-Sketch-Cleanup

from __future__ import print_function, division
import sys
import os
import scipy.spatial
import scipy.optimize
import numpy as np
import xml.etree.ElementTree as ET
import math
import cv2

from tqdm import tqdm
# from svgpathtools import Document, svg2paths
from svgpathtools import *
from numpy import asfarray, arange, abs
from aabbtree import AABBTree, AABB
from os.path import *
from preprocess import rasterize
from io import BytesIO
from skimage.morphology import skeletonize
from multiprocessing import Pool

eps = 1e-8


def distance_point_to_segment(point, segment):
    # From https://github.com/mathandy/svgpathtools/blob/master/svgpathtools/path.py
    # Simpler than: https://gist.github.com/mathandy/c85736a70b7a54ba301696aacfbc4dbb
    return segment.radialrange(complex(*point))[0][0]


def get_global_scale(tree):
    root = tree.getroot()

    # Search for transform nodes. If there is one and everything above it is a <g> or <svg>,
    # and it has a uniform scale, we can extract the scale and apply that manually.
    transforms = root.findall('.//*[@transform]')
    # If there are no transforms, we're fine. Scale is 1.0.
    if len(transforms) == 0:
        return 1.0

    # If there is more than one transform, we can't handle it.
    if len(transforms) > 1:
        raise NotImplementedError("More than one transform.")

    # We may be able to handle a single transform node.
    transform = transforms[0]

    # If the transform node is not the only child of the root or a chain
    # of only children from the root.
    xml_prefix = '{http://www.w3.org/2000/svg}'
    # These tags are the ones that svgpathtools parses, along with groups (g).
    forbidden_siblings = frozenset(xml_prefix + tag for tag in [
                                   'g', 'path', 'polyline', 'polygon', 'line', 'ellipse', 'circle', 'rect'])
    # for every children, get their parents
    parent_map = {c: p for p in tree.iter() for c in p}
    ancestors = set([transform])
    node = transform
    while True:
        parent = parent_map[node]
        for sibling in parent:
            if sibling not in ancestors and sibling.tag in forbidden_siblings:
                raise NotImplementedError("Transform is not global.")
        if parent.tag == xml_prefix + 'svg':
            break
        ancestors.add(parent)
        node = parent

    import re
    match = re.match(
        r'matrix\(([^,]*),([^,]*),([^,]*),([^,]*),([^,]*),([^,]*)\)', transform.attrib['transform'])
    if match is None:
        raise NotImplementedError("Unsupported transform.")

    import numpy.linalg
    m = [float(match.group(i)) for i in range(1, 5)]
    m = numpy.asarray([[m[0], m[1]], [m[2], m[3]]], order='F')
    _, s, _ = numpy.linalg.svd(m)
    if (s[0] - s[1]) > eps:
        raise NotImplementedError("Non-uniform scale.")

    scale = s[0]
    print("Found a global transform with uniform scale", scale)
    return scale


def set_segment_by_point(seg, start=None, end=None):
    '''
    Given:
        seg: a segement of a path element
        start: new start point that close the segement junction
        end: new end point that close the segement junction
    Return:
        A new snapped segment

    Now this function only support Line, QuadraticBezier, CubicBezier and Arc elements
    '''
    if start == None and end == None:
        raise SyntaxError("At least one point should be given.")

    if type(seg) == Line:
        if start != None:
            if seg.end == start:
                return seg
            new_seg = Line(start, seg.end)
        if end != None:
            if seg.start == end:
                return seg
            new_seg = Line(seg.start, end)

    elif type(seg) == QuadraticBezier:
        if start != None:
            if seg.end == start:
                return seg
            new_seg = QuadraticBezier(start, seg.control, seg.end)
        if end != None:
            if seg.start == end:
                return seg
            new_seg = QuadraticBezier(seg.start, seg.control, end)

    elif type(seg) == CubicBezier:
        if start != None:
            if seg.end == start:
                return seg
            new_seg = CubicBezier(start, seg.control1, seg.control2, seg.end)
        if end != None:
            if seg.start == end:
                return seg
            new_seg = CubicBezier(seg.start, seg.control1, seg.control2, end)

    elif type(seg) == Arc:
        if start != None:
            if start == seg.end:
                return seg
            new_seg = Arc(start, seg.radius, seg.rotation,
                          seg.large_arc, seg.sweep, seg.end)
        if end != None:
            if seg.start == end:
                return seg
            new_seg = Arc(seg.start, seg.radius, seg.rotation,
                          seg.large_arc, seg.sweep, end)
    else:
        raise ValueError("Unspport segment types! %s" % type(seg))

    return new_seg


def dot(tan1, tan2):
    eps = 1e-8
    tan1_arr = np.array([tan1.real, tan1.imag])
    tan2_arr = np.array([tan2.real, tan2.imag])
    tan1_norm = np.sqrt((tan1_arr**2).sum())
    tan2_norm = np.sqrt((tan2_arr**2).sum())
    return np.dot(tan1_arr, tan2_arr) / (tan1_norm * tan2_norm + eps)


def on_same_line(pt1, pt2, pt3, f):
    # https://blog.csdn.net/zsnowwolfy/article/details/82793489
    # eps = 1.5
    eps = 1e-4
    x1, y1 = pt1.real, pt1.imag
    x2, y2 = pt2.real, pt2.imag
    x3, y3 = pt3.real, pt3.imag
    area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) +
                     x3 * (y1 - y2)) * (f ** 2)
    # print(area)
    if area < eps:
        return True
    else:
        return False


def to_found_pt_arr(endpoints_find):
    fl = []
    for ls in endpoints_find:
        fl += ls
    return np.unique(np.array(fl))


def debug_pt_to_svg(points, name, doc, endpoints=None):
    pt_svg = []
    if endpoints is not None:
        pt_svg = []
        for idxs in points:
            avg_pt = []
            for idx in idxs:
                avg_pt.append(complex(*endpoints[idx]))
            avg_pt = sum(avg_pt)/len(avg_pt)
            pt_svg.append(avg_pt)
    else:
        pts = points
        for pt in pts:
            pt_svg.append(complex(*pt))

    pt_num = len(pt_svg)
    wsvg(node_radii=[2]*pt_num, node_colors=["red"]*pt_num, nodes=pt_svg,
         filename=name + "_debug.svg")
    insert_rough_clean(doc, name, True)


def detect_svg(path_to_svg, distance1, distance2, distance3,
               path_to_result=None, to_svg=True, multiproc = False):

    if ".svg" not in path_to_svg: return 0
    # set the output path the same as the input if the output path is not given
    if path_to_result is None:
        path_to_result, _ = split(path_to_svg)

    # open svg
    try:
        paths, filepath, name, distance1, distance2, distance3, doc = read_svg(
            path_to_svg, float(distance1), float(distance2), float(distance3))
    except Exception as e:
        print(str(e))
        return

    # skip if already have results
    out_npz_path = join(path_to_result, name) + ".npz"
    if exists(out_npz_path):
        return 0

    if multiproc:
        _, fname = split(path_to_svg)
        print("Log:\tprocessing %s"%fname)

    # remove the path with 0 length
    short_p_idx = []
    short_seg_idx = []
    for i in range(len(paths)):
        try:
            if paths[i].length() == 0:
                short_p_idx.append(i)
                continue
        except:
            continue
        for j in range(len(paths[i])):
            if paths[i][j].length() == 0:
                short_seg_idx.append((i, j))
    for a in range(len(short_seg_idx) - 1, -1, -1):
        i, j = short_seg_idx[a]
        paths[i].pop(j)
    for i in range(len(short_p_idx) - 1, -1, -1):
        paths.pop(short_p_idx[i])

    # get raw svg root
    root_org = doc.getroot()

    # get endpoint lists
    try:
        endpoints, endpoint_addresses = endpoint_generate(paths)
    except:
        return None

    # 1st pass, remove closed smooth junctions, thresholded by distance1
    endpoints_find = []
    # pts_smooth = []  # ok, we need one more detection result to pick up smooth points
    to_org_pt_idx = []  # map the idx of rest points to the origin idx in endpoints
    pt_neighbours = endpoint_detection(endpoints, distance1)
    for pts in pt_neighbours:
        if len(pts) == 1:
            # and we also don't need to record the color right now
            endpoints_find.append(endpoints[pts[0]])
            to_org_pt_idx.append(pts[0])
        elif len(pts) == 2:
            p_idx1, seg_idx1, t1 = endpoint_addresses[pts[0]]
            p_idx2, seg_idx2, t2 = endpoint_addresses[pts[1]]
            seg1 = paths[p_idx1][seg_idx1]
            seg2 = paths[p_idx2][seg_idx2]
            # here we don't merge any points, we only skip points on the smooth junctions
            if is_parallel((seg1, t1), (seg2, t2), distance2) == False:
                endpoints_find.append(endpoints[pts[0]])
                endpoints_find.append(endpoints[pts[1]])
                to_org_pt_idx.append(pts[0])
                to_org_pt_idx.append(pts[1])
            # else:
            #     pt1 = complex(*endpoints[pts[0]])
            #     pt2 = complex(*endpoints[pts[1]])
            #     pts_smooth.append((pt1+pt2)/2)
        else:
            # if these points can be caught by distance1, and there will definitely be caught by distance2
            # so we don't need to do anything here, just append them
            for pt_idx in pts:
                endpoints_find.append(endpoints[pt_idx])
                to_org_pt_idx.append(pt_idx)

    # 2nd pass, add t junction points that have distance less than distance2 to each other
    # also record the loop so that we can remove points caused by overshoting
    endpoints_find, endpoints_update, endpoint_addresses_update, to_org_pt_idx, junction_idx_start = \
        open_junction_detection(
            paths, distance2, endpoints, endpoint_addresses, endpoints_find, to_org_pt_idx)

    # snap points that are very close to each other
    endpoints_find_rough = endpoint_detection(endpoints_find, distance2)
    pt_detect = []  # final detection results as array of complex number
    pt_colors = []  # color for each point
    pt_to_stroke = {}  # tables for recording the overshotting strokes
    stroke_to_stroke = {}
    stroke_to_pt = {}
    pt_to_idx = {}
    endpoints_find_counter = 0
    for i_find, pts in enumerate(endpoints_find_rough):
        i = to_org_pt_idx[i_find]
        if type(pts) == list:
            if len(pts) == 1:
                # red color for single endpoints
                pt_idx_new = pts[0]
                pt_idx = to_org_pt_idx[pts[0]]
                if complex(*endpoints_update[pt_idx]) in pt_to_idx:
                    continue
                pt_detect.append(complex(*endpoints_update[pt_idx]))
                pt_to_idx[complex(*endpoints_update[pt_idx])
                          ] = endpoints_find_counter
                endpoints_find_counter += 1
                if pt_idx_new > junction_idx_start:
                    pt_colors.append("purple")
                else:
                    pt_colors.append("red")

            elif len(pts) == 2:
                # green color for sharp corners
                # purple color for junction points
                pt_idx_new1 = pts[0]
                pt_idx1 = to_org_pt_idx[pts[0]]
                pt_idx_new2 = pts[1]
                pt_idx2 = to_org_pt_idx[pts[1]]
                p_idx1, seg_idx1, t1 = endpoint_addresses_update[pt_idx1]
                p_idx2, seg_idx2, t2 = endpoint_addresses_update[pt_idx2]
                seg1 = paths[p_idx1][seg_idx1]
                seg2 = paths[p_idx2][seg_idx2]
                if is_parallel((seg1, t1), (seg2, t2), distance2) == False:
                    pts_old = [to_org_pt_idx[pt] for pt in pts]
                    avg_point = get_avg_point(
                        pts_old, endpoints_update, endpoint_addresses_update)
                    if avg_point in pt_to_idx:
                        continue
                    pt_detect.append(avg_point)
                    pt_to_idx[avg_point] = endpoints_find_counter
                    endpoints_find_counter += 1
                    # build up the point-stroke, stroke-stroke, stroke-point graphs
                    # no overshoting, sharp corner
                    if is_endpoint_t(t1, -0.045) and is_endpoint_t(t2, -0.045):
                        pt_colors.append("green")
                    # overshoting case 1
                    elif not is_endpoint_t(t1, -0.045) and is_endpoint_t(t2, -0.045):
                        t11 = push_pt_to_end(t1)
                        t22 = push_pt_to_end(t2)
                        # pt_to_stroke = push_to_dict(pt_to_stroke, avg_point, (p_idx2, seg_idx2, t22))
                        # stroke_to_pt = push_to_dict(stroke_to_pt, (p_idx2, seg_idx2, t22), avg_point)
                        # stroke_to_stroke = push_to_dict(stroke_to_stroke, (p_idx2, seg_idx2, t22), (p_idx1, seg_idx1, t11))
                        pt_to_stroke = push_to_dict(
                            pt_to_stroke, avg_point, (p_idx2, seg_idx2))
                        stroke_to_pt = push_to_dict(
                            stroke_to_pt, (p_idx2, seg_idx2), avg_point)
                        stroke_to_stroke = push_to_dict(
                            stroke_to_stroke, (p_idx2, seg_idx2), (p_idx1, seg_idx1))
                        # stroke_to_stroke = push_to_dict(stroke_to_stroke, (p_idx1, seg_idx1, t11), (p_idx2, seg_idx2, t22))
                        pt_colors.append("purple")
                    # overshoting case 2
                    elif not is_endpoint_t(t2, -0.045) and is_endpoint_t(t1, -0.045):
                        t11 = push_pt_to_end(t1)
                        t22 = push_pt_to_end(t2)
                        # pt_to_stroke = push_to_dict(pt_to_stroke, avg_point, (p_idx1, seg_idx1, t11))
                        # stroke_to_pt = push_to_dict(stroke_to_pt, (p_idx1, seg_idx1, t11), avg_point)
                        # stroke_to_stroke = push_to_dict(stroke_to_stroke, (p_idx1, seg_idx1, t11), (p_idx2, seg_idx2, t22))
                        pt_to_stroke = push_to_dict(
                            pt_to_stroke, avg_point, (p_idx1, seg_idx1))
                        stroke_to_pt = push_to_dict(
                            stroke_to_pt, (p_idx1, seg_idx1), avg_point)
                        stroke_to_stroke = push_to_dict(
                            stroke_to_stroke, (p_idx1, seg_idx1), (p_idx2, seg_idx2))
                        # stroke_to_stroke = push_to_dict(stroke_to_stroke, (p_idx2, seg_idx2, t22), (p_idx1, seg_idx1, t11))
                        pt_colors.append("purple")
                    # no overshoting, closed junction
                    else:
                        pt_colors.append("purple")
            else:
                pts_old = [to_org_pt_idx[pt] for pt in pts]
                avg_point = get_avg_point(
                    pts_old, endpoints_update, endpoint_addresses_update)
                if avg_point in pt_to_idx:
                    continue
                pt_detect.append(avg_point)
                end_pt = [is_endpoint_t(
                    endpoint_addresses_update[idx][-1], -0.04) for idx in pts_old]
                if sum(end_pt) == len(pts_old):
                    pt_colors.append("green")
                else:
                    # overshoting case3, multi strokes involved
                    sts = {}
                    # find all closed end points on each stroke to the avg point
                    for ptid in pts_old:
                        pid, sid, t = endpoint_addresses_update[ptid]
                        dist2 = abs(paths[pid][sid].point(t) - avg_point)
                        if (pid, sid) not in sts:
                            sts[(pid, sid)] = (t, dist2)
                        else:
                            _, dist1 = sts[(pid, sid)]
                            if dist1 > dist2:
                                sts[(pid, sid)] = (t, dist2)
                    # build up the point-stroke, stroke-stroke, stroke-point graphs
                    for st1, pt1 in sts.items():
                        pid1, sid1 = st1
                        t1, _ = pt1
                        t1 = push_pt_to_end(t1)
                        # pt_to_stroke = push_to_dict(pt_to_stroke, avg_point, (pid1, sid1, t1))
                        # stroke_to_pt = push_to_dict(stroke_to_pt, (pid1, sid, t1), avg_point)
                        pt_to_stroke = push_to_dict(
                            pt_to_stroke, avg_point, (pid1, sid1))
                        stroke_to_pt = push_to_dict(
                            stroke_to_pt, (pid1, sid), avg_point)
                        # for st2, pt2 in sts.items():
                        #     pid2, sid2 = st2
                        #     t2, _ = pt2
                        #     if pid1 == pid2 and sid1 == sid2 and t1 == t2: continue
                        #     stroke_to_stroke = push_to_dict(stroke_to_stroke, (pid1, sid1, t1), (pid2, sid2, t2))
                        #     stroke_to_stroke = push_to_dict(stroke_to_stroke, (pid2, sid2, t2), (pid1, sid1, t1))
                    pt_colors.append("purple")
                pt_to_idx[complex(avg_point)] = endpoints_find_counter
                endpoints_find_counter += 1

    # build up the table for point to point relations
    # this could solve the overshoting problem
    # 3rd pass, find all loop points and merge them if the distance between them under distance3
    pt_to_pt = {}
    for pt1 in pt_to_stroke:
        for st1 in pt_to_stroke[pt1]:
            if st1 in stroke_to_stroke:
                for st2 in stroke_to_stroke[st1]:
                    if st2 in stroke_to_pt:
                        for pt2 in stroke_to_pt[st2]:
                            if pt1 != pt2:
                                pt_to_pt = push_to_dict(pt_to_pt, pt1, pt2)
    # also, if one stroke mapped to multiple points, those points are already have connection to each other
    for _, pts in stroke_to_pt.items():
        if len(pts) > 1:
            for pt1 in pts:
                for pt2 in pts:
                    if pt1 == pt2:
                        continue
                    pt_to_pt = push_to_dict(pt_to_pt, pt1, pt2)

    # refine the points list the last time
    loops = []
    for pt in pt_to_pt:
        loop = [pt]
        if get_loop(pt_to_pt, loop):
            loops.append(loop)
    pt_detect, pt_colors = merge_loops(
        loops, pt_to_idx, pt_detect, pt_colors, distance3, pt_to_stroke)

    # 4th loop, to detect all x junctions
    x_junc_idx = len(pt_detect)
    check_x_junction(paths, pt_detect, pt_colors)
    # compute intersection point inside each path
    for i in range(len(paths)):
        if len(paths[i]) > 0:
            check_x_junction(paths[i], pt_detect, pt_colors, seg_mode = True)
    # remove duplicate detections 
    pt_neighbours = endpoint_detection([(pt.real, pt.imag) for pt in pt_detect], distance2)
    skip_list = []
    for pts in pt_neighbours:
        if len(pts) == 1: continue
        elif len(pts) == 2:
            for i in pts:
                if i >= x_junc_idx: skip_list.append(i)
    skip_list = sorted(set(skip_list), reverse = True)
    for idx in skip_list:
        pt_detect.pop(idx)
        pt_colors.pop(idx)
    # save detection result into different svg files
    pts_endpoint = []
    pts_sharpturn = []
    pts_T_junctions = []
    pts_X_junctions = []
    for i in range(len(pt_detect)):
        if pt_colors[i] == "red":
            pts_endpoint.append(pt_detect[i])
        if pt_colors[i] == "green":
            pts_sharpturn.append(pt_detect[i])
        if pt_colors[i] == "purple":
            pts_T_junctions.append(pt_detect[i])
        if pt_colors[i] == "black":
            pts_X_junctions.append(pt_detect[i])
    if to_svg:
        pt_num = len(pts_endpoint)
        if pt_num > 0:
            # let's try a smaller radius
            wsvg(node_radii=[distance2]*pt_num, nodes=pts_endpoint,
                 node_colors=["red"]*pt_num, filename=join(path_to_result, name) + "_1.svg")
            set_canvas_from(root_org, join(path_to_result, name) + "_1.svg")

        pt_num = len(pts_sharpturn)
        if pt_num > 0:
            wsvg(node_radii=[distance2]*pt_num, nodes=pts_sharpturn,
                 node_colors=["green"]*pt_num, filename=join(path_to_result, name) + "_2.svg")
            set_canvas_from(root_org, join(path_to_result, name) + "_2.svg")

        pt_num = len(pts_T_junctions)
        if pt_num > 0:
            wsvg(node_radii=[distance2]*pt_num, nodes=pts_T_junctions,
                 node_colors=["purple"]*pt_num, filename=join(path_to_result, name) + "_3.svg")
            set_canvas_from(root_org, join(path_to_result, name) + "_3.svg")

        '''this is not necessary any more, we should use sketch skeleton instead of it'''
        # pt_num = len(pts_smooth)
        # if pt_num > 0:
        #     wsvg(node_radii=[distance2]*pt_num, nodes=pts_smooth,
        #          node_colors=["black"]*pt_num, filename=join(path_to_result, name) + "_4.svg")
        #     set_canvas_from(root_org, join(path_to_result, name) + "_4.svg")

        pt_num = len(pts_X_junctions)
        if pt_num > 0:
            wsvg(node_radii=[distance2]*pt_num, nodes=pts_X_junctions,
                 node_colors=["black"]*pt_num, filename=join(path_to_result, name) + "_4.svg")
            set_canvas_from(root_org, join(path_to_result, name) + "_4.svg")
        return 0
    else:
        # let's still save all keypoints with different category, ready for future work
        # be aware that here is x, y so when generating GT, we need to swith the coordination to h, w later!
        res1 = np.array([[a.real, a.imag] for a in pts_endpoint])
        res2 = np.array([[a.real, a.imag] for a in pts_sharpturn])
        res3 = np.array([[a.real, a.imag] for a in pts_T_junctions])
        res4 = np.array([[a.real, a.imag] for a in pts_X_junctions])
        res5 = np.array([[a.real, a.imag] for a in pt_detect])
        out_npz_path = join(path_to_result, name) + ".npz"
        np.savez_compressed(out_npz_path, end_point=res1, sharp_turn=res2,
                 t_junction=res3, x_junction=res4, all=res5)
        return res1, res2, res3, res4
    
    '''
    For Debug
        Insert clean sketch and rough sketch for better visualization
    '''
    # insert the detection result as a new layer
    # insert_rough_clean(doc, name)

def check_x_junction(paths, pt_detect, pt_colors, seg_mode = False):
    for i in range(len(paths)):
        for j in range(i, len(paths)):
            if i == j: continue
            # compute the intersection between two strokes
            if paths[i] == paths[j]: continue
            try:
                x_junction = paths[i].intersect(paths[j])
            except:
                x_junction = []
            # the intersection may not be only one
            if len(x_junction) != 0:
                if seg_mode:
                    for t1, t2 in x_junction:
                        if (t1 > 1e-3 and t1 < 0.999) or (t2 > 1e-3 and t2 < 0.999):
                            pt_detect.append(paths[i].point(t1))
                            pt_colors.append("black")
                else:
                    for (T1, seg1, t1), (T2, seg2, t2) in x_junction:
                        pt_detect.append(paths[i].point(T1))
                        pt_colors.append("black")

def merge_loops(loops, pt_to_idx, pts, pt_colors, dist, pt_to_st):
    remove_idx = []
    if len(loops) == 0:
        return pts, pt_colors
    for loop in loops:
        if len(loop) == 1:
            continue
        # involved_sts = []
        # for pt in loop:
        #     involved_sts += pt_to_st[pt]
        # if len(set(involved_sts)) > 2:
        #     dist = dist/2
        pt_idx = []
        avg_point = loop.pop(0)
        skip = False
        for pt in loop:
            pt_diff = avg_point - pt
            if abs(pt_diff) > dist:
                skip = True
                break
            pt_idx.append(pt_to_idx[pt])
            avg_point += pt
        if skip:
            continue
        avg_point /= (len(loop)+1)
        pt_idx.sort()
        pts[pt_idx[0]] = avg_point
        pt_colors[pt_idx[0]] = "green"
        pt_idx.pop(0)
        remove_idx += pt_idx
    pts_new = []
    pt_colors_new = []
    for i in range(len(pts)):
        if i in remove_idx:
            continue
        pts_new.append(pts[i])
        pt_colors_new.append(pt_colors[i])
    return pts_new, pt_colors_new


def get_loop(pt_to_pt, pt, dep=5):
    if dep < 0:
        return False
    pt2 = pt_to_pt[pt[-1]][0]
    if pt2 in pt_to_pt:
        if pt_to_pt[pt2][0] == pt[0]:
            pt.append(pt2)
            return True
        else:
            pt.append(pt2)
            return get_loop(pt_to_pt, pt, dep - 1)
    else:
        return False


def set_canvas_from(root_org, path_target_svg):
    tree_temp = ET.parse(path_target_svg)
    root_temp = tree_temp.getroot()
    for key in root_org.attrib:
        root_temp.attrib[key] = root_org.attrib[key]
    tree_temp.write(path_target_svg)


def push_pt_to_end(t):
    if t > 0.5:
        return 1
    else:
        return 0


def push_to_dict(t_dict, key, value):
    if key in t_dict:
        if value not in t_dict[key]:
            t_dict[key].append(value)
    else:
        t_dict[key] = [value]
    return t_dict


def get_avg_point(pts, pts_all, pts_addr_all):
    avg_point = complex(*(0, 0))
    pt_count = 0
    juc_pt = []
    for pt_idx in pts:
        _, _, pt = pts_addr_all[pt_idx]
        if is_endpoint_t(pt) == False:
            juc_pt.append(pt_idx)
            pt_count += 1
    if pt_count == 0:
        for pt_idx in pts:
            avg_point += complex(*pts_all[pt_idx])
        assert len(pts) != 0
        avg_point = avg_point / len(pts)
    else:
        for pt_idx in juc_pt:
            avg_point += complex(*pts_all[pt_idx])
        assert len(juc_pt) != 0
        avg_point = avg_point / len(juc_pt)
    return avg_point


def to_svg_debug1(seg_with_t, name="test"):
    paths = []
    nodes = []
    for p, t in seg_with_t:
        paths.append(p)
        nodes.append(p.point(t))

    wsvg(paths, node_radii=[2]*len(nodes), node_colors=["red"]
         * len(nodes), nodes=nodes, filename=name + "_debug.svg")


def to_svg_debug2(points, paths, name="test"):
    nodes = []
    is_complex = False
    for pt in points:
        if type(pt) == complex:
            nodes = points
            break
        nodes.append(complex(*(pt)))
    wsvg(paths, node_radii=[0.5]*len(nodes), node_colors=["red"]
         * len(nodes), nodes=nodes, filename=name + "_debug.svg")


def insert_rough_clean(doc, name, debug=False):
    if debug:
        doc_detection = ET.parse(name + "_debug.svg")
    else:
        doc_detection = ET.parse(name + "_detection.svg")
    detection_layer = ET.Element("{http://www.w3.org/2000/svg}g")
    detection_layer.set("id", "detections")
    tree_detection = doc_detection.getroot()
    if len(tree_detection.findall('{http://www.w3.org/2000/svg}path')) == 0:
        for p in tree_detection.findall('{http://www.w3.org/2000/svg}circle'):
            detection_layer.append(p)
    else:
        detection_layer.append(doc_detection.getroot()[1])

    tree = doc.getroot()
    tree.append(detection_layer)
    clean_layer = tree.find('{http://www.w3.org/2000/svg}g')
    clean_layer.set("id", "clean")

    doc_rough = ET.parse(
        join("images", "_".join(name.split("_")[:-3]) + ".svg"))
    tree_rough = doc_rough.getroot()
    rough_layer = tree_rough.find('{http://www.w3.org/2000/svg}g')
    if rough_layer is None:
        rough_layer = ET.Element("{http://www.w3.org/2000/svg}g")
        for p in tree_rough.findall('{http://www.w3.org/2000/svg}path'):
            rough_layer.append(p)
    rough_layer.set("id", "rough")
    tree.append(rough_layer)

    if debug:
        doc.write(name + "_debug.svg")
    else:
        doc.write(name + "_detection.svg")
    return None


def draw_box(center, radii=1):
    tl = center + complex(-radii, -radii)  # top left
    tr = center + complex(radii, -radii)  # top right
    br = center + complex(radii, radii)  # bottom right
    bl = center + complex(-radii, radii)  # bottom left
    return Path(Line(tl, tr), Line(tr, br), Line(br, bl), Line(bl, tl))


def endpoint_generate(paths):
    '''
    Given:
        paths, the list of path elements of whole SVG
    Return:
        endpoints, a list of tuple of point coordination, which get from the end point of each segment
        endpoint_addtresses, a list of segment index and point index corresponding to the endpoints list
    '''
    endpoints = []  # a copy of point coordinations
    endpoint_addresses = []

    for path_index, path in enumerate(paths):
        for seg_index, seg in enumerate(path):
            for t in (0, 1):
                pt = seg.point(t)
                endpoints.append((pt.real, pt.imag))
                endpoint_addresses.append((path_index, seg_index, t))
    return endpoints, endpoint_addresses


def read_svg(path_to_svg, distance1, distance2, distance3):
    filepath, svg = split(path_to_svg)
    name, _ = splitext(svg)
    # what this scale does?
    global_scale = 1.0
    try:
        doc = Document(path_to_svg)
        flatpaths = doc.flatten_all_paths()
        paths = [path for (path, _, _) in flatpaths]
    except:
        doc = Document(path_to_svg)
        global_scale = get_global_scale(doc.tree)
        # Let's truly fail if there are transform nodes we can't handle.
        # try: global_scale = get_global_scale( doc.tree )
        # except: print( "WARNING: There are transforms, but flatten_all_paths() failed. Falling back to unflattened paths and ignoring transforms.", file = sys.stderr )
        paths, _ = svg2paths(path_to_svg)
    for i in range(len(paths)):
        if paths[i] == Path():
            paths.pop(i)

    # convert relative distance1 to real distance1
    if 'viewBox' in doc.root.attrib:
        import re
        _, _, width, height = [float(v) for v in re.split(
            '[ ,]+', doc.root.attrib['viewBox'].strip())]
        width = float(width)
        height = float(height)
        long_edge = max(width, height)
        diagonal = math.sqrt(width ** 2 + height ** 2)

    elif "width" in doc.root.attrib and "height" in doc.root.attrib:
        width = float(doc.root.attrib["width"].strip().strip("px"))
        height = float(doc.root.attrib["height"].strip().strip("px"))
        long_edge = max(width, height)
        diagonal = math.sqrt(width ** 2 + height ** 2)
    else:
        raise ValueError("Can't find viewBox or Width&Height info!")

    # why compute like this?
    # distance1 = distance1 * long_edge /1000 / global_scale
    # distance2 = distance2 * long_edge /1000 / global_scale
    # distance3 = distance3 * long_edge /1000 / global_scale
    assert global_scale != 0
    distance1 = distance1 * diagonal / global_scale
    distance2 = distance2 * diagonal / global_scale
    distance3 = distance3 * diagonal / global_scale

    return paths, filepath, name, distance1, distance2, distance3, ET.parse(path_to_svg)


def is_parallel(path1_with_t, path2_with_t, dist):
    return is_parallel_tangent(path1_with_t, path2_with_t, dist)
    # return is_parallel_curvature(path1_with_t, path2_with_t, dist)


def is_parallel_curvature(path1_with_t, path2_with_t, dist):
    # let's compute the curvature instead of tangent
    seg1, t1 = path1_with_t
    seg2, t2 = path2_with_t
    p_temp = Path()
    if is_endpoint_t(t1) and is_endpoint_t(t2):
        # use the curvature() is not always work, since it requires path continuity
        # so we need to compute the curvature mannually
        # let's say the length of most segments will longer than 5
        len1 = seg1.length()
        len2 = seg2.length()
        f = 1
        if len1 < dist or len2 < dist:
            dist /= 20
            f = 20
        if (len1 < dist or len2 < dist):
            print("Log:\tignore one (or two) too short stroke(s)")
            return True
        assert len1 != 0 and len2 != 0
        tx = (len1 - dist) / len1 if t1 > 0.8 else dist / len1
        tw = 1 if t1 > 0.8 else 0
        ty = (len2 - dist) / len2 if t2 > 0.8 else dist / len2
        x = seg1.point(tx)
        x1, y1 = x.real, x.imag
        w = seg1.point(tw)
        x2, y2 = w.real, w.imag
        y = seg2.point(ty)
        x3, y3 = y.real, y.imag
        if on_same_line(x, w, y, f):
            return True
        # let's use this method
        # https://www.geeksforgeeks.org/equation-of-circle-when-three-points-on-the-circle-are-given/
        a1 = ((x1**2 - x2**2) + (y1**2 - y2**2)) / 2
        a2 = ((x1**2 - x3**2) + (y1**2 - y3**2)) / 2
        x0 = ((y1 - y2)*a2 - (y1 - y3)*a1) / ((y1-y2)*(x1-x3)-(x1-x2)*(y1-y3))
        y0 = ((x1 - x3)*a1 - (x1 - x2)*a2) / ((y1-y2)*(x1-x3)-(x1-x2)*(y1-y3))
        c = complex(*(x0, y0))
        cur = 1 / (abs(x - c) * (f ** 2))
        if cur < 1 / dist:
            return True
        else:
            return False
    else:
        return False


def is_parallel_tangent(path1_with_t, path2_with_t, dist):
    # compute the average tangent
    p1 = path1_with_t
    p2 = path2_with_t
    # to_svg_debug([p1,p2])
    if is_endpoint_t(p1[1]) and is_endpoint_t(p2[1]):
        # make sure the two segments have the same t direction
        # which means one endpoint should always followed by a start point or vice versa
        if abs(p1[1] - p2[1]) < 0.9:
            reverse = True
        else:
            reverse = False
        tan1 = get_tangent(p1, reverse)
        tan2 = get_tangent(p2)
        cosine = dot(tan1, tan2)
        if cosine > math.cos(45/180*math.pi):  # 0.95
            return True
        else:
            return False
    else:
        return False


def get_tangent(path_with_t, reverse=False):
    direction = -1 if reverse else 1
    seg, t = path_with_t
    if t > 0.9:
        t_range = np.arange(0.9, 1, 0.01)
    elif t < 0.1:
        t_range = np.arange(0, 0.1, 0.01)
    tangents = []
    for pt in t_range:
        tangents.append(direction * seg.unit_tangent(pt))
    tangents = np.array(tangents)
    return np.mean(tangents)


def is_endpoint_t(t, offset=0):
    # sometimes we need release the threshold value by an offset
    if t > 0.95-offset or t < 0.05+offset:
        return True
    else:
        return False


def open_junction_detection(paths, dist, pts, pts_addr, pts_find, to_org_pt_idx):
    pts_find_udpate = pts_find.copy()
    endpoints_update = pts.copy()
    endpoint_addresses_update = pts_addr.copy()
    to_org_update = to_org_pt_idx.copy()
    junction_idx_start = len(pts_find)
    # initialize all segments to aabbtree
    bbtree = build_aabbtree([])
    pt_idx = 0
    for path_index, path in enumerate(paths):
        for seg_index, seg in enumerate(path):
            xmin, xmax, ymin, ymax = seg.bbox()  # record bbox of each segmentation?
            bbtree.add(AABB([(xmin, xmax), (ymin, ymax)]),
                       (path_index, seg_index, seg, pt_idx, pt_idx + 1))
            pt_idx += 2

    # search through all end points
    bbox_edge_half = dist  # box size is twice as the dist
    # for i, (pt, ( path_index, seg_index, t )) in enumerate(zip( pts, pts_addr )):
    pt_idx_add = len(pts)
    for i_find, pt in enumerate(pts_find):
        # get point and its corresponding segment
        i = to_org_pt_idx[i_find]
        path_index, seg_index, t = pts_addr[i]
        current_seg = paths[path_index][seg_index]
        # AABB((xmin, xmax), (ymin, ymax))
        query = AABB([(pt[0] - bbox_edge_half, pt[0] + bbox_edge_half),
                     (pt[1] - bbox_edge_half, pt[1] + bbox_edge_half)])
        min_j_dist = float('inf')
        min_t = None
        min_path_index = None
        min_seg_index = None
        min_seg = None
        min_pt_idx1 = None
        min_pt_idx1 = None

        # find the closest segment that hitted by this query
        res = bbtree.overlap_values(query)
        for other_path_index, other_seg_index, seg, pt_idx1, pt_idx2 in res:
            # skip the endpoint itself, it will always be detected
            if other_path_index == path_index and other_seg_index == seg_index:
                continue
            try:
                j_dist, j_t = seg.radialrange(complex(*pt))[0]
            except Exception as e:
                # sometimes there will be noisy strokes that have lenght of 0
                # print(str(e))
                continue
            # we try to find all points that is less than the threshold
            if min_j_dist > j_dist:
                min_j_dist = j_dist
                min_t = j_t
                min_path_index = other_path_index
                min_seg_index = other_seg_index
                min_seg = seg
                min_pt_idx1 = pt_idx1
                min_pt_idx2 = pt_idx2

        # record all found junction points
        if min_j_dist <= dist:
            pt_found = min_seg.point(min_t)
            pts_find_udpate.append((pt_found.real, pt_found.imag))
            endpoints_update.append((pt_found.real, pt_found.imag))
            endpoint_addresses_update.append(
                (min_path_index, min_seg_index, min_t))
            to_org_update.append(pt_idx_add)
            pt_idx_add += 1

    return pts_find_udpate, endpoints_update, endpoint_addresses_update, to_org_update, junction_idx_start


def endpoint_detection(endpoints, distance):
    # print( "Creating spatial data structures:" )

    dist_finder = scipy.spatial.cKDTree(endpoints)

    endpoints_find = np.array(dist_finder.query_ball_tree(
        dist_finder, distance), dtype=object)
    try:
        endpoints_find = np.unique(endpoints_find.astype(int), axis=0)
    except:
        endpoints_find = np.unique(endpoints_find)
        assert np.dtype("O") == endpoints_find.dtype

    # if np.dtype("O") == endpoints_find.dtype:
    #     endpoints_find = np.unique(endpoints_find)
    # else:
    #     endpoints_find = np.unique(endpoints_find, axis = 0)

    return endpoints_find.tolist()


def endpoint_close(paths, distance):
    # Gather all endpoints, path index, segment index, t value
    endpoints, endpoint_addresses = endpoint_generate(paths)

    endpoints_find = endpoint_detection(endpoints, distance)

    snapped = []
    # Close open junctions
    for pts in endpoints_find:
        if len(pts) == 1:
            continue
        # first pass to get average point that is used to snap junctions
        avg_point = complex()
        pc = 0  # point counter
        for i in range(len(pts)):
            if endpoint_addresses[pts[i]] not in snapped:
                path, seg, t = endpoint_addresses[pts[i]
                                                  ][0], endpoint_addresses[pts[i]][1], endpoint_addresses[pts[i]][2]
                avg_point += paths[path][seg].point(t)
                pc += 1

        if avg_point != complex() and pc > 1:  # if there have points that need to be snapped
            avg_point = avg_point / pc
            # second pass to set new segments
            for i in range(len(pts)):
                path, seg, t = endpoint_addresses[pts[i]
                                                  ][0], endpoint_addresses[pts[i]][1], endpoint_addresses[pts[i]][2]
                if endpoint_addresses[pts[i]] not in snapped:
                    snapped.append(endpoint_addresses[pts[i]])
                    endpoints[pts[i]] = (avg_point.real, avg_point.imag)
                    if t == 0:
                        new_seg = set_segment_by_point(
                            paths[path][seg], start=avg_point)
                    elif t == 1:
                        new_seg = set_segment_by_point(
                            paths[path][seg], end=avg_point)
                    else:
                        raise ValueError("Invalid point value %f" % t)
                    paths[path][seg] = new_seg

    return paths, snapped


def t_junction_close(paths, distance, snapped):
    def t_junction_close_recursive(pt_with_index, distance, pre_seg_with_index, paths, bbtree, depth, snapped):
        # end the recursion if max depth is reached or all points are snapped
        if depth == 0:
            return paths
        if pt_with_index[1] in snapped:
            return paths

        depth = depth - 1
        bbox_edge = 2 * distance  # box size is twice as the distance
        pt = pt_with_index[0]
        path_index, seg_index, t = pt_with_index[1]
        query = AABB([(pt[0] - bbox_edge, pt[0] + bbox_edge),
                     (pt[1] - bbox_edge, pt[1] + bbox_edge)])
        min_j_dist = float('inf')
        min_t = None
        min_path_index = None
        min_seg_index = None
        min_seg = None
        for other_path_index, other_seg_index, seg in bbtree.overlap_values(query):
            if other_path_index == path_index and other_seg_index == seg_index:
                continue
            try:
                j_dist, j_t = seg.radialrange(complex(*pt))[0]
            except Exception as e:
                print(str(e))
                continue
            if min_j_dist > j_dist:
                min_j_dist = j_dist
                min_t = j_t
                min_path_index = other_path_index
                min_seg_index = other_seg_index
                min_seg = seg

        # if find target segment
        if min_j_dist < distance and min_j_dist > eps:
            # if the fixment of current pt(seg) depends on pre_seg, then fixment of pre_seg also depends on current pt
            # in other word, seg and pre_seg need to be fixed that the same time
            if (min_path_index, min_seg_index) == pre_seg_with_index[1]:
                # find closest point of two segmet endpoints to each other
                # cloeset point on min_seg(pre_seg) to cur_seg endpoint
                point1 = min_seg.point(min_t)
                # cloeset point on cur_seg to min_seg(pre_seg) endpoint
                t1 = 0 if min_t < 0.5 else 1
                dist2, t2 = paths[path_index][seg_index].radialrange(min_seg.point(t1))[
                    0]
                point2 = paths[path_index][seg_index].point(t2)

                # point2 should also satisfy the distance requirement
                assert(dist2 < distance and dist2 > eps)
                # fix both segments
                avg_point = (point1 + point2) / 2
                # set current segment
                if t == 0:
                    new_seg_1 = set_segment_by_point(
                        paths[path_index][seg_index], start=avg_point)
                elif t == 1:
                    new_seg_1 = set_segment_by_point(
                        paths[path_index][seg_index], end=avg_point)
                else:
                    raise ValueError("Invalid point value %f" % t)
                paths[path_index][seg_index] = new_seg_1

                # set previous segment
                if t1 == 0:
                    new_seg_2 = set_segment_by_point(min_seg, start=avg_point)
                elif t1 == 1:
                    new_seg_2 = set_segment_by_point(min_seg, end=avg_point)
                else:
                    raise ValueError("Invalid point value %f" % t1)
                paths[min_path_index][min_seg_index] = new_seg_2

                # return result
                return paths

            else:
                org_seg = paths[path_index][seg_index]
                # call it self recursively by two endpoints of min_seg to find if there is addtional dependency
                pt_start = ((min_seg.start.real, min_seg.start.imag),
                            (min_path_index, min_seg_index, 0))
                pre_seg = (paths[path_index][seg_index],
                           (path_index, seg_index))
                paths = t_junction_close_recursive(
                    pt_start, distance, pre_seg, paths, bbtree, depth, snapped)

                pt_end = ((min_seg.end.real, min_seg.end.imag),
                          (min_path_index, min_seg_index, 1))
                paths = t_junction_close_recursive(
                    pt_end, distance, pre_seg, paths, bbtree, depth, snapped)

                # generate current new segments after all denpent segments are fixed
                if org_seg == paths[path_index][seg_index]:
                    t_point = paths[min_path_index][min_seg_index].point(min_t)
                    if t == 0:
                        new_seg = set_segment_by_point(
                            paths[path_index][seg_index], start=t_point)
                    elif t == 1:
                        new_seg = set_segment_by_point(
                            paths[path_index][seg_index], end=t_point)
                    else:
                        raise ValueError("Invalid point value %f" % t)
                    paths[path_index][seg_index] = new_seg

                return paths
        else:
            # nothing need to change, return paths directly
            return paths

    endpoints, endpoint_addresses = endpoint_generate(paths)
    bbtree = build_aabbtree(paths)

    # for each point, find the hit segments and fix them if necessary
    for i, (pt, (path_index, seg_index, t)) in enumerate(zip(endpoints, endpoint_addresses)):
        pt_with_index = (pt, (path_index, seg_index, t))
        pre_seg_with_index = (None, None)
        paths = t_junction_close_recursive(
            pt_with_index, distance, pre_seg_with_index, paths, bbtree, 10, snapped)

    return paths

def build_aabbtree(paths):
    # Build an axis-aligned bounding box tree for the segments.
    bbtree = AABBTree()

    for path_index, path in enumerate(paths):
        for seg_index, seg in enumerate(path):
            xmin, xmax, ymin, ymax = seg.bbox()
            bbtree.add(AABB([(xmin, xmax), (ymin, ymax)]),
                       (path_index, seg_index, seg))
    return bbtree

if __name__ == '__main__':
    import argparse
    __spec__ = None
    parser = argparse.ArgumentParser(description="Detect *CLOSED* junctions")
    parser.add_argument("path_to_svg", help="The SVG file to analyze.")
    parser.add_argument("-o", help="The ouput result")
    parser.add_argument("-x", action="store_true", help="detect x junctions")
    parser.add_argument(
        "-th1", help="the distance threshold to identify a pair of closed endpoints", default=0.0001)
    parser.add_argument(
        "-th2", help="the distance threshold to identify a closed junction", default=0.00095)
    parser.add_argument(
        "-th3", help="the distance threshold to identify a junction loop", default=0.007)

    args = parser.parse_args()

    # detect single svg image
    if ".svg" in args.path_to_svg:
        detect_svg(args.path_to_svg, args.th1, args.th2, args.th3, args.o, to_svg = True)
    # or detect all svg images in a folder
    else:
        '''signle process version'''
        # for img in tqdm(os.listdir(args.path_to_svg)):
        #     detect_svg(join(args.path_to_svg, img), args.th1,
        #                args.th2, args.th3, args.o, to_svg=False)

        '''multiprocess version'''
        # let's add multiprocessing
        # thanks for https://stackoverflow.com/questions/5442910/how-to-use-multiprocessing-pool-map-with-multiple-arguments
        # we use multiprocessing pool
        multi_args = []
        if args.o is None:
            args.o = args.path_to_svg
        for svg in os.listdir(args.path_to_svg):
            if exists(join(args.o, svg.replace(".svg", ".npz"))): continue
            if 'svg' not in svg: continue
            multi_args.append((join(args.path_to_svg, svg), args.th1, args.th2, args.th3, args.o, False, True))
            # detect_svg(join(args.path_to_svg, svg), args.th1, args.th2, args.th3, args.o, False, True)
        with Pool(7) as pool:
            pool.starmap(detect_svg, multi_args)
