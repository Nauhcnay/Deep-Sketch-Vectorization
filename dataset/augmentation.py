from pathlib import Path as P
import sys
import cv2
import time
directory = P(__file__)
directory = str(directory.parent.parent)
if directory not in sys.path:
    sys.path.append(directory)

import numpy as np
from utils.svg_tools import crop_path, get_hw_offset

def flip_edge(edge_map, axis):
    '''
    flip numpy array along given axis, the input edge map should has shape like (H, W, 2)
    the last channel stores the edge flag along different 
    '''
    edge_map_x = edge_map[:, :, 0]
    edge_map_y = edge_map[:, :, 1]
    if axis == 'x':
        # flip along x-aixs
        edge_map_x = np.flip(edge_map_x, axis = 0)
        edge_map_y = np.flip(edge_map_y, axis = 0)
        # but move y along y-axis!
        # this is based on how you understand the flip and roll
        edge_map_x = np.roll(edge_map_x, 1, axis = 0) 
        res = np.stack((edge_map_x, edge_map_y), axis = -1)
    elif axis == 'y':
        # flip along y-aixs
        edge_map_x = np.flip(edge_map_x, axis = 1)
        edge_map_y = np.flip(edge_map_y, axis = 1)
        # but move x along y-axis!
        edge_map_y = np.roll(edge_map_y, 1, axis = 1) 
        res = np.stack((edge_map_x, edge_map_y), axis = -1)
    else:
        raise ValueError("not supported mode %s!"%axis)
    return res

def flip_keypt_map(keypt_map, axis, canvas_size, index = 'xy'):
    h, w = canvas_size
    # move image center to 0, 0
    h_offset, w_offset = get_hw_offset((0, 0, h, w))
    offset = np.array([[[w_offset, h_offset]]])

    # convert ij index to xy
    if index == 'ij':
        keypt_map[:,:,[0, 1]] = keypt_map[:,:,[1, 0]]
    else:
        if index != 'xy':
            raise ValueError("index mode should only be xy or ij")

    # move origin point to the center of the canvas
    # res = keypt_map[:-1, :-1, ...]
    res = keypt_map
    res = res - offset

    if axis == 'y' :
        res[:, :, 0] = res[:, :, 0] * -1
        res = np.flip(res, axis = 1)
        # res = np.roll(res, -1, axis = 1)
    elif axis == 'x':
        res[:, :, 1] = res[:, :, 1] * -1
        res = np.flip(res, axis = 0)
        # res = np.roll(res, -1, axis = 0)
    else:
        raise ValueError("unsupported axis %s!"%axis)

    # move image center back
    res += offset
    # keypt_map[:-1, :-1, ...] = res
    
    if index == 'ij':
        res[:,:,[0, 1]] = res[:,:,[1, 0]]
    return res



def rotate_edge():
    # this function may be non-trivia
    pass

def rotate_keypt_map(keypt_map, ang):
    h, w = img_size
    # move image center to 0, 0
    h_offset, w_offset = get_hw_offset((0, 0, h, w))
    res = keypt_map.reshape((-1, 2)) - np.array([[h_offset, w_offset]])
    # make rotaton
    res = np.matmul(get_rotate_matrix(ang), res.T).T
    # translate point coordinates back to image coord system   
    res += np.array([[h_offset, w_offset]])
    return res.reshape(keypt_map.shape)

def random_bbox(canvas_size, gsize = 1, bbox_size = 256, edge_maps = None, junc_pts = None, udf_mode = False):
    '''
    generate a bbox of given size 
    '''
    bbox_size += 1
    h, w = canvas_size
    if bbox_size >= h or bbox_size >= w:
        raise ValueError("input image should not smaller than the bounding box")
    if edge_maps is not None:
        edge = np.logical_or(edge_maps[:,:,0], edge_maps[:,:,1])
        h, w = edge.shape
        anchors = []
        flag_nums = []
        have_juncs = []
        visited = np.zeros((h,w)).astype(bool)
        assert h == int(canvas_size[0] / gsize)
        assert w == int(canvas_size[1] / gsize)
        if udf_mode:
            bbox_size = int((bbox_size - 1) / gsize) + 1
        # scan all possible sampling locations
        search_step = (h - bbox_size) if (h - bbox_size) < (w - bbox_size) else (w - bbox_size)
        for i in range(0, h - bbox_size + 1):
            for j in range(0, w - bbox_size + 1):
                sbox = np.random.randint(low = int(search_step / 16), high = int(search_step / 8))
                if visited[i, j]: continue
                fnum = edge[i:i+bbox_size, j:j+bbox_size].sum()
                fnum_ = edge[i:i+bbox_size - 1, j:j+bbox_size - 1].sum()
                i_ = i if i % 2 == 0 else i - 1
                j_ = j if j % 2 == 0 else j - 1
                fnum__ = edge[i_:i_+bbox_size - 1, j_:j_+bbox_size - 1].sum()
                has_junc = False
                if junc_pts is not None:
                    bbox_real = (i*gsize, j*gsize, (i+bbox_size)*gsize, (j+bbox_size)*gsize)
                    has_junc = len(crop_path(junc_pts, bbox_real, canvas_size)) > 0
                visited[i:i+sbox, j:j+sbox] = True
                if fnum > 0 and fnum_ > 0 and fnum__ > 0:
                    anchors.append([i, j])
                    flag_nums.append(fnum)
                    if has_junc:
                        have_juncs.append(True)
                    else:
                        have_juncs.append(False)
        
        if len(anchors) == 0:
            i = h - bbox_size
            j = w - bbox_size
            anchors.append([i, j])
            flag_nums.append(edge[i:i+bbox_size, j:j+bbox_size].sum())
        idxs = np.argsort(flag_nums)
        
        if True in have_juncs:
            dice = np.random.uniform()
            if dice < 0.8:
                idxs = np.intersect1d(np.where(have_juncs)[0], idxs)
            else:
                idxs = idxs[int(len(anchors)/2):]    
        else:
            idxs = idxs[int(len(anchors)/2):]
        idx = np.random.choice(idxs)
        top, left = anchors[idx]
        if udf_mode:
            if top % 2 != 0: top -=1
            if left % 2 != 0: left -=1
        # assert edge[top:top+bbox_size, left:left+bbox_size].sum() > 0
        # assert edge[top:top+bbox_size - 1, left:left+bbox_size - 1].sum() > 0
        bottom = top + bbox_size
        right = left + bbox_size
        bbox0 = (top, left, bottom - 1, right - 1)
        bbox1 = (top, left, bottom, right)
    else:
        # generate bbox at random locations
        def bbox(h, w, size, udf_mode = False):
            if udf_mode:
                assert (bbox_size - 1) % 2 == 0
                top = np.random.randint(low = 1, high = h - size)
                left = np.random.randint(low = 1, high = w - size)
                if top % 2 != 0: top -= 1
                if left % 2 != 0: left -= 1
            else:
                top = np.random.randint(low = 0, high = h - size)
                left = np.random.randint(low = 0, high = w - size)
            bottom = top + size
            right = left + size
            bbox0 = (top, left, bottom - 1, right - 1)
            bbox1 = (top, left, bottom, right)
            return bbox0, bbox1
        bbox0, bbox1 = bbox(h, w, bbox_size, udf_mode)
    return bbox0, bbox1

def crop_edge(edge_map, bbox):
    top, left, bottom, right = bbox
    return edge_map[top:bottom, left:right, ...]

def crop_keypt_map(keypt_map, bbox, gsize):
    '''
    Given:
        keypt_map:  A float numpy array with shape (h, w, 2), it stores the 
                    keypoint location in each grid, the locaion of the format
                    (x, y)
        bbox:       A list or tuple that contains the bounding box
    '''
    top, left, bottom, right = bbox
    offset = np.array([[[left * gsize, top * gsize]]])
    res = keypt_map[top:bottom, left:right, ...]
    res = res - offset
    return res

def blend_skeletons(skel_gt, skel_pre, alpha = 0.5, usm_mode = False):

    green = np.array([0, 255, 0])
    red = np.array([255, 0, 0])
    blue = np.array([0, 0, 255])
    black = np.array([0, 0, 0])
    white = np.array([255, 255, 255])
    if usm_mode:
        usm, usm_gt = skel_pre
        if len(skel_gt.shape) == 2:
            svg_img = np.expand_dims(skel_gt, axis = -1).repeat(3, axis = -1)
        else:
            assert len(skel_gt.shape) == 3
            svg_img = skel_gt
        usm_img = np.array([white, red])
        usm_gt_img = np.array([white, blue])
        usm_img = usm_img[usm]
        usm_gt_img = usm_gt_img[usm_gt]
        h, w, _ = svg_img.shape
        usm_gt_img = cv2.resize(usm_gt_img, (w, h), interpolation = cv2.INTER_NEAREST)
        usm_img = cv2.resize(usm_img, (w, h), interpolation = cv2.INTER_NEAREST)
        svg_img = (svg_img * alpha  + usm_gt_img * (1 - alpha))
        svg_img = (svg_img * alpha  + usm_img * (1 - alpha))
        return svg_img.astype(np.uint8)
    else:
        skel_gt_img = np.array([white, black])
        skel_pre_img = np.array([white, green])
        skel_gt_img = skel_gt_img[skel_gt]
        skel_pre_img = skel_pre_img[skel_pre]
        return (skel_gt_img * alpha  + skel_pre_img * (1 - alpha)).astype(np.uint8)
