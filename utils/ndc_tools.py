from pathlib import Path as P
import sys
import os
directory = os.path.realpath(os.path.dirname(__file__))
directory = str(P(directory).parent)
if directory not in sys.path:
    sys.path.append(directory)

import torch
import time
import numpy as np
import cv2
import xml.etree.ElementTree as ET
import ctypes
import multiprocessing
import argparse
import copy

from numpy import asarray
from os.path import split
from aabbtree import AABB
from aabbtree import AABBTree
from torch.nn import functional as F
from os import path

from torch import nn
from svgpathtools import Path, Line, wsvg, svg2paths, CubicBezier, Document
from PIL import Image
from multiprocessing import Pool
from tqdm import tqdm

from utils.svg_tools import open_svg_flatten, resize_path, get_line_approximation, compute_dist_M
import edge_distance_aabb
from utils.dual_contouring import norm
from scipy.signal import convolve2d
from scipy.ndimage import correlate
from colorama import Fore
from skimage.morphology import skeletonize

def get_pt_mask(edge_map_x, edge_map_y):
    edge_map_x = edge_map_x.astype(bool)
    edge_map_y = edge_map_y.astype(bool)
    grid_mask_x = roll_edge(edge_map_x, -1, 'x').astype(int) + edge_map_x.astype(int)
    grid_mask_y = roll_edge(edge_map_y, -1, 'y').astype(int) + edge_map_y.astype(int)
    grid_mask_all = (grid_mask_x.astype(bool) | grid_mask_y.astype(bool)).astype(float)
    return grid_mask_x.astype(int), grid_mask_y.astype(int), grid_mask_all

def get_valence_np(edge_map_x, edge_map_y):
    edge_map_x = edge_map_x.astype(bool)
    edge_map_y = edge_map_y.astype(bool)
    valence_x, valence_y, pt_mask = get_pt_mask(edge_map_x, edge_map_y)

    # sum up all valences for each grid
    valence_map = valence_x + valence_y
    assert valence_map.max() <= 4

    # should we adjust the counter numbers accrodingly?
    return valence_map, pt_mask

def init_base_coord(pt_map, device, gsizes):
    '''
    Given:
        pt_map, batch x 2 x height x width tensor, stores the center point of each UDF grid
        device, string which should be "cuda" or "cpu", decide where to store this base coord map
    Action:
        return base_coord
    Note:
        height and width from the corresponding UDF's shape, remember that 4 UDF points forms on grid
        therefore the point map size should always be height -1 and width -1
    '''
    b, _, h, w = pt_map.shape
    base_coord = []
    for gsize in gsizes:
        gsize = float(gsize)
        hg = int(h * gsize)
        wg = int(w * gsize)
        assert hg == h * gsize
        assert wg == w * gsize
        xs = torch.linspace(start = gsize / 2, end = wg - gsize / 2, steps = w) 
        ys = torch.linspace(start = gsize / 2, end = hg - gsize / 2, steps = h)
        xx, yy = torch.meshgrid(xs, ys, indexing = 'xy')
        base_coord.append(torch.stack((xx, yy), dim = -1).unsqueeze(0).permute(0, 3, 1, 2))
    base_coord = torch.cat(base_coord, dim = 0)
    return base_coord.to(device)
    
def get_udf_mask(gt, axis = 'x'):
    if axis == 'x':
        gt_mask_x = np.roll(gt, 1, axis = 1)
        gt_mask_x[:, 0] = False
        return np.logical_or(gt, gt_mask_x)
    if axis == 'y':
        gt_mask_y = np.roll(gt, 1, axis = 0)
        gt_mask_y[0, :] = False
        return np.logical_or(gt, gt_mask_y)
    if axis == 'xy':
        gt_mask_x = np.roll(gt, 1, axis = 1)
        gt_mask_y = np.roll(gt, 1, axis = 0)
        gt_mask_xy = np.roll(gt_mask_y, 1, axis = 1)
        gt_mask_xy[0, :] = False
        gt_mask_xy[:, 0] = False
        gt_mask_x[:, 0] = False
        gt_mask_y[0, :] = False
        gt_mask = np.logical_or(gt, gt_mask_x)
        gt_mask = np.logical_or(gt_mask, gt_mask_y)
        gt_mask = np.logical_or(gt_mask, gt_mask_xy)
        return gt_mask

def pre_to_map(edge_pre, stroke_mask = None):
    '''
    Given:
        edge_pre:   A int tensor with shape (b, h, w, 4) or (b, h, w, 1), the last dimension stores the edge flags, 
                    this function support both the edge values from network prediction tensor or the ground truth tensor 
                    , but only the first tensor format will not block the gradient

                    the value defination of edge flag ground truth:
                    0: None, 1: edge flag along x-axis, 2: edge flag along y-axis, 3: edge flag along both axis
                    
                    PLEASE NOTE: the edge flag along x-axis will produce a polyline in y-axis direction! cause the polyline will
                    go across the grid edge. this could be kind of confusing...
    Return:
        edge_map_x: A float tensor with shape (b, h, w, 1), the edge flag along x-axis
        edge_map_y: A float tensor with shape (b, h, w, 1), the edge flag along y-axis
        edge_map:   A float tensor with shape (b, h, w, 4), the full edge flag in one hot vector format
    '''
    softmax = torch.nn.Softmax(dim = -1)
    relu = torch.nn.ReLU()

    # check which type of the edge flag is
    c = edge_pre.shape[-1]
    if c == 4:
        edge_map_temp = softmax(edge_pre)
        edge_mask = edge_map_temp.argmax(dim = -1)
        edge_mask = F.one_hot(edge_mask, num_classes=4)
        
        # https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836
        # push the choosen value (the edge flag prediction) greater than 1 and the rest less than 0
        edge_map_temp = (edge_map_temp + 2) * edge_mask +  edge_map_temp * (1 - edge_mask)
        edge_map_temp = edge_map_temp - 1
        
        # we have to do this to avoid inplace tensor assignment
        edge_map = torch.clamp(relu(edge_map_temp), min = 0, max = 1)
        edge_map_x = edge_map[...,1] + edge_map[...,3]
        edge_map_y = edge_map[...,2] + edge_map[...,3]

    
    elif c == 2:
        edge_mask = (torch.sigmoid(edge_pre) > 0.5).float()
        edge_prediction = torch.sigmoid(edge_pre)
        edge_prediction = (edge_prediction + 1) * edge_mask + (edge_prediction - 1) * (1 - edge_mask)
        edge_map = torch.clamp(relu(edge_prediction), min = 0, max = 1)
        edge_map_x = edge_map[...,0]
        edge_map_y = edge_map[...,1]
    
    else:
        # generate edges from GT, this branch will break the gradient
        assert c == 1
        edge_map = edge_pre.squeeze(dim = -1).long()
        edge_map_x = torch.logical_or((edge_map == 1), (edge_map == 3)).float()
        edge_map_y = torch.logical_or((edge_map == 2), (edge_map == 3)).float()
        edge_map = F.one_hot(edge_map, num_classes = 4).float()
    
    if stroke_mask is not None:
        edge_map_x = edge_map_x * stroke_mask.squeeze().long()
        edge_map_y = edge_map_y * stroke_mask.squeeze().long()
    return edge_map_x, edge_map_y, edge_map

def pre_to_lines(edge_pre, keypt_map, edge_mask = None):
    '''
    Given:
        edge_pre:   A float tensor with shape (h, w, 4) or (h, w, 1) as the edge prediction value 
                    or ground truth edge flags
        keypt_map:  A float tensor with shape (h, w, 2), the last dimension stores the coordinates 
                    of center points of each grid, the fromat of each coordinate should be (x, y)
    Return:
        res:        A float tensor with shape (k, 2), it stores the start and end point of each line
                    reconstruction, so the k should always satisfy k % 2 == 0
    '''    
    # todo: also return a valence map that indicate all segement point's valence
    assert len(edge_pre.shape) == 3
    h, w, c = edge_pre.shape
    assert c % 2 == 0 or c == 1
    edge_map_x, edge_map_y, _ = pre_to_map(edge_pre, edge_mask)

    # reconstruct all lines that intersect with edges along x-axis
    lines_v = [] 
    for i in range(1, h):
        line_mask = edge_map_x[i, :].unsqueeze(-1).unsqueeze(0).expand(2, w, 2)
        if line_mask.sum() > 0 and line_mask.sum()%4 == 0:
            line_pts = keypt_map[i - 1:i + 1, :, :] * line_mask
            # here the keypoint sequence is A1, B1, C1, ... A2, B2, C2, ...
            pt_mask = line_mask.mean(dim = (0, 2)).bool().unsqueeze(0).expand(2, -1)
            temp = line_pts[pt_mask].view(-1, 1).split(2)
            idx = int(len(temp) / 2)
            line_pts_s = torch.stack(temp[:idx], dim = 0)
            line_pts_e = torch.stack(temp[idx:], dim = 0)
            line_pts_n = torch.stack([line_pts_s, line_pts_e], dim = 1)
            lines_v.append(line_pts_n.reshape(-1, 2))

    # reconstruct all lines that intersect with edges along y-axis
    lines_h = []
    for i in range(1, w):
        # get the ith line mask, edge_map_y should always has shape h x w
        line_mask = edge_map_y[:, i].unsqueeze(-1).unsqueeze(-1).expand(h, 2, 2)
        if line_mask.sum() > 0 and line_mask.sum()%4 == 0:
            line_pts = keypt_map[:, i - 1:i + 1, :] * line_mask
            pt_mask = line_mask.mean(dim = (1, 2)).bool()
            lines_h.append(line_pts[pt_mask].reshape(-1, 2))
    
    # return the results
    res = lines_h + lines_v
    try:
        return torch.cat(res)
    except:
        return None

def pre_to_lines_ver2(edge_map_x, edge_map_y, pt_map_x, pt_map_y):
    h, w = edge_map_x.shape
    assert edge_map_y.shape == (h, w)
    assert pt_map_x.shape == (h, w, 2)
    assert pt_map_y.shape == (h, w, 2)
    # we need a record of lines at each grid edge
    edge_records = np.zeros((h, w, 4)).astype(bool).astype(object)
    lines = []
    line_idx = 0
    line_idx = scan_grids(edge_map_x, edge_map_y, pt_map_x, pt_map_y, lines, line_idx, edge_records, second_pass = False)
    scan_grids(edge_map_x, edge_map_y, pt_map_x, pt_map_y, lines, line_idx, edge_records,second_pass = True)
    return np.array(lines)

def scan_grids(edge_map_x, edge_map_y, pt_map_x, pt_map_y, lines, line_idx, edge_records, second_pass = False):
    h, w = edge_map_x.shape
    for i in range(h - 1):
        for j in range(w - 1):
            # get edge flag of current grid
            top = edge_map_x[i][j]
            top_pt = pt_map_x[i][j]
            bottom  = edge_map_x[i + 1][j]
            bottom_pt = pt_map_x[i + 1][j]
            left = edge_map_y[i][j]
            left_pt = pt_map_y[i][j]
            right = edge_map_y[i][j + 1]
            right_pt = pt_map_y[i][j + 1]
            has_strokes = np.array([top, left, bottom, right]).astype(bool)
            point_on_edges = np.array([top_pt, left_pt, bottom_pt, right_pt])
            # compute the valence of current grid
            valence = has_strokes.sum()
            if valence <= 1: 
                continue
            elif valence == 2 and second_pass == False:
                if top:
                    assert (top_pt != -1).all()
                    lines.append(top_pt)
                    edge_records[i][j][0] = line_idx
                if left:
                    assert (left_pt != -1).all()
                    lines.append(left_pt)
                    edge_records[i][j][1] = line_idx
                if bottom:
                    assert (bottom_pt != -1).all()
                    lines.append(bottom_pt)
                    edge_records[i][j][2] = line_idx
                if right:
                    assert (right_pt != -1).all()
                    lines.append(right_pt)
                    edge_records[i][j][3] = line_idx
                line_idx += 2
            elif valence == 3 and second_pass == False:
                # get center point of this grid
                avg_pt = point_on_edges[has_strokes].mean(axis = 0)
                # connect each edge point to the center point respectively
                if top:
                    lines.append(top_pt)
                    lines.append(avg_pt)
                    edge_records[i][j][0] = line_idx
                    line_idx += 2
                if left:
                    lines.append(left_pt)
                    lines.append(avg_pt)
                    edge_records[i][j][1] = line_idx
                    line_idx += 2
                if bottom:
                    lines.append(bottom_pt)
                    lines.append(avg_pt)
                    edge_records[i][j][2] = line_idx
                    line_idx += 2
                if right:
                    lines.append(right_pt)
                    lines.append(avg_pt)
                    edge_records[i][j][3] = line_idx
                    line_idx += 2
            elif valence == 4:
                if second_pass:
                    # get lines around 
                    line_top_idx = edge_records[i - 1][j][2]
                    line_top_dir = lines[line_top_idx] - lines[line_top_idx + 1]
                    line_top_dir /= np.linalg.norm(line_top_dir)

                    line_left_idx = edge_records[i][j - 1][3]
                    line_left_dir = lines[line_left_idx] - lines[line_left_idx + 1]
                    line_left_dir /= np.linalg.norm(line_left_dir)
                    
                    line_bottom_idx = edge_records[i + 1][j][0]
                    line_bottom_dir = lines[line_bottom_idx] - lines[line_bottom_idx + 1]
                    line_bottom_dir /= np.linalg.norm(line_bottom_dir)

                    line_right_idx = edge_records[i][j + 1][1]
                    line_right_dir = lines[line_right_idx] - lines[line_right_idx + 1]
                    line_right_dir /= np.linalg.norm(line_right_dir)

                    # compare the direction between
                    dot_top_left = abs(np.dot(line_top_dir, line_left_dir))
                    dot_top_bottom = abs(np.dot(line_top_dir, line_bottom_dir))
                    dot_top_right = abs(np.dot(line_top_dir, line_right_dir))
                    junction_case = np.argmax([dot_top_left, dot_top_bottom, dot_top_right])
                    if junction_case == 0:
                        lines.append(top_pt)
                        lines.append(left_pt)
                        lines.append(right_pt)
                        lines.append(bottom_pt)
                    elif junction_case == 1:
                        lines.append(top_pt)
                        lines.append(bottom_pt)
                        lines.append(left_pt)
                        lines.append(right_pt)
                    else:
                        lines.append(top_pt)
                        lines.append(right_pt)
                        lines.append(left_pt)
                        lines.append(bottom_pt)
    return line_idx

def align_keypt(pt_map, key_pts, valence_map, val, pt_type, idx_to_keypts, thr = 1.5):
    ndc_pts = pt_map[np.where(valence_map == val)].reshape(-1, 2)
    if len(ndc_pts) == 0:
        return key_pts
    pt_idxs = np.array(np.where(valence_map == val)).T # idx are in (i, j) format
    
    # compute distance matrix between to point groups
    # row is the key points
    # column is the ndc grid points
    M_dist = compute_dist_M(ndc_pts, key_pts)
    
    # for each DC vertex, find the closest keypoint
    # M_idx = np.argsort(M_dist, axis = 0)
    M_idx = np.argmin(M_dist, axis = 0)
    # assert ( M_idx[0,:] == np.argmin(M_dist, axis = 0) ).all()

    # get the closest distance for each keypoint
    ndc_pts_update = []
    match_idx = M_dist[M_idx, np.arange(M_dist.shape[1])]
    for i in range(len(match_idx)):
        # update point type map if we find one match
        if match_idx[i] < thr:
            ndc_idx = M_idx[i]
            pt_type[tuple(pt_idxs[ndc_idx].T)] = val
            # this key point is save as (x, y) format
            idx_to_keypts[tuple(pt_idxs[ndc_idx].T)] = key_pts[i]
            ndc_pts_update.append(pt_map[tuple(pt_idxs[ndc_idx].T)])
    return np.array(ndc_pts_update)

def gen_sample_idx(size, ratio, head = True, offset = 0):
    # location could be: head/tail
    assert size % ratio == 0
    if head:
        return np.arange(0 + offset, size, ratio)
    else:
        return np.arange(ratio - (1 + offset), size, ratio)
    
def roll_by_axis(array, shift = -1, axis = 'x'):
    assert len(array.shape) == 2
    assert shift < 0
    arr = array.copy()
    if axis == 'x':
        arr[0:abs(shift), :] = False
        return np.roll(arr, shift, axis = 0)
    elif axis == 'y':
        arr[:, 0:abs(shift)] = False
        return np.roll(arr, shift, axis = 1)
    else:
        raise ValueError("Axis should only be 'x' or 'y' while get %s"%axis)

def apply_kernel_biase(kernel, axis, size):
    if axis == 'x':
        mask = np.zeros(kernel.shape).astype(bool)
        mask[:, 0:int(size/2)] = True
        kernel[mask] = kernel[mask] * 2
    elif axis == 'y':
        mask = np.zeros(kernel.shape).astype(bool)
        mask[0:int(size/2), :] = True
        kernel[mask] = kernel[mask] * 2
    else:
        assert axis is None
    return kernel

# https://stackoverflow.com/questions/56948729/how-to-create-a-triangle-kernel-in-python
def triangle_kernel(size = 9, biased = None):
    r = np.arange(size)
    kernel1d = (size + 1 - np.abs(r - r[::-1])) / 2
    kernel2d = np.outer(kernel1d, kernel1d)
    kernel2d = apply_kernel_biase(kernel2d, biased, size)
    kernel2d = kernel2d / kernel2d.sum()
    return kernel2d

def cross_kernel(size = 9, biased = None):
    r = np.arange(size)
    kernel1d = (size + 1 - np.abs(r - r[::-1])) / 2
    kernel2d = np.zeros((size, size))
    kernel2d[int(size / 2), :] = kernel1d
    kernel2d[:, int(size / 2)] = kernel1d.T
    kernel2d = apply_kernel_biase(kernel2d, biased, size)
    kernel2d = kernel2d / kernel2d.sum()
    return kernel2d

def gaussian_kernel(size=5, sig=1.):
    """\
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    ax = np.linspace(-(size - 1) / 2., (size - 1) / 2., size)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel / np.sum(kernel)

def box_kernel(size = 9):
    return np.ones((size, size)) / size ** 2

def downsample_by_shift(in_array, axis, ratio, debug = False):
    if axis == 'x':
        down = downsample_by_axis(in_array, ratio, 'x') >= 1
        in_array_x = in_array.copy()
        in_array_x[-1,:] = False
        in_array_x = np.roll(in_array_x, 1, axis = 0)
        down_x = downsample_by_axis(in_array_x, ratio, 'x') >= 1
        if debug:
            print(Fore.BLUE+"log:\t org edge flag"+Fore.WHITE)
            print(down)
            print(Fore.BLUE+"log:\t vertically shifted edge flag"+Fore.WHITE)
            print(down_x)
        down = down | down_x
    elif axis == 'y':
        down = downsample_by_axis(in_array, ratio, 'y') >= 1
        in_array_y = in_array.copy()
        in_array_y[:,-1] = False
        in_array_y = np.roll(in_array_y, 1, axis = 1)
        down_y = downsample_by_axis(in_array_y, ratio, 'y') >= 1
        if debug:
            print(Fore.BLUE+"log:\t org edge flag"+Fore.WHITE)
            print(down)
            print(Fore.BLUE+"log:\t horizontally shifted edge flag"+Fore.WHITE)
            print(down_y)
        down = down | down_y
    else:
        raise ValueError("axis only support x or y!")
    return down

def downsample_by_conv(in_array, kernel, axis = 'all', ratio = 2, debug = False):
    h, w = in_array.shape[0], in_array.shape[1]
    assert h % ratio == 0
    assert w % ratio == 0
    h_new = int(h / ratio)
    w_new = int(w / ratio)
    in_array_ = convolve2d(in_array, kernel, mode = 'same', boundary = 'fill', fillvalue = 0)
    if axis == 'x':
        down = in_array_[gen_sample_idx(h, ratio), :].reshape(h_new, w_new, ratio).sum(axis = 2)
    elif axis == 'y':
        down = in_array_[:, gen_sample_idx(w, ratio)].reshape(h_new, ratio, w_new).sum(axis = 1)
    elif axis == 'all':
        down = in_array_.reshape(h, w_new, ratio).transpose(0,2,1).reshape(h_new, ratio ** 2, w_new).transpose(0,2,1).sum(axis = -1)
    else:
        raise ValueError("wrong axis!")
    down_none_zero = down[down!=0]
    # thr = (down_none_zero.max() + down_none_zero.min()) * 0.55
    thr = down_none_zero.mean()
    if debug:
        print('\n')
        print(Fore.BLUE+"log:\t input edge flag"+Fore.WHITE)
        print(in_array)
        print(Fore.BLUE+"log:\t output convoluted flag map"+Fore.WHITE)
        print(in_array_)
        print(Fore.BLUE+"log:\t output after pooling"+Fore.WHITE)
        print(down)
        print(Fore.BLUE+"log:\t current threshold"+Fore.WHITE)
        print(thr)
        print(Fore.BLUE+"log:\t downsampled edge flag"+Fore.WHITE)
        print(down > thr)
    # down = downsample_by_axis(in_array, ratio, axis = 'x')
    return down > thr

def downsample_edge(in_array_x, in_array_y, axis, ratio = 2, debug = False):
    h, w = in_array_x.shape[0], in_array_x.shape[1]
    assert in_array_x.shape ==  in_array_y.shape
    assert h % ratio == 0
    assert w % ratio == 0
    h_new = int(h / ratio)
    w_new = int(w / ratio)
    if axis == 'x':
        ## for x-axis edge flag
        # sample first row of each grid
        down_x_head = in_array_x[gen_sample_idx(h, ratio, head = True), :].reshape(h_new, w_new, ratio)
        # sample second row of each grid
        down_x_second = in_array_x[gen_sample_idx(h, ratio, head = True, offset = 1), :].reshape(h_new, w_new, ratio)
        # sample last row of each grid
        down_x_tail = in_array_x[gen_sample_idx(h, ratio, head = False), :].reshape(h_new, w_new, ratio)
        # set the last row to 0
        down_x_tail[-1, ...] = False
        down_x_tail = np.roll(down_x_tail, shift = 1, axis = 0)
        edge_x = down_x_head.sum(axis = 2) >= 1
        gap_x_01 = ((down_x_second + down_x_tail) == 2).any(axis = 2)
        ## for y-axis edge flag
        # sample first row of each grid
        down_y_head = in_array_y[gen_sample_idx(h, ratio, head = True), :].reshape(h_new, w_new, ratio)
        # sample last row of each grid
        down_y_tail = in_array_y[gen_sample_idx(h, ratio, head = False), :].reshape(h_new, w_new, ratio)
        down_y_tail[-1, ...] = False
        down_y_tail = np.roll(down_y_tail, shift = 1, axis = 0)
        gap_x_02 = ((down_y_head + down_y_tail) > 0).sum(axis = 2) >= int(ratio * 0.75 )
        gap_x_02 = (gap_x_02 & np.roll(down_y_tail.sum(axis = 2) >= 1, shift = 1, axis = 1)) & (gap_x_02 & ~(down_y_tail.sum(axis = 2) >= 1))
        if debug:
            print('\n')
            print(Fore.BLUE+"log:\t input edge flag"+Fore.WHITE)
            print(in_array_x)
            print(Fore.BLUE+"log:\t output edge map"+Fore.WHITE)
            print(edge_x)
            print(Fore.BLUE+"log:\t output gap map 01"+Fore.WHITE)
            print(gap_x_01)
            print(Fore.BLUE+"log:\t current gap map 02"+Fore.WHITE)
            print(gap_x_02)
        return edge_x|gap_x_01|gap_x_02

    elif axis == 'y':
        ## for y-axis edge flag
        # sample first column of each grid
        down_y_head = in_array_y[:, gen_sample_idx(w, ratio, head=True)].reshape(h_new, ratio, w_new)
        # sample second column of each grid
        down_y_second = in_array_y[:, gen_sample_idx(w, ratio, head = True, offset = 1)].reshape(h_new, ratio, w_new)
        # sample last column of each grid
        down_y_tail = in_array_y[:, gen_sample_idx(w, ratio, head = False)].reshape(h_new, ratio, w_new)
        # set the last column to 0
        down_y_tail[..., -1] = False
        down_y_tail = np.roll(down_y_tail, shift = 1, axis = 1)
        # get edge flag along y-axis
        edge_y = down_y_head.sum(axis = 1) >= 1
        gap_y_01 = ((down_y_second + down_y_tail) == 2).any(axis = 1)
        ## for x-axis edge flag
        # sample first column of each grid
        down_x_head = in_array_x[:, gen_sample_idx(w, ratio, head = True)].reshape(h_new, ratio, w_new)
        # sample last column of each grid
        down_x_tail = in_array_x[:, gen_sample_idx(w, ratio, head = False)].reshape(h_new, ratio, w_new)
        down_x_tail[..., -1] = False
        down_x_tail = np.roll(down_x_tail, shift = 1, axis = 1)
        gap_y_02 = ((down_x_head + down_x_tail) > 0).sum(axis = 1) >= int(ratio * 0.75)
        gap_y_02 = (gap_y_02 & np.roll(down_x_tail.sum(axis = 1) >= 1, shift = 1, axis = 0)) & (gap_y_02 & ~(down_x_tail.sum(axis = 1) >= 1))
        if debug:
            print('\n')
            print(Fore.BLUE+"log:\t input edge flag"+Fore.WHITE)
            print(in_array_y)
            print(Fore.BLUE+"log:\t output edge map"+Fore.WHITE)
            print(edge_y)
            print(Fore.BLUE+"log:\t output gap map 01"+Fore.WHITE)
            print(gap_y_01)
            print(Fore.BLUE+"log:\t current gap map 02"+Fore.WHITE)
            print(gap_y_02)
        return edge_y|gap_y_01|gap_y_02
    else:
        raise ValueError("axis could only be one of ['x', 'y'] but not %s"%axis)

def downsample_by_axis(in_array, ratio = 2, axis = 'all', return_sum = True, head = True):
    h, w = in_array.shape[0], in_array.shape[1]
    assert h % ratio == 0
    assert w % ratio == 0
    h_new = int(h / ratio)
    w_new = int(w / ratio)
    if axis == 'all':
        down_all = in_array.reshape(h, w_new, ratio).transpose(0,2,1).reshape(h_new, ratio ** 2, w_new).transpose(0,2,1)
        return down_all.sum(axis = -1) if return_sum else down_all
    elif axis == 'x':
        # np.arange(0, h, ratio)
        down_x = in_array[gen_sample_idx(h, ratio, head), :].reshape(h_new, w_new, ratio)
        return down_x.sum(axis = 2) if return_sum else down_x
    elif axis == 'y':
        # np.arange(0, w, ratio)
        down_y = in_array[:, gen_sample_idx(w, ratio, head)].reshape(h_new, ratio, w_new)
        return down_y.sum(axis = 1) if return_sum else down_y
    else:
        raise ValueError("axis could only be one of ['all', 'x', 'y'] but not %s"%axis)

def expand_xy(array, mode = 'xy'):
    if mode == 'x':
        array_ = roll_by_axis(array, axis = 'x')
    elif mode == 'y':
        array_ = roll_by_axis(array, axis = 'y')
    elif mode == 'xy':
        array_ = roll_by_axis(array, axis = 'x') | roll_by_axis(array, axis = 'y')
    else:
        raise ValueError("expand mode only support one of the following options: 'x', 'y', 'xy' while got mode %s"%s)
    return array | array_

def get_downsampled_idx(h, w, ratio):
    iis = np.arange(h)
    jjs = np.arange(w)
    iiv, jjv = np.meshgrid(iis, jjs, indexing = 'ij')
    iiv = downsample_by_axis(iiv, ratio, return_sum = False)
    jjv = downsample_by_axis(jjv, ratio, return_sum = False)
    return iiv.flatten(), jjv.flatten()

def downsample_all(edge_map_x, edge_map_y, pt_map, usm, valence_map, pt_type, lines_map_x = None, 
    lines_map_y = None, ratio = 2, return_lines = True):
    kernel = triangle_kernel(size = 3)
    assert edge_map_x.shape == edge_map_y.shape
    assert edge_map_x.shape == usm.shape
    assert edge_map_x.shape == (pt_map.shape[0], pt_map.shape[1])
    h, w = edge_map_x.shape
    assert h % ratio == 0
    assert w % ratio == 0
    h_down = int(h / ratio)
    w_down = int(w / ratio)
    # expand usm to algin to its downsized version
    usm_pre_down = downsample_by_axis(usm, ratio, 'all') # USM from prediction
    # Image.fromarray(usm.astype(bool)).save("usm_pre_2x.png")
    usm_pre_down = usm_pre_down > 0 # we DON'T need to extend USM maps from the prediction, it has already done so.
    # Image.fromarray(usm_pre_down).save("usm_pre.png")
    # sample edge map along x-axis
    x_down = downsample_by_axis(edge_map_x, ratio, 'x')
    edge_x_down = x_down >= 1
    # edge_x_down = downsample_edge(edge_map_x, edge_map_y, 'x', ratio)
    # edge_x_down = downsample_by_shift(edge_map_x, axis = 'x', ratio = ratio)
    # edge_x_down = downsample_by_conv(edge_map_x, kernel, axis = 'x', ratio = ratio)
    # sample edge map along y-axis
    y_down = downsample_by_axis(edge_map_y, ratio, 'y')
    edge_y_down = y_down >=1
    # edge_y_down = downsample_edge(edge_map_x, edge_map_y, 'y', ratio)
    # edge_y_down = downsample_by_shift(edge_map_y, axis = 'y', ratio = ratio)
    # edge_y_down = downsample_by_conv(edge_map_y, kernel, axis = 'y', ratio = ratio)
    ## compute all USM brought by the downsampling
    valence_down = downsample_by_axis(valence_map, ratio, 'all', return_sum = False)
    pt_type_down = downsample_by_axis(pt_type, ratio, 'all', return_sum = False)
    top_down = x_down
    bottom_down = roll_by_axis(x_down, axis = 'x')
    left_down = y_down
    right_down = roll_by_axis(y_down, axis = 'y')
    # consider any grid contains more than 1 endpoints, or 1 endpoint with any other keypoints as under sampled region 
    usm_valence1 = ((pt_type_down == 1).sum(axis = -1) >= 1)
    usm_valence2 = (pt_type_down == 2).all(axis = -1)
    usm_left_down = (left_down > 1) | usm_valence1 & ~usm_valence2
    usm_top_down = (top_down > 1) | usm_valence1 & ~usm_valence2
    usm_bottom_down = (bottom_down > 1) | usm_valence1 & ~usm_valence2
    usm_right_down = (right_down > 1) | usm_valence1 & ~usm_valence2
    # usm_left_down = left_down > 1
    # usm_top_down = top_down > 1
    # usm_bottom_down = bottom_down > 1
    # usm_right_down = right_down > 1
    usm_4way_down = ((top_down >= 1).astype(int) + (bottom_down >= 1).astype(int) + 
        (left_down >= 1).astype(int) + (right_down >= 1).astype(int)) == 4
    usm_xy_down = (top_down + left_down) > 1
    # make sure the boundary of USM won't contain any under sampling cases
    usm_top_down = expand_xy(usm_top_down, 'x')
    usm_left_down = expand_xy(usm_left_down, 'y')
    usm_4way_down = expand_xy(usm_4way_down, 'xy')
    usm_downsample = usm_left_down | usm_right_down | usm_top_down | usm_bottom_down | usm_4way_down | usm_xy_down
    # check every USM region if it is really contains important structure (contain keypoints)
    usm_ = cv2.resize(usm_downsample.astype(np.uint8), (w, h), interpolation = cv2.INTER_NEAREST).astype(bool)
    _, regions = cv2.connectedComponents(usm_.astype(np.uint8), connectivity = 4)
    for r in np.unique(regions):
        if r == 0: continue
        r_mask = regions == r
        # drop current USM region if it doesn't contain any keypoint or it is too large
        if (pt_type[r_mask] > 0).sum() == 0:
          r_mask = cv2.resize(r_mask.astype(np.uint8), (w_down, h_down), interpolation = cv2.INTER_NEAREST)
          usm_downsample[r_mask] = False
    # Image.fromarray(usm_downsample).save("usm_down.png")
    usm_down = usm_pre_down | usm_downsample
    # usm_down = usm_pre_down
    ## sample pt map
    '''
    this will be kind of difficult...
    so we will need to have a new key point map and a new dual contouring reconstruction method
    '''
    # generate the resmaple idx
    # setting the edge flag by USM
    if return_lines:
        edge_x_down[usm_down] = False
        edge_y_down[usm_down] = False
        # the logic here is really annoying!
        # get line mask
        line_mask = cv2.resize(usm_down.astype(np.uint8), (w, h), interpolation = cv2.INTER_NEAREST).astype(bool)
        usm_down_x = (expand_xy(usm_down, 'x') | usm_down).copy()
        usm_down_y = (expand_xy(usm_down, 'y') | usm_down).copy()
        line_mask_x = cv2.resize(usm_down_x.astype(np.uint8), (w, h), interpolation = cv2.INTER_NEAREST).astype(bool)
        line_mask_y = cv2.resize(usm_down_y.astype(np.uint8), (w, h), interpolation = cv2.INTER_NEAREST).astype(bool)
        line_mask_x_base = line_mask_x.copy()
        line_mask_y_base = line_mask_y.copy()
        line_mask_x[gen_sample_idx(h, ratio, True), :] = False
        line_mask_y[:, gen_sample_idx(w, ratio, True)] = False
        line_mask = line_mask_x | line_mask_y | line_mask
        line_mask_x = line_mask | line_mask_y_base
        line_mask_y = line_mask | line_mask_x_base
        # Image.fromarray(line_mask_x).save("line_mask_x.png")
        # Image.fromarray(line_mask_y).save("line_mask_y.png")
        # Image.fromarray(line_mask).save("line_mask.png")
    ## compute downsampled center point map
    # generate center point map of each super grid
    # edge_x_down_ = edge_x_down.copy()
    # edge_x_down_[0,...] = False
    # edge_y_down_ = edge_y_down.copy()
    # edge_y_down_[..., 0] = False
    # pt_mask_center_ = edge_x_down|np.roll(edge_x_down_, shift = -1, axis = 0)|edge_y_down|np.roll(edge_y_down_, shift=-1, axis=1)
    pt_mask_center_2x = valence_map > 0
    pt_mask_center_2x_v1 = valence_map == 1
    pt_mask_center_2x_v3 = valence_map > 2
    pt_count_center = downsample_by_axis(pt_mask_center_2x, ratio)
    pt_mask_center = pt_count_center > 0 

    pt_count_center_v1 = downsample_by_axis(pt_mask_center_2x_v1, ratio)
    pt_mask_center_v1 = pt_count_center_v1 > 0
    
    pt_count_center_v1[pt_count_center_v1 == 0] = 1
    pt_map_masked_2x = pt_map * (pt_mask_center_2x)[..., np.newaxis]
    pt_map_masked_2x_v1 = pt_map * (pt_mask_center_2x_v1)[..., np.newaxis]
    iiv, jjv = get_downsampled_idx(h, w, ratio)
    
    pt_mask_center_v3 = downsample_by_axis(pt_mask_center_2x_v3, ratio) > 0
    pt_mask_center_v1 = np.logical_xor(pt_mask_center_v1, pt_mask_center_v3) & pt_mask_center_v1
    pt_map_down_center_v1 = pt_map_masked_2x_v1[(iiv, jjv)].reshape(-1, ratio**2, 2).sum(axis = 1).reshape(h_down, w_down, -1) / pt_count_center_v1[...,np.newaxis]
    pt_count_center[~pt_mask_center] = 1
    pt_map_down_center = pt_map_masked_2x[(iiv, jjv)].reshape(-1, ratio**2, 2).sum(axis = 1).reshape(h_down, w_down, -1) / pt_count_center[...,np.newaxis]
    pt_map_down_center = pt_map_down_center * (~pt_mask_center_v1)[..., np.newaxis] + pt_map_down_center_v1 * pt_mask_center_v1[..., np.newaxis]
    
    # pt_mask_center_add = (np.logical_xor(pt_mask_center_, pt_mask_center.astype(bool)) & pt_mask_center_)|(np.logical_xor(pt_mask_center_, pt_mask_center.astype(bool)) & pt_mask_center)
    # pt_map_down_center_add = pt_map[(iiv, jjv)].reshape(-1, ratio**2, 2).mean(axis = 1).reshape(h_down, w_down, -1) * pt_mask_center_add[...,np.newaxis]
    # pt_map_down_center = pt_map_down_center + pt_map_down_center_add
    if return_lines:
        # update top point maps
        pt_mask_top_2x = edge_map_x
        pt_map_top = pt_map * (pt_mask_top_2x)[..., np.newaxis]
        pt_mask_top = downsample_by_axis(pt_mask_top_2x, ratio, axis = 'x', head = True)
        pt_mask_top[pt_mask_top == 0] = 1
        pt_map_down_top = pt_map_top[gen_sample_idx(h, ratio, True), :, ...].reshape(h_down, w_down, ratio, 2).sum(axis = 2) / pt_mask_top[...,np.newaxis]
        # update left point maps
        pt_mask_left_2x = edge_map_y
        pt_map_left = pt_map * (pt_mask_left_2x)[..., np.newaxis]
        pt_mask_left = downsample_by_axis(pt_mask_left_2x, ratio, axis = 'y', head = True)
        pt_mask_left[pt_mask_left == 0] = 1
        pt_map_down_left = pt_map_left[:, gen_sample_idx(w, ratio, True), ...].reshape(h_down, ratio,w_down, 2).transpose(0,2,1,3).sum(axis = 2) / pt_mask_left[...,np.newaxis]
        # update bottom point maps
        pt_mask_bottom_2x = roll_by_axis(edge_map_x, axis = 'x')
        pt_map_bottom = pt_map * (pt_mask_bottom_2x)[..., np.newaxis]
        pt_mask_bottom = downsample_by_axis(pt_mask_bottom_2x, ratio, axis = 'x', head = False)
        pt_mask_bottom[pt_mask_bottom == 0] = 1
        pt_map_down_bottom = pt_map_bottom[gen_sample_idx(h, ratio, False), :, ...].reshape(h_down, w_down, ratio, 2).sum(axis = 2) / pt_mask_bottom[...,np.newaxis]    
        # update right point maps
        pt_mask_right_2x = roll_by_axis(edge_map_y, axis = 'y')
        pt_map_right = pt_map * (pt_mask_right_2x)[..., np.newaxis]
        pt_mask_right = downsample_by_axis(pt_mask_right_2x, ratio, axis = 'y', head = False)
        pt_mask_right[pt_mask_right == 0] = 1
        pt_map_down_right = pt_map_right[:, gen_sample_idx(w, ratio, False), ...].reshape(h_down, ratio, w_down, 2).transpose(0,2,1,3).sum(axis = 2) / pt_mask_right[...,np.newaxis]
        # extract the polylines need to be reserved
        
        l1 = lines_map_x[line_mask_x]
        l1 = np.concatenate(l1[l1 != False])
        l2 = lines_map_y[line_mask_y]
        l2 = np.concatenate(l2[l2 != False])
        lines_reserved = np.concatenate((l1, l2), axis = 0)
        
        # lines_to_svg(lines_reserved, 512, 512, "line_reserved.svg")
        # reconstruct from the downsampled UDF
        edge_map_down = cut_branch(np.stack((edge_x_down, edge_y_down), axis = -1))
        edge_x_down = edge_map_down[..., 0]
        edge_y_down = edge_map_down[..., 1]
        usm_down = expand_xy(usm_down)
        lines_down = []
        for i in range(h_down):
            for j in range(w_down):
                # test x-axis direction
                if edge_x_down[i][j] and i != 0:
                    # at the top USM boundary
                    if usm_down[i - 1][j] == False and usm_down[i][j] and (pt_map_down_center[i - 1][j] != pt_map_down_top[i][j]).any():
                        lines_down.append(pt_map_down_center[i - 1][j])
                        lines_down.append(pt_map_down_top[i][j])
                    # at the bottom USM boundary
                    elif usm_down[i - 1][j] and usm_down[i][j] == False and (pt_map_down_bottom[i - 1][j] != pt_map_down_center[i][j]).any():
                        lines_down.append(pt_map_down_bottom[i - 1][j])
                        lines_down.append(pt_map_down_center[i][j])
                    # no USM boudnary
                    elif usm_down[i - 1][j] == False and usm_down[i][j] == False and (pt_map_down_center[i - 1][j] != pt_map_down_center[i][j]).any():
                        lines_down.append(pt_map_down_center[i - 1][j])
                        lines_down.append(pt_map_down_center[i][j])
                    elif usm_down[i - 1][j] and usm_down[i][j] and (pt_map_down_bottom[i - 1][j] != pt_map_down_top[i][j]).any():
                        lines_down.append(pt_map_down_bottom[i - 1][j])
                        lines_down.append(pt_map_down_top[i][j])    
                if edge_y_down[i][j] and j != 0: # edge map y
                    # at the left USM boundary
                    if usm_down[i][j - 1] == False and usm_down[i][j] and (pt_map_down_center[i][j - 1] != pt_map_down_left[i][j]).any():
                        lines_down.append(pt_map_down_center[i][j - 1])
                        lines_down.append(pt_map_down_left[i][j])
                    # at the right USM boundary
                    elif usm_down[i][j - 1] and usm_down[i][j] == False and (pt_map_down_right[i][j - 1] != pt_map_down_center[i][j]).any():
                        lines_down.append(pt_map_down_right[i][j - 1])
                        lines_down.append(pt_map_down_center[i][j])
                    # no USM boudnary
                    elif usm_down[i][j - 1] == False and usm_down[i][j] == False and (pt_map_down_center[i][j - 1] != pt_map_down_center[i][j]).any():
                        lines_down.append(pt_map_down_center[i][j - 1])
                        lines_down.append(pt_map_down_center[i][j])
                    elif usm_down[i][j - 1] and usm_down[i][j] and (pt_map_down_right[i][j - 1] != pt_map_down_left[i][j]).any():
                        lines_down.append(pt_map_down_right[i][j - 1])
                        lines_down.append(pt_map_down_left[i][j])
        # combine all results and return
        lines_down = np.array(lines_down)
        # lines_to_svg(lines_down, 512, 512, "line_down.svg")
        # lines_res = lines_down
        lines_res = np.concatenate((lines_reserved, lines_down), axis = 0)
        line = lines_down

        return lines_res
    else:
        return edge_x_down, edge_y_down, usm_down, pt_map_down_center

def refine_topology(edge_map, pt_map, under_sampled_map, lines_map_x, 
    lines_map_y, keypts, 
    tensor_input = False, 
    down_rate = 2, 
    downsample = True, 
    manual = False,
    full_auto_mode = False):
    '''
    lines_map_x, line_map_y will be inplace modified
    '''
    kernel = np.array([[1, 1, 1],[1, 1, 1],[1, 1, 1]], dtype = np.uint8)
    if tensor_input:
        assert manual == False
        edge_map = edge_map.squeeze().to(bool).detach().cpu().numpy()
        pt_map = pt_map.squeeze().detach().cpu().numpy()
        under_sampled_map = under_sampled_map.squeeze().to(bool).detach().cpu().numpy()
    h, w = edge_map.shape[0], edge_map.shape[1]
    long_side = h if h > w else w
    ## register keypoints to pt map
    edge_map_x = edge_map[..., 0]
    edge_map_y = edge_map[..., 1]

    valence_map, pt_mask = get_valence_np(edge_map_x, edge_map_y)
    pt_type = np.zeros(valence_map.shape)
    idx_to_keypts = {}

    # align key points and update them back to dict
    if manual == False:
        keypts['end_point'] = align_keypt(pt_map, keypts['end_point'], valence_map, 1, pt_type, idx_to_keypts, 3.5)
        keypts['sharp_turn'] = align_keypt(pt_map, keypts['sharp_turn'], valence_map, 2, pt_type, idx_to_keypts, 3.5)
        # some times there could exists T or X junction but with just valence 2 vertexs
        align_keypt(pt_map, keypts['junc'], valence_map, 2, pt_type, idx_to_keypts)
        keypts['junc'] = align_keypt(pt_map, keypts['junc'], valence_map, 3, pt_type, idx_to_keypts, 3.5)
    # for the predicted keypoints, they should have been aligned, 
    # for the new keypoint, they are user input, should be accurate
    # therefore, there is no need to do the alignment
    # else:
    #     align_keypt(pt_map, keypts['end_point'], valence_map, 1, pt_type, idx_to_keypts, 3.5)
    #     align_keypt(pt_map, keypts['sharp_turn'], valence_map, 2, pt_type, idx_to_keypts, 3.5)
    #     align_keypt(pt_map, keypts['junc'], valence_map, 2, pt_type, idx_to_keypts)
    #     align_keypt(pt_map, keypts['junc'], valence_map, 3, pt_type, idx_to_keypts, 3.5)

    # remove possible holes
    if manual == False:
        _, regions_raw = cv2.connectedComponents(under_sampled_map.astype(np.uint8), connectivity = 4)
        under_sampled_map = cv2.dilate( under_sampled_map.astype(np.uint8), kernel, iterations = 1 )
        under_sampled_map = cv2.erode( under_sampled_map.astype(np.uint8), kernel, iterations = 1 )
        # split under sampled map into individual regions
        _, regions = cv2.connectedComponents(under_sampled_map.astype(np.uint8), connectivity = 4)
    else:
        regions = under_sampled_map

    under_sampled_map_applied = np.zeros(under_sampled_map.shape).astype(int)
    under_sampled_map_uncertain = np.zeros(under_sampled_map.shape).astype(int)

    # for each region reconstruct the topology
    if len(np.unique(regions)) == 1: 
        lines = linemap_to_lines(lines_map_x, lines_map_y)
        return lines, {}, under_sampled_map, under_sampled_map_uncertain
    
    # record current region need usm surgery
    refined_lines = []
    refined_dict = {}
    for idx in np.unique(regions):
        if idx == 0: continue # skip background
        # get the key point (if have) from the given region
        region = regions == idx
        # skip if the USM is too large or it connect two USM regions
        if manual == False:
            r = np.unique(regions_raw[region])
            r = r[r!=0]
            if (region.sum() > long_side / 50) or len(r) > 1:
                continue
        under_sampled_map_updated = np.zeros(under_sampled_map.shape, dtype = bool)
        under_sampled_map_updated[region] = True
        # get the connection points from the given region
        connection_pts = find_connection_points(valence_map != 0, pt_map, region.astype(int) * 255, under_sampled_map.astype(bool))
        if connection_pts == False: continue
        # update the region topology
        applied, added_lines = update_line_maps(lines_map_x, lines_map_y, pt_map, pt_mask, pt_type, connection_pts, region, 
            refined_lines, idx_to_keypts, under_sampled_map_updated, manual, keypts, full_auto_mode)
        if applied is not None:
            if applied:
                under_sampled_map_applied[under_sampled_map_updated] = idx
                refined_dict[idx] = added_lines
            else:
                under_sampled_map_uncertain[under_sampled_map_updated] = idx

    # it is unsafe to downsample SVG under manual mode, cause now the USM is probably not complete
    if downsample:
        start = time.time()
        lines = downsample_all(edge_map_x, edge_map_y, pt_map, regions != 0, 
            valence_map, pt_type, lines_map_x, lines_map_y, ratio = down_rate)
        end = time.time()
        print("log:\tdual contouring downsampling finished in %.2f seconds"%(end-start))
    else:
        lines = linemap_to_lines(lines_map_x, lines_map_y)    

    if len(refined_lines) > 0:
        assert len(refined_lines) % 2 == 0
        for i in range(len(refined_lines)-1, -1, -2):
            if len(refined_lines[i].shape) < 1 or len(refined_lines[i - 1].shape) < 1:
                refined_lines.pop(i)
                refined_lines.pop(i-1)
        refined_lines = np.stack(refined_lines, axis = 0)
        if len(refined_lines) > 0:
            lines = np.concatenate((lines, refined_lines), axis = 0)
    return lines, refined_dict, under_sampled_map_applied, under_sampled_map_uncertain

def downsample_ndc(edge_map, pt_map, keypts, usm, lines_map_x, lines_map_y, down_rate = 4):
    edge_map_x = edge_map[..., 0]
    edge_map_y = edge_map[..., 1]

    valence_map, pt_mask = get_valence_np(edge_map_x, edge_map_y)
    pt_type = np.zeros(valence_map.shape)
    idx_to_keypts = {}
    
    align_keypt(pt_map, keypts['end_point'], valence_map, 1, pt_type, idx_to_keypts, 3.5)
    align_keypt(pt_map, keypts['sharp_turn'], valence_map, 2, pt_type, idx_to_keypts, 3.5)
    align_keypt(pt_map, keypts['junc'], valence_map, 2, pt_type, idx_to_keypts)
    align_keypt(pt_map, keypts['junc'], valence_map, 3, pt_type, idx_to_keypts, 3.5)

    lines = downsample_all(edge_map_x, edge_map_y, pt_map, usm, 
            valence_map, pt_type, lines_map_x, lines_map_y, ratio = down_rate)

    return lines

def vis_edge_map(edge_map, png = 'edge.png'):
    h, w = edge_map.shape[0], edge_map.shape[1]
    res = np.zeros((4*h, 4*w)).astype(bool)
    for i in range(h):
        for j in range(w):
            # draw grid edge along x-axis
            if edge_map[i][j][0]:
                res[4*i, 4*j : 4*(j + 1)] = True
            if edge_map[i][j][1]:
                res[4*i : 4*(i + 1), 4*j] = True
    Image.fromarray(res).save(png)

def vis_grad(grad, save_path = 'grad.svg'):
    '''
    Given,
        grad, array with H x W x 2
    Action,
        visualize gradient direction as svg image
    '''
    h, w = grad.shape[0], grad.shape[1]
    P = []
    PT = []
    for i in range(h):
        for j in range(w):
            start = complex(j, i)
            direction = complex(*grad[i][j])
            end = start + direction
            P.append(Line(start = start, end = end))
            PT.append(start)
    wsvg(P, colors = ['cyan'] * len(P), stroke_widths = [0.2] * len(P), 
        nodes = PT, node_colors = ['purple'] * len(PT), node_radii = [0.2] * len(PT),
        dimensions = (w, h), filename = save_path)

def length_too_long(start, end, thres = 20):
    line = start - end.mean(axis = 0)
    line_length = np.sqrt((line * line).sum())
    if line_length > 20:
        return True
    else:
        return False

def pt_num(pt):
    if len(pt.shape) == 1:
        return 0 if pt.shape[0] == 0 else 1
    if len(pt.shape) == 2: 
        return len(pt)

def connect_usm_pts(connection_pts, pt, added_lines):
    assert pt_num(pt) == 1
    for i in range(len(connection_pts)):
        added_lines.append(connection_pts[i][0])
        added_lines.append(pt)

def update_line_maps(lines_map_x, lines_map_y, pt_map, pt_mask, pt_type, connection_pts, 
        region, refined_lines, idx_to_keypts, 
        usm_updated = None, 
        manual = False, 
        keypts = None,
        full_auto_mode = False):
    # this is a inplace function!
    # rebuild the topology inside this region
    # get the region valence
    valence = len(connection_pts)
    added_lines = []
    if manual and full_auto_mode == False:
        assert keypts is not None
        end, sharp, junc, avg_pt = query_keypt_by_region_no_alginment(keypts, region.astype(bool), pt_map, pt_mask)
    else:
        end, sharp, junc, avg_pt = query_keypt_by_region(pt_type, region.astype(bool), idx_to_keypts, pt_map, pt_mask)
    manual = manual or full_auto_mode
    need_update = False
    if valence == 0: 
        return None, None
    # end point
    elif valence == 1:
        if pt_num(end) == 1 and (pt_num(junc) == 0 or manual):
            connect_usm_pts(connection_pts, end, added_lines)
            need_update = True
    elif valence == 2:
        need_update = True
        if pt_num(sharp) == 1 and (pt_num(end) == 0 or manual):
            connect_usm_pts(connection_pts, sharp, added_lines)
        elif pt_num(junc) == 1 and (pt_num(end) == 0 or manual):
            connect_usm_pts(connection_pts, junc, added_lines)
        elif pt_num(end) >= 1 and pt_num(sharp) >= 1 and pt_num(junc) >=1:
            s = sharp if pt_num(sharp) == 1 else sharp[0]
            connect_usm_pts(connection_pts, s, added_lines)
        elif pt_num(sharp) > 1:
            need_update = False
        elif pt_num(end) > 0 and manual == False:
            need_update = False
        elif manual:
            added_lines.append(connection_pts[0][0])
            added_lines.append(connection_pts[1][0])
        else:
            need_update = False

    # T, Y junction
    elif valence == 3:
        need_update = True
        if pt_num(junc) == 1 and (pt_num(end) == 0 or manual):
            connect_usm_pts(connection_pts, junc, added_lines)
        elif pt_num(sharp) == 1 and (pt_num(end) == 0 or manual):
            connect_usm_pts(connection_pts, sharp, added_lines)
        elif pt_num(end) >= 1 and pt_num(sharp) >= 1 and pt_num(junc) >=1:
            j = junc if pt_num(junc) == 1 else junc[0]
            connect_usm_pts(connection_pts, j, added_lines)
        elif pt_num(junc) > 1:
            need_update = False
        elif len(end) > 0 and manual == False:
            need_update = False
        elif manual:
            # for i in range(len(connection_pts)):
            #     added_lines.append(connection_pts[i][0])
            #     added_lines.append(avg_pt)
            dir_matrix = get_direction_matrix(connection_pts)
            assert (3, 3) == dir_matrix.shape
            pt_idx01 = np.unravel_index(np.argmax(dir_matrix), dir_matrix.shape)
            assert len(pt_idx01) == 2
            pt_idx2 = [i for i in [0, 1, 2] if i not in pt_idx01]
            pt0 = connection_pts[pt_idx01[0]][0]
            pt1 = connection_pts[pt_idx01[1]][0]
            pt2 = connection_pts[pt_idx2[0]][0]
            pt3 = (pt0 + pt1) / 2
            added_lines.append(pt0)
            added_lines.append(pt1)
            added_lines.append(pt2)
            added_lines.append(pt3)
        else:
            need_update = False

    # X junction
    elif valence == 4:
        need_update = True
        if pt_num(junc) == 1 and (len(end) == 0 or manual):
            connect_usm_pts(connection_pts, junc, added_lines)
        elif pt_num(sharp) == 1 and (pt_num(end) == 0 or manual):
            connect_usm_pts(connection_pts, sharp, added_lines)
        elif pt_num(end) >= 1 and pt_num(sharp) >= 1 and pt_num(junc) >=1:
            j = junc if pt_num(junc) == 1 else junc[0]
            connect_usm_pts(connection_pts, j, added_lines)
        elif (pt_num(junc) > 1 or len(end) > 0) and manual == False:
            need_update = False
        elif manual:
            try:
                tans_edge = []
                # let's emumerate all possible connections here
                # (0, 1), (2, 3)
                line0 = Line(complex(*connection_pts[0][0]), complex(*connection_pts[1][0]))
                line1 = Line(complex(*connection_pts[2][0]), complex(*connection_pts[3][0]))
                tans_edge.append(line0.unit_tangent(0))
                tans_edge.append(line1.unit_tangent(0))
                # (1, 2), (0, 3)
                line0 = Line(complex(*connection_pts[1][0]), complex(*connection_pts[2][0]))
                line1 = Line(complex(*connection_pts[0][0]), complex(*connection_pts[3][0]))
                tans_edge.append(line0.unit_tangent(0))
                tans_edge.append(line1.unit_tangent(0))
                # (0, 2), (1, 3)
                line0 = Line(complex(*connection_pts[0][0]), complex(*connection_pts[2][0]))
                line1 = Line(complex(*connection_pts[1][0]), complex(*connection_pts[3][0]))
                tans_edge.append(line0.unit_tangent(0))
                tans_edge.append(line1.unit_tangent(0))
                tans_stroke = []
                for i in range(4):
                    tans_stroke.append(connection_pts[i][1])
                dir_matrix = np.zeros((6, 4))
                for i in range(6):
                    for j in range(4):
                        dir_matrix[i][j] = complex_dot(tans_edge[i], tans_stroke[j])
                dir_matrix = dir_matrix.mean(axis = 1).squeeze()
                parallel_idx = np.argsort(dir_matrix)
                group_list = [((0, 1), (2, 3)), ((1, 2), (0, 3)), ((0, 2), (1, 3))]

                if parallel_idx[-1] == 0 and dir_matrix[1] > 0.9 or parallel_idx[-1] == 1 and dir_matrix[0] > 0.9:
                    group_a = group_list[0][0]
                    group_b = group_list[0][1]

                elif parallel_idx[-1] == 2 and dir_matrix[3] > 0.9 or parallel_idx[-1] == 3 and dir_matrix[2] > 0.9:
                    group_a = group_list[1][0]
                    group_b = group_list[1][1]

                elif parallel_idx[-1] == 4 and dir_matrix[5] > 0.9 or parallel_idx[-1] == 5 and dir_matrix[4] > 0.9:
                    group_a = group_list[2][0]
                    group_b = group_list[2][1]
                update_x_junction(connection_pts, group_a, group_b, added_lines)    
            except:
                for i in range(len(connection_pts)):
                    added_lines.append(connection_pts[i][0])
                    added_lines.append(avg_pt)
        else:
            need_update = False

    # valence > 4:
    else:
        need_update = True
        if pt_num(junc) == 1 and (len(end) == 0 or manual):
            connect_usm_pts(connection_pts, junc, added_lines)
        elif pt_num(end) >= 1 and pt_num(sharp) >= 1 and pt_num(junc) >=1:
            j = junc if pt_num(junc) == 1 else junc[0]
            connect_usm_pts(connection_pts, j, added_lines)
        elif (pt_num(junc) > 1 or len(end) > 0) and manual == False:
            need_update = False
        elif manual:
            for i in range(len(connection_pts)):
                added_lines.append(connection_pts[i][0])
                added_lines.append(avg_pt)
        else:
            need_update = False
            
    # reset all edge flags in the under sampled region
    if need_update:
        cpt_coord = np.array(np.where(region.astype(bool))).T
        h, w = lines_map_x.shape[0], lines_map_x.shape[1]
        for idx in range(len(cpt_coord)):
            i, j = cpt_coord[idx]
            lines_map_x[i][j] = False
            lines_map_y[i][j] = False
        for k in range(len(connection_pts)):
            i, j, edge = connection_pts[k][2]
            if edge == 2: # bottom
                lines_map_x[i][j] = False
            if edge == 3: # right
                lines_map_y[i][j] = False
            if usm_updated is not None:
                usm_updated[i][j] = True
        assert len(added_lines) > 0
        assert len(added_lines) % 2 == 0
        for i in range(len(added_lines)):
            refined_lines.append(added_lines[i])
    
    return need_update, added_lines

def update_x_junction(connection_pts, group_a, group_b, refined_lines):
    idx0, idx1 = group_a
    idx2, idx3 = group_b
    pt0 = connection_pts[idx0][0]
    pt1 = connection_pts[idx1][0]
    pt2 = connection_pts[idx2][0]
    pt3 = connection_pts[idx3][0]
    done = False
    if (pt0 != pt1).any() and (pt2 != pt3).any(): 
        line1 = Line(complex(*pt0), complex(*pt1))
        line2 = Line(complex(*pt2), complex(*pt3))
        if len(line1.intersect(line2)) == 0:
            refined_lines.append(pt0)
            refined_lines.append(pt1)
            refined_lines.append(pt2)
            refined_lines.append(pt3)
            done = True
    if (pt0 != pt3).any() and (pt2 != pt1).any() and not done: 
        line1 = Line(complex(*pt0), complex(*pt3))
        line2 = Line(complex(*pt2), complex(*pt1))
        if (len(line1.intersect(line2)) == 0):
            refined_lines.append(pt0)
            refined_lines.append(pt3)
            refined_lines.append(pt2)
            refined_lines.append(pt1)

def get_direction_matrix(connection_pts):
    tans = []
    size = len(connection_pts)
    for i in range(size):
        tans.append(connection_pts[i][1])
    dir_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            dir_matrix[i][j] = complex_dot(tans[i], tans[j])
    return dir_matrix

def get_dist_matrix(connection_pts):
    pts = []
    size = len(connection_pts)
    for i in range(size):
        pts.append(connection_pts[i][0])
    dist_matrix = np.zeros((size, size))
    for i in range(size):
        for j in range(i + 1, size):
            dist_matrix[i][j] = np.linalg.norm(pts[i] - pts[j])
    return dist_matrix

def complex_dot(a, b):
    a_np = np.array((a.real, a.imag))
    b_np = np.array((b.real, b.imag))
    return abs(np.dot(a_np, b_np))

def find_connection_points(pt_grid_map, pt_map, region, under_sampled_map):
    # let's just find the connection points only in this function
    # get region boundary
    # the coordinate in cnt is xy, not ij!
    assert pt_grid_map.shape[0] == pt_map.shape[0] and pt_grid_map.shape[1] == pt_map.shape[1]
    h, w = pt_grid_map.shape[0], pt_grid_map.shape[1]
    # get the coords of region boundary
    cnt, _ = cv2.findContours(region.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    if len(cnt) != 1: return False
    cnt = cnt[0].squeeze(axis = 1)
    connection_pts = []
    for idx in range(len(cnt)):
        j, i = cnt[idx]
        if i == 0 or j == 0: continue
        if i > h -5 or j > w - 5: continue # skip if it is too close to right and bottom
        if pt_grid_map[i][j] == False: continue

        # find if there exists poly lines coming from (top, left, bottom, right)
        ef = get_edge_flag_per_grid(pt_grid_map, i, j) # ef is edge flag that indicate if poly line exists
        # find if the (top, left, bottom, right) of current location also has under sampling flag
        # this could be useful to tell if we are on a boundary
        uf = get_under_sampled_flag_per_grid(under_sampled_map, i, j) # usm flag
        # find the direction that has the edge coming and is outside the USM
        cpf = np.logical_and(ef, np.logical_not(uf))# connection points flag
        # connection_pts = [start point, line tangent at break point, coord, where to connect]
        '''
        where to connect:
            0: top
            1: left
            2: bottom
            3: right
        '''
        if cpf.any():
            con_pt_idx = np.where(cpf)[0]
            end_pt = pt_map[i][j] # why we need this?
            if 0 in con_pt_idx:
                start_pt0 = pt_map[i - 1][j]
                connection_pts.append((start_pt0, Line(complex(*start_pt0), complex(*end_pt)).unit_tangent(0), (i-1, j, 0)))
            if 1 in con_pt_idx:
                start_pt1 = pt_map[i][j - 1]
                connection_pts.append((start_pt1, Line(complex(*start_pt1), complex(*end_pt)).unit_tangent(0), (i, j-1, 1)))
            if 2 in con_pt_idx:
                start_pt2 = pt_map[i + 1][j]
                connection_pts.append((start_pt2, Line(complex(*start_pt2), complex(*end_pt)).unit_tangent(0), (i+1, j, 2)))
            if 3 in con_pt_idx:
                start_pt3 = pt_map[i][j + 1]
                connection_pts.append((start_pt3, Line(complex(*start_pt3), complex(*end_pt)).unit_tangent(0), (i, j+1, 3)))

    return connection_pts

def find_end_point(edge_flags, i, j, pt_map, skip_edge):
    skip_edge_map = [2, 3, 0, 1]
    skip_edge = skip_edge_map[skip_edge]
    ef = np.where(edge_flags)[0]
    if 0 in ef and 0 != skip_edge:
        return pt_map[i - 1][j]
    if 1 in ef and 1 != skip_edge:
        return pt_map[i][j - 1]
    if 2 in ef and 2 != skip_edge:
        return pt_map[i + 1][j]
    if 3 in ef and 3 != skip_edge: 
        return pt_map[i][j + 1]

def get_edge_flag_per_grid(edge_map, i, j):
    edge0 = edge_map[i-1][j] # top
    edge1 = edge_map[i][j-1] # left
    edge2 = edge_map[i+1][j] # bottom
    edge3 = edge_map[i][j+1] # right
    return np.array([edge0, edge1, edge2, edge3])

def get_under_sampled_flag_per_grid(under_sampled_map, i, j):
    un0 = under_sampled_map[i-1][j] # unsampled neighbour
    un1 = under_sampled_map[i][j-1]
    un2 = under_sampled_map[i+1][j]
    un3 = under_sampled_map[i][j+1]
    return np.array([un0, un1, un2, un3])

def choose_pt(pt_map, idx_to_keypts, idx, thr):
    # get the keypoint and DC grid center point
    dc_pt = pt_map[tuple(idx)]
    
    '''
    if the two point are close to each other enough
    end keypoint probably will be a better choose
    but if there are too faraway, then dual contouring 
    grid center may be better.
    ''' 
    # key_pt = idx_to_keypts[tuple(idx)]
    # dist = np.sqrt(np.square(dc_pt - key_pt).sum())
    # if dist > thr:
    #     return dc_pt
    # else:
    #     return key_pt
    
    return dc_pt

def find_match_keypts(pt_type, idx_to_keypts, val, pt_map, thr = 0.2):
    res = []
    idxs = np.array(np.where(pt_type == val)).T
    if len(idxs) > 1:
        for idx in idxs:
            res.append(choose_pt(pt_map, idx_to_keypts, idx, thr))
        res = np.stack(res, axis = 0)
        # res = res.mean(axis = 0)
    else:
        res = choose_pt(pt_map, idx_to_keypts, idxs[0], thr)
    return np.array(res)

def get_avg_pt(pt_map, pt_region):
    if pt_region.sum() > 0:
        pts = pt_map[np.where(pt_region)].reshape(-1, 2)
        return pts.mean(axis = 0)
    else:
        return []

def query_coord_by_region(region, keypts, key):
    keypt_map = np.zeros(region.shape).astype(bool)
    keypt_coord = keypts[key]
    h, w  = region.shape
    # set overflow coords
    keypt_coord[np.where(keypt_coord[..., 0] >= h)[0], 0] = h - 1
    keypt_coord[np.where(keypt_coord[..., 1] >= w)[0], 1] = w - 1
    keypt_map[tuple(keypt_coord.T)] = True
    keypt = (np.array(np.where(keypt_map & region)).T / 2) 
    keypt[..., (0, 1)] = keypt[..., (1, 0)]
    return keypt.squeeze()

def query_keypt_by_region_no_alginment(keypts, region, pt_map, pt_mask):
    ## re-scale the keypoint coordination and converte them into int coordinations are in (x, y) format
    keypts = copy.deepcopy(keypts) # we don't want to change the input
    for key in keypts:
        keypts[key] = ((keypts[key] + 0.5) * 2).astype(int)
    ## get average points of current region
    region_pt = np.logical_and(region, pt_mask)
    avg_pt = get_avg_pt(pt_map, region_pt)
    
    ## query keypoints covered by region
    end = query_coord_by_region(region, keypts, 'end_point')
    sharp = query_coord_by_region(region, keypts, 'sharp_turn')
    junc = query_coord_by_region(region, keypts, 'junc')
    return np.array(end), np.array(sharp), np.array(junc), avg_pt

def query_keypt_by_region(pt_type, region, idx_to_keypts, pt_map, pt_mask):
    end = []
    sharp = []
    junc = []
    avg_pt = []
    assert region.shape == pt_type.shape
    pt_type = pt_type.copy()
    # make sure region is boolean matrix
    region = region.astype(bool)
    region_pt = np.logical_and(region, pt_mask)
    # set all type info that out of current region to 0 
    pt_type[~region] = 0
    # get query point type
    types = np.unique(pt_type[region])
    # end point
    if 1 in types:
        end = find_match_keypts(pt_type, idx_to_keypts, 1, pt_map)    
    # sharp turn
    if 2 in types:
        sharp = find_match_keypts(pt_type, idx_to_keypts, 2, pt_map)
    # junctions
    if 3 in types:
        junc = find_match_keypts(pt_type, idx_to_keypts, 3, pt_map)
    avg_pt = get_avg_pt(pt_map, region_pt)
    return np.array(end), np.array(sharp), np.array(junc), avg_pt

def lines_to_svg(lines, canvas_w, canvas_h, svg_path = None, indexing = 'xy', paths2Drawing = False):
    # coordinates in lines are y (height), x (width)
    paths_ = Path()
    # assume lines is numpy array
    pt_nums, dims = lines.shape
    # skip if the lines array is not correct
    if dims != 2: return False
    for i in range(0, int(len(lines)//2*2), 2):
        # but svg requires coordinate like x (width), y (height)!
        if indexing == 'ij':
            start = complex(*(lines[i][1], lines[i][0]))
            end = complex(*(lines[i+1][1], lines[i+1][0]))
        elif indexing == "xy":
            start = complex(*(lines[i][0], lines[i][1]))
            end = complex(*(lines[i+1][0], lines[i+1][1]))
        else:
            raise ValueError("Unspported indexing method %s"%indexing)
        if start != end:
            paths_.append(Line(start, end))
    if len(paths_) > 0:
        stroke_widths = [0.5]*len(paths_)
        dimensions = (canvas_w, canvas_h),
        attributes = [{"fill":'none', "stroke":"#000000", "stroke-width":'1', "stroke-linecap":"round"}] * len(paths_)
        if paths2Drawing:
            return wsvg(
                paths_, 
                stroke_widths = stroke_widths, 
                dimensions = (canvas_w, canvas_h), 
                filename = svg_path, 
                attributes = attributes,
                paths2Drawing = True).tostring()
        else:
            assert svg_path is not None
            wsvg(
                paths_, 
                stroke_widths = stroke_widths, 
                dimensions = (canvas_w, canvas_h),
                attributes = attributes, 
                filename = svg_path)
    
    if path.exists(svg_path):    
        return True
    else:
        return False

def roll_edge(edge, shift, axis):
    assert len(edge.shape) == 2
    assert shift != 0
    edge_ = edge.copy().astype(bool)
    if axis == 'x':
        if shift > 0:
            edge_[-shift:, :] = False
        else:
            edge_[:-shift, :] = False
        return np.roll(edge_, shift = shift, axis = 0)
    elif axis == 'y':
        if shift > 0:
            edge_[:, -shift:] = False
        else:
            edge_[:, :-shift] = False
        return np.roll(edge_, shift = shift, axis = 1)
    else:
        raise ValueError("axis could only be x or y but got %s"%axis)

def logical_minus(a, b):
    assert a.shape == b.shape
    assert a.dtype == bool
    assert b.dtype == bool
    return a^b&a

def map_to_lines(edge_map, keypt_map, tensor_input = False, refine = False):
    '''
    import pickle
    with open("map_to_lines.pickle","wb") as f: pickle.dump( ( edge_map, keypt_map, tensor_input, refine ),  f )
    with open('map_to_lines_test.py','w') as f: print( """from utils.ndc_tools import map_to_lines
import pickle
args = pickle.load( open('map_to_lines.pickle','rb') )
import time
start = time.time()
map_to_lines( *args )
print( time.time() - start, "seconds" )
""", file = f )
    print( "map_to_lines_test.py" )
    '''
    
    '''
    Given:
        edge_map:   A boolean array with shape (h, w, 2)
        keypt_map:  A float array with shape (h, w, 2)
    Return:
        lines:          A float array with shape (k, 2), stores the start and end point of each
                        line reconstruction in a compact way
        lines_map_x:    A float array with shape (h, w , 2), stores the lines intersect with each
                        grid edge along x-axis in a sparse way
        lines_map_y:    A float array with shape (h, w , 2), stores the lines intersect with each
                        grid edge along y-axis in a sparse way
    '''
    if tensor_input:
        edge_map = edge_map.squeeze().detach().to(bool).cpu().numpy()
        keypt_map = keypt_map.squeeze().detach().cpu().numpy()
    h, w, _= edge_map.shape
    # remove tiny lines in the edge map, DON'T use this flag when training the network!
    if refine:
        edge_map = edge_map.astype(bool)
        # cut small branch
        edge_map = cut_branch(edge_map)
        skel = remove_stray(edge_map[..., 0]|edge_map[..., 1])
        '''
            below function are still experimental, and may still buggy
        '''
        # remove repeat (adjacent) strokes
        skel = edge_map[..., 0] | edge_map[..., 1]
        skel = remove_parallel(skel, edge_map)
        edge_map = fill_edge_hole(edge_map & skel[..., np.newaxis])
        edge_map = cut_branch(edge_map & skel[..., np.newaxis])
        

    assert edge_map.shape[0] == keypt_map.shape[0] and edge_map.shape[1] == keypt_map.shape[1]
    
    '''
    lines_map_x = np.zeros((h, w)).astype(object)
    lines_map_y = np.zeros((h, w)).astype(object)
    # it is not possible to record dual contouring lines along y-axis at the 1st row
    # and also not possbile to record dc lines along x-axis at the 1st column
    for i in range(h):
        for j in range(w):
            if edge_map[i][j][0] and i != 0: # edge map x
                if (keypt_map[i - 1][j] == keypt_map[i][j]).all() == False:
                    s = keypt_map[i - 1][j]
                    e = keypt_map[i][j]
                    # if (s > 1e-2).all() or (e > 1e-2).all():
                    #     lines_map_x[i][j] = (s, e)
                    lines_map_x[i][j] = (s, e)
            if edge_map[i][j][1] and j != 0: # edge map y
                if (keypt_map[i][j - 1] == keypt_map[i][j]).all() == False:
                    s = keypt_map[i][j - 1]
                    e = keypt_map[i][j]
                    # if (s > 1e-2).all() or (e > 1e-2).all():
                    #     lines_map_y[i][j] = (s, e)
                    lines_map_y[i][j] = (s, e)
    
    return linemap_to_lines(lines_map_x, lines_map_y), lines_map_x, lines_map_y, edge_map
    '''
    
    # Get all the True i,j locations.
    # We don't want the first row/column, so skip them in the `where` and offset the resulting row/column indices.
    wherex_left = np.where( edge_map[1:,:,0] )
    wherex_right = ( wherex_left[0] + 1, wherex_left[1] )
    wherey_up = np.where( edge_map[:,1:,1] )
    wherey_down = ( wherey_up[0], wherey_up[1] + 1 )
    
    # Allocate space for the line-segments x 2 (start and end) x 2 (x and y) data
    line_segments = np.zeros( ( len(wherex_left[0]) + len(wherey_up[0]), 2, 2 ), dtype = keypt_map.dtype )
    # Fill it.
    line_segments[ :len(wherex_left[0]), 0, : ] = keypt_map[ wherex_left ]
    line_segments[ :len(wherex_left[0]), 1, : ] = keypt_map[ wherex_right ]
    line_segments[ len(wherex_left[0]):, 0, : ] = keypt_map[ wherey_up ]
    line_segments[ len(wherex_left[0]):, 1, : ] = keypt_map[ wherey_down ]
    # Extract zero-length segments
    line_segments = line_segments[ ( line_segments[:,0,:] != line_segments[:,1,:] ).any( axis = 1 ) ]
    lines = line_segments.reshape( -1, 2 )
    # assert ( lines == linemap_to_lines(lines_map_x, lines_map_y) ).all()
    
    # Make 2D object array from the data.
    # This would be better as a np.ma.MaskedArray, but there are a lot of places in the code to change.
    lines_map_x2 = np.zeros((h, w)).astype(object)
    lines_map_y2 = np.zeros((h, w)).astype(object)
    xmask = ( keypt_map[ wherex_left ] != keypt_map[ wherex_right ] ).any( axis = 1 )
    lines_map_x2[ wherex_right[0][xmask], wherex_right[1][xmask] ] = list(zip( keypt_map[ wherex_left ][xmask], keypt_map[ wherex_right ][xmask] ))
    ymask = ( keypt_map[ wherey_up ] != keypt_map[ wherey_down ] ).any( axis = 1 )
    lines_map_y2[ wherey_down[0][ymask], wherey_down[1][ymask] ] = list(zip( keypt_map[ wherey_up ][ymask], keypt_map[ wherey_down ][ymask] ))
    # assert ( np.where(lines_map_x)[0] == np.where(lines_map_x2)[0] ).all()
    # assert ( np.where(lines_map_x)[1] == np.where(lines_map_x2)[1] ).all()
    # assert ( np.where(lines_map_y)[0] == np.where(lines_map_y2)[0] ).all()
    # assert ( np.where(lines_map_y)[1] == np.where(lines_map_y2)[1] ).all()
    # assert ( lines_map_y[ np.where(lines_map_y) ][0] == lines_map_y2[ np.where(lines_map_y2) ][0] ).all()
    # assert ( np.array( lines_map_x[ np.where(lines_map_x) ][0] ) == np.array( lines_map_x2[ np.where(lines_map_x) ][0] ) ).all()
    # assert ( np.array( lines_map_x[ np.where(lines_map_x) ][1] ) == np.array( lines_map_x2[ np.where(lines_map_x) ][1] ) ).all()
    # assert ( np.array( lines_map_y[ np.where(lines_map_y) ][0] ) == np.array( lines_map_y2[ np.where(lines_map_y) ][0] ) ).all()
    # assert ( np.array( lines_map_y[ np.where(lines_map_y) ][1] ) == np.array( lines_map_y2[ np.where(lines_map_y) ][1] ) ).all()
    lines_map_x = lines_map_x2
    lines_map_y = lines_map_y2
    
    return lines, lines_map_x, lines_map_y, edge_map

def neighbormap_from_edge_map( edge_map, MAX_VALENCE = 10 ):
    '''
    Given:
        edge_map: A boolean array with shape (h, w, 2) of the kind used in `map_to_lines()`
        MAX_VALENCE: An optional parameter of the maximum valence of any vertex.
    Returns:
        A tuple of data for mapping from a 2D index to a sequence of other indices.
        Use this tuple as the `neighbormap` parameter to `neighbors_from_neighbormap_ij()`
        or `all_indices_from_neighbormap()`.
    '''
    # A 2D array to 2D coordinate map
    valence = np.zeros( ( edge_map.shape[0], edge_map.shape[0] ), dtype = int )
    ij2neighbor = np.zeros( ( edge_map.shape[0], edge_map.shape[0], MAX_VALENCE, 2 ), dtype = int )
    ij2neighbor -= 1 # initialize to -1
    for i,j in np.where( edge_map[1:,:,0] ):
        ij2neighbor[ i, j, valence[i,j], : ] = ( i+1, j )
        valence[i,j] += 1
        ij2neighbor[ i+1, j, valence[i+1,j], : ] = ( i, j )
        valence[i+1,j] += 1
    for i,j in np.where( edge_map[:,1:,1] ):
        ij2neighbor[ i, j, valence[i,j], : ] = ( i, j+1 )
        valence[i,j] += 1
        ij2neighbor[ i, j+1, valence[i,j+1], : ] = ( i, j )
        valence[i,j+1] += 1
    
    return ij2neighbor, valence

def neighbors_from_neighbormap_ij( neighbormap, i, j ):
    '''
    Given:
        neighbormap: The return value from `neighbormap_from_edge_map()` (a tuple of two things)
        i,j: The 2D index of a point
    Returns:
        A sequence of 2D indices of neighbors (length valence)
    '''
    ij2neighbor, valence = neighbormap
    return neighbormap[ i,j, :valence[i,j] ]

def all_indices_from_neighbormap( neighbormap ):
    '''
    Given:
        neighbormap: The return value from `neighbormap_from_edge_map()` (a tuple of two things)
    Return:
        A sequence of all 2D indices of vertices.
    '''
    return np.array( np.where( valence ) ).T

def remove_stray(skel, thr = 6):
    _, edge_regions = cv2.connectedComponents(skel.astype(np.uint8), connectivity = 8)
    mask = np.zeros(skel.shape).astype(bool)
    for r in np.unique(edge_regions):
        if r == 0: continue
        m = edge_regions == r
        if m.sum() < thr: mask[m] = True
    skel = skel & ~mask
    return skel

def remove_parallel(skel, edge_map, stray_thr = 6):
    skel = skeletonize(skel)
    # remove stray short lines 
    skel = remove_stray(skel, stray_thr)
    # fill 8-way connection of -45 degree diagonal to 4 way conncection
    skel_4way = logical_minus(roll_edge(skel, -1, 'x'), skel) & roll_edge(skel, 1, 'y')
    skel = skel | skel_4way
    skel = np.repeat(skel[..., np.newaxis], 2, axis = -1)
    # the rest edge labels could still exist 8-way connections, need to check this again
    skel_ = edge_map & skel
    skel_ = skel_[..., 0] | skel_[..., 1]
    skel_4way = logical_minus(roll_edge(skel_, -1, 'x'), skel_)
    skel_4way = roll_edge(roll_edge(skel_4way, -1, 'y'), 1, 'x') & roll_edge(skel_, 1, 'x')
    skel = skel_ | skel_4way
    return skel

def cut_branch(edge_map):
    edge_x = edge_map[..., 0]
    edge_y = edge_map[..., 1]
    cross_k = np.array([[0,1,0],[1,1,1],[0,1,0]]).astype(np.uint8)
    val_map, _ = get_valence_np(edge_x, edge_y)
    end_map = val_map == 1
    junc_map = val_map > 2
    branch_map = convolve2d(end_map.astype(float), cross_k, mode = 'same', boundary = 'fill', fillvalue = 0) > 0
    branch_map = branch_map & junc_map
    branch_map = cv2.dilate(branch_map.astype(np.uint8), cross_k, iterations = 1 ) > 0
    branch_map = branch_map & end_map
    edge_x_ = logical_minus(edge_x, branch_map)
    edge_y_ = logical_minus(edge_y, branch_map)
    branch_map_ = logical_minus(branch_map, edge_x | edge_y)
    edge_x = logical_minus(edge_x_, roll_edge(branch_map_, 1, 'x'))
    edge_y = logical_minus(edge_y_, roll_edge(branch_map_, 1, 'y'))
    return np.stack((edge_x, edge_y), axis = -1)

def fill_edge_hole(edge_map):
    edge_x = edge_map[..., 0]
    edge_y = edge_map[..., 1]
    # fill holes in matched lines
    kernel_x_hole = np.array([[-1, 1, -1], [-1, -1, -1], [-1, 1, -1]])
    kernel_y_hole = np.array([[-1, -1, -1], [1, -1, 1], [-1, -1, -1]])
    hole_x = convolve2d(edge_x, kernel_x_hole, mode = 'same', boundary = 'fill', fillvalue = 0) == 2
    hole_y = convolve2d(edge_y, kernel_y_hole, mode = 'same', boundary = 'fill', fillvalue = 0) == 2
    edge_x = edge_x | hole_x
    edge_y = edge_y | hole_y
    '''
    fill holes in mismatched lines:
    case1:
    -------
           -------
    -------
         -------
    case2:
    |
    |
      |
      |

    |
    | |
    | |
      |
   
    '''

    kernel_xy1_hole = np.array([[1, -1, 0], [1, -1, 0], [-1, 1, 0]])
    kernel_xy2_hole = np.array([[-1, 1, 0], [-1, 1, 0], [1, -1, 0]])
    kernel_yx1_hole = np.array([[1, 1, -1], [-1, -1, 1], [0, 0, 0]])
    kernel_yx2_hole = np.array([[-1, -1, 1], [1, 1, -1], [0, 0, 0]])
    kernel_yx1_overlap = np.array([[1, 1, -1], [-1, 1, 1], [0, 0, 0]])
    kernel_yx2_overlap = np.array([[-1, -1, 1], [1, 1, 1], [0, 0, 0]])
    kernel_xy1_overlap = np.array([[1, -1, 0], [1, 1, 0], [-1, 1, 0]])
    kernel_xy2_overlap = np.array([[-1, 1, 0], [-1, 1, 0], [1, 1, 0]])
    hole_y1 = correlate(edge_x.astype(int), kernel_xy1_hole, mode = 'constant') == 3
    hole_y2 = correlate(edge_x.astype(int), kernel_xy2_hole, mode = 'constant') == 3
    hole_y3 = correlate(edge_x.astype(int), kernel_xy1_overlap, mode = 'constant') == 4
    hole_y4 = correlate(edge_x.astype(int), kernel_xy2_overlap, mode = 'constant') == 4
    hole_x1 = correlate(edge_y.astype(int), kernel_yx1_hole, mode = 'constant') == 3
    hole_x2 = correlate(edge_y.astype(int), kernel_yx2_hole, mode = 'constant') == 3
    hole_x3 = correlate(edge_y.astype(int), kernel_yx1_overlap, mode = 'constant') == 4
    hole_x4 = correlate(edge_y.astype(int), kernel_yx2_overlap, mode = 'constant') == 4
    edge_x = logical_minus( edge_x | hole_x1 | hole_x2 | hole_x3 | hole_x4, roll_edge(hole_y4, 1, 'x') | hole_y3)
    edge_y = logical_minus( edge_y | hole_y1 | hole_y2 | hole_y3 | hole_y4, roll_edge(hole_x4, 1, 'y') | hole_x3)
    # remove connection between adjacent 2 t-junctions
    val_map, _ = get_valence_np(edge_map[..., 0], edge_map[..., 1])
    kernel_tt_x = np.array([[0, 1, 0], [0, 1, 0], [0, 0, 0]])
    kernel_tt_y = np.array([[0, 0, 0], [1, 1, 0], [0, 0, 0]])
    branch_x = correlate((val_map >= 3).astype(int), kernel_tt_x, mode = 'constant') == 2
    branch_y = correlate((val_map >= 3).astype(int), kernel_tt_y, mode = 'constant') == 2
    edge_x = logical_minus(edge_x, branch_x)
    edge_y = logical_minus(edge_y, branch_y)
    return np.stack((edge_x, edge_y), axis = -1)

def linemap_to_lines(lines_map_x, lines_map_y):
    lines_x = lines_map_x[np.where(lines_map_x != False)]
    lines = []
    if len(lines_x) > 0:
        lines_x = np.concatenate(lines_x)
        lines.append(lines_x)
    lines_y = lines_map_y[np.where(lines_map_y != False)]
    if len(lines_y) > 0:
        lines_y = np.concatenate(lines_y)
        lines.append(lines_y)
    if len(lines) == 2:
        lines = np.concatenate((lines_x, lines_y), axis = 0)
    elif len(lines) ==1:
        lines = lines[0]
    else:
        raise ValueError("irregular line maps!")
    return lines.reshape(-1, 2)

def query_keypt_by_bbox(keypt_svg, bbox, re_scale):
    A, C = bbox
    left, top = A
    right, bottom = C
    keypts = []
    if keypt_svg is not None:
        end_points = filter_pts(keypt_svg["end_point"] * re_scale, left, top, right, bottom)
        sharp_turns = filter_pts(keypt_svg["sharp_turn"] * re_scale, left, top, right, bottom)
        t_junctions = filter_pts(keypt_svg["t_junction"] * re_scale, left, top, right, bottom)
        x_junctions = filter_pts(keypt_svg["x_junction"] * re_scale, left, top, right, bottom)
            
        if len(end_points) > 0:
            keypts.append(end_points)
        if len(sharp_turns) > 0:
            keypts.append(sharp_turns)
        if len(t_junctions) > 0:
            keypts.append(t_junctions)
        if len(x_junctions) > 0:
            keypts.append(x_junctions)
        if len(keypts) > 0:
            avg_pt = np.concatenate(keypts, axis = 0)
            if len(avg_pt.shape) > 1:
                avg_pt = avg_pt.mean(axis = 0)
        else:
            avg_pt = []

        # end_points = np.array(end_points)
        # sharp_turns = np.array(sharp_turns)
        # t_junctions = np.array(t_junctions)
        # x_junctions = np.array(x_junctions)
        # star_junctions = np.array(star_junctions)
        # if len(end_points.shape) == 1:
        #     end_points = end_points[np.newaxis, ...]
        # if len(sharp_turns.shape) == 1:
        #     sharp_turns = sharp_turns[np.newaxis, ...]
        # if len(t_junctions.shape) == 1:
        #     t_junctions = t_junctions[np.newaxis, ...]
        # if len(x_junctions.shape) == 1:
        #     x_junctions = x_junctions[np.newaxis, ...]
        # if len(star_junctions.shape) == 1:
        #     star_junctions = star_junctions[np.newaxis, ...]
    else:
        end_points = []
        sharp_turns = []
        t_junctions = []
        x_junctions = []
        star_junctions = []
        avg_pt = []
    
    
    return end_points, sharp_turns, t_junctions, x_junctions, avg_pt
    

def filter_pts(pts, left, top, right, bottom, eps = 0):
    if len(pts) == 0:
        return []
    mask_x = np.logical_and(pts[:, 0] >= max(0, left - eps), pts[:, 0] <= right + eps)
    mask_y = np.logical_and(pts[:, 1] >= max(0, top - eps), pts[:, 1] <= bottom + eps)
    res_mask = np.logical_and(mask_x, mask_y)
    if len(res_mask.shape) == 1:
        res_mask = np.repeat(res_mask[..., np.newaxis], 2, axis = -1)
    res = pts[res_mask].reshape(-1, 2)
    # if len(res) > 0:
    #     res = np.unique(pts[res_mask], axis = 0)
    # if len(res.shape) == 1:
    #     res = res[np.newaxis, ...]
    return res

def avg_pts(pts):
    if len(pts.shape) > 1:
        return pts.mean(axis = 0)
    else:
        return pts
def build_grid_box(xs, ys):
    '''
    each item (pixel) is seemed as an edge box, the edge flag is bounded to the
    bottom-right corner of each box like:
      __y__ __y__      
     |     |      
     x     x     ...
     |     | 
      ...    ...

    please note that x-edge flag and y-edge flag are stored in seperate array
    '''
    edge_boxes_x, edge_boxes_y = build_query_box_edge(xs, ys)
    assert edge_boxes_x.shape == edge_boxes_y.shape
    assert edge_boxes_x.shape[0] == len(ys) - 1
    assert edge_boxes_x.shape[1] == len(xs) - 1
    coord_x, coord_y = np.meshgrid(xs, ys, indexing = 'xy')
    pts = np.stack((coord_x, coord_y), axis = -1).reshape(-1, 2)
    return edge_boxes_x, edge_boxes_y, pts

def get_edge_flags(paths, pts, keypt_svg, map_size, edge_boxes_x, edge_boxes_y, edge_map_x, edge_map_y, bbtree, edge_flag_mask, multi_grid_sizes = False):   
    # get current map size
    h, w = map_size
    scaling = 5

    # grid lines for visualization
    grid_lines = Path()
    '''
    For each grid, it will be stored like:
    A ____0____ D
     |         |
     |         |
     1         3
     |         |
      ____2____
    B           C
    '''
    # array for recording how many polylines segment with current edge box (grid)
    intersect_per_grid = np.zeros((h, w, 4)).astype(bool).astype(object)
    # compute edge flag
    for i in range(h):
        for j in range(w):
            # if i == 59 and j == 225:
            #     import pdb
            #     pdb.set_trace()
            if multi_grid_sizes == False:
                i1 = i * scaling
                i2 = (i + 1) * scaling 
                j1 = j * scaling
                j2 = (j + 1) * scaling

            # test x flags
            if edge_boxes_x[i][j][0] != False:
                assert len(edge_boxes_x[i][j]) == 2
                edge_box_x = edge_boxes_x[i][j][0]
                edge_line_x = edge_boxes_x[i][j][1]
                stroke_found = []
                stroke_temp = []
                # find AABB hit result of each edge box
                for path_index, p in bbtree.overlap_values(edge_box_x):
                    for t1, t2 in edge_line_x.intersect(p):
                        # we found a intersection point, else skip current edge
                        if t1 >= 0 and t1 <= 1:
                            if edge_map_x[i][j] == False:
                                edge_map_x[i][j] = True
                            if multi_grid_sizes == False:
                                if edge_flag_mask[i1, j1 + 1: j2].all() == False:
                                    edge_flag_mask[i1, j1 + 1: j2] = True
                            stroke_found.append((path_index, t2))
                            stroke_temp.append(path_index)

                '''
                if one edge intersect with the same in even times (first come out then
                come back eventually), that means the stroke is probably straight and 
                is quite close to the edge box boundary, so the intersection could be 
                generated by overlapping rather than two stroke with clearly oppisite 
                directions. Therefore, it very likely NOT a sharp turn and we should 
                not record it.
                '''
                if len(np.unique(stroke_temp)) == 1 and len(stroke_temp) % 2 == 0:
                    edge_map_x[i][j] = False
                    if multi_grid_sizes == False:
                        edge_flag_mask[i1, j1 + 1: j2] = False
                # else record all poly lines that intersected with current edge box
                else:
                    for path_index, t2 in stroke_found:
                        push_or_add(intersect_per_grid[i][j], 0, (path_index, t2))
                        if i > 0:
                            push_or_add(intersect_per_grid[i - 1][j], 2, (path_index, t2))

                # here 3 controls the density of edge lines for visualization, draw one line every two rows/columns
                if i % 3 == 0: 
                    grid_lines.append(edge_line_x)

            # test y flags, the logical is exactly the same
            if edge_boxes_y[i][j][0] != False:
                assert len(edge_boxes_y[i][j]) == 2
                edge_box_y = edge_boxes_y[i][j][0]
                edge_line_y = edge_boxes_y[i][j][1]
                stroke_found = []
                stroke_temp = []
                for path_index, p in bbtree.overlap_values(edge_box_y):
                    for t1, t2 in edge_line_y.intersect(p):
                        if t1 >= 0 and t1 <= 1:
                            if edge_map_y[i][j] == False:
                                edge_map_y[i][j] = True
                            if multi_grid_sizes == False:
                                if edge_flag_mask[i1+1: i2, j1].all() == False:
                                    edge_flag_mask[i1+1: i2, j1] = True
                            stroke_found.append((path_index, t2))
                            stroke_temp.append(path_index)
                if len(np.unique(stroke_temp)) == 1 and len(stroke_temp) % 2 == 0:
                    edge_map_y[i][j] = False
                    if multi_grid_sizes == False:
                        edge_flag_mask[i1+1: i2, j1] = False
                else:
                    for path_index, t2 in stroke_found:
                        push_or_add(intersect_per_grid[i][j], 1, (path_index, t2))
                        if i > 0:
                            push_or_add(intersect_per_grid[i][j - 1], 3, (path_index, t2))

                if j % 3 == 0:
                    grid_lines.append(edge_line_y)


    

    '''
    Compute the center point, the ambiguous map and the keypoint maps
    '''
    keypt_center = np.zeros((h, w, 2)).astype(float)
    under_sampled_map = np.zeros((h, w)).astype(bool)
    pts = pts.reshape((h + 1, w + 1, 2)) # we will also need to iterate all vertex coordinates as well
    pt_map = np.ones((h, w, 2)) * -1 # the intersection point with negative coordinates is not allowed in our case
    
    '''
        we will refine the keypoint from svg if they are provided,
        the ground truth key point will be different accroding to the resolution 
        of the UDF, lower UDF resolution will merge more keypoints.
    '''
    keypt_sharp_turn = []
    keypt_endpoint = []
    keypt_T = []
    keypt_X = []
    keypt_star = []
    # iterate over each edge box (grid)
    '''
        For each grid, it will be stored like:
        A ____0____ D
         |         |
         |         |
         1         3
         |         |
          ____2____
        B           C
    '''
    for i in range(h):
        for j in range(w):
            # get the vertex coordinate
            A = pts[i][j]
            B = pts[i+1][j]
            C = pts[i+1][j+1]
            D = pts[i][j+1]
            center_point = (A + B + C + D) / 4
            # if i == 59 and j == 225:
            #     import pdb
            #     pdb.set_trace()
            end_points, sharp_turns, t_junctions, x_junctions, avg_pt = query_keypt_by_bbox(keypt_svg, (A, C), 1)
            if keypt_svg is not None:
                has_end_pt = len(end_points) > 0
                has_sharp_turn = len(sharp_turns) > 0
                has_t_junction = len(t_junctions) > 0
                has_x_junction = len(x_junctions) > 0
                has_keypt = np.array([has_end_pt, has_sharp_turn, has_t_junction, has_x_junction], dtype=object).any()
            else:
                has_end_pt = False
                has_sharp_turn = False
                has_t_junction = False
                has_x_junction = False
                has_keypt = False

            # if reaches to the last row or the last column, there is no need to check key point anymore.
            if i == h - 1 or j == w - 1:
                pt_map[i][j] = center_point
                continue
            
            # check the edge intersection status
            has_intersect_path = intersect_per_grid[i][j] != False
            intersect_edge_idx = np.where(has_intersect_path)[0]
            valence = has_intersect_path.sum()
            
            # no intersection, no additional check needed
            if valence == 0:
                pt_map[i][j] = center_point
                continue
            # 1 edge has intersection, this indicates an endpoint or a sharp turn point
            # the logic will be:
            # if query result is not empty and the keypoint type matches the valence, record
            # if query result is not empty but the keypoint type doesn't match the valence, average then record
            # if not query result, compute the center point
            elif valence == 1:
                intersect_paths = intersect_per_grid[i][j][intersect_edge_idx[0]]
                if has_keypt:
                    center_point = avg_pt
                else:
                    center_point = get_avg_endpt(intersect_paths, paths, (A, C))
                if len(intersect_paths) == 1:
                    if has_end_pt:
                        center_point = avg_pts(end_points)
                        keypt_endpoint.append(center_point)
                    else:
                        keypt_endpoint.append(center_point)
                elif len(intersect_paths) == 2:
                    update_undersampled_map(under_sampled_map, i, j, h, w, intersect_edge_idx, intersect_per_grid)
                    if has_sharp_turn:
                        center_point = avg_pts(sharp_turns)
                        keypt_sharp_turn.append(center_point)
                    else:
                        keypt_sharp_turn.append(center_point)
                elif len(intersect_paths) > 2:
                    update_undersampled_map(under_sampled_map, i, j, h, w, intersect_edge_idx, intersect_per_grid)
                    keypt_star.append(center_point)

            # either an indermediate point or a sharp turn point
            elif valence == 2:
                strokes1 = []
                for pidx, _ in intersect_per_grid[i][j][intersect_edge_idx[0]]:
                    strokes1.append(pidx)
                strokes2 = []
                for pidx, _ in intersect_per_grid[i][j][intersect_edge_idx[1]]:
                    strokes2.append(pidx)

                strokes = np.array([len(strokes1), len(strokes2)])
                strokes_idx = np.unique(np.concatenate((strokes1, strokes2)))

                is_undersampled = (strokes > 1).any()
                if is_undersampled:
                    update_undersampled_map(under_sampled_map, i, j, h, w, intersect_edge_idx, intersect_per_grid)
                
                common_strokes = np.intersect1d(strokes1, strokes2)
                # if there exists a common stroke, current grid doesn't contain any keypoint
                if len(common_strokes) > 0:
                    if has_t_junction and is_undersampled:
                        center_point = avg_pts(t_junctions)
                        keypt_T.append(center_point)
                    else:
                        # always choose the first stroke
                        common_stroke = common_strokes[0]
                        min_dist, _ = paths[common_stroke].radialrange(complex(*center_point))
                        dist, t = min_dist
                        # assert dist < 0.75 # distance should always smaller than sqrt(2)/2
                        closest_pt = paths[common_stroke].point(t)
                        center_point = np.array([closest_pt.real, closest_pt.imag])
                        clip_pt(center_point, (A, C))
                # things will become a little bit complex here
                # we need more context to tell if this is a sharp turn or just indermediate point
                else:
                    if has_sharp_turn:
                        center_point = avg_pts(sharp_turns)
                        keypt_sharp_turn.append(center_point)
                    elif has_keypt: 
                        center_point = avg_pt
                        if is_undersampled:
                            keypt_star.append(avg_pt)
                    else:
                        center_point = find_center_point(strokes_idx, center_point, paths, (A, C))
                        
            # T-junction
            elif valence == 3:
                strokes1 = []
                for pidx, _ in intersect_per_grid[i][j][intersect_edge_idx[0]]:
                    strokes1.append(pidx)
                strokes2 = []
                for pidx, _ in intersect_per_grid[i][j][intersect_edge_idx[1]]:
                    strokes2.append(pidx)
                strokes3 = []
                for pidx, _ in intersect_per_grid[i][j][intersect_edge_idx[2]]:
                    strokes3.append(pidx)

                strokes = np.array([len(strokes1), len(strokes2), len(strokes3)])
                strokes_idx = np.unique(np.concatenate((strokes1, strokes2, strokes3)))

                is_undersampled = (strokes > 1).any()
                if has_t_junction:
                    center_point = avg_pts(t_junctions)
                elif has_x_junction:
                    center_point = avg_pts(x_junctions)
                elif has_keypt:
                    center_point = avg_pt
                else:
                    center_point = find_center_point(strokes_idx, center_point, paths, (A, C))
                
                if is_undersampled:
                    update_undersampled_map(under_sampled_map, i, j, h, w, intersect_edge_idx, intersect_per_grid)
                if has_keypt:
                    if is_undersampled:
                        keypt_star.append(center_point)
                    else:
                        keypt_T.append(center_point)
                    
            # X-junction
            elif valence == 4:
                strokes1 = []
                for pidx, _ in intersect_per_grid[i][j][intersect_edge_idx[0]]:
                    strokes1.append(pidx)
                strokes2 = []
                for pidx, _ in intersect_per_grid[i][j][intersect_edge_idx[1]]:
                    strokes2.append(pidx)
                strokes3 = []
                for pidx, _ in intersect_per_grid[i][j][intersect_edge_idx[2]]:
                    strokes3.append(pidx)
                strokes4 = []
                for pidx, _ in intersect_per_grid[i][j][intersect_edge_idx[3]]:
                    strokes4.append(pidx)

                strokes = np.array([len(strokes1), len(strokes2), len(strokes3), len(strokes4)])
                strokes_idx = np.unique(np.concatenate((strokes1, strokes2, strokes3, strokes4)))
                is_undersampled = (strokes > 1).any() or has_keypt == False

                if has_x_junction:
                    center_point = avg_pts(x_junctions)
                elif has_keypt:
                    center_point = avg_pt
                else:
                    center_point = find_center_point(strokes_idx, center_point, paths, (A, C))

                if is_undersampled:
                    update_undersampled_map(under_sampled_map, i, j, h, w, intersect_edge_idx, intersect_per_grid)
                if has_keypt:
                    if is_undersampled:
                        keypt_star.append(center_point)
                    else:
                        keypt_X.append(center_point)

        

            if A[0] > center_point[0]:
                print("Warning:\tfind point coordinates underflow, which should no smaller than %f but got %f"%(A[0], center_point[0]))
                center_point[0] = A[0]
            if A[1] > center_point[1]:
                print("Warning:\tfind point coordinates underflow, which should no smaller than %f but got %f"%(A[1], center_point[1]))
                center_point[1] = A[1]

            
            if C[0] < center_point[0]:
                print("Warning:\tfind point coordinates overflow, which should no larger than %f but got %f"%(C[0], center_point[0]))
                center_point[0] = C[0]
            if C[1] < center_point[1]:
                print("Warning:\tfind point coordinates overflow, which should no larger than %f but got %f"%(C[1], center_point[1]))
                center_point[1] = C[1]                
            pt_map[i][j] = center_point
    return pt_map, under_sampled_map, keypt_endpoint, keypt_sharp_turn, keypt_T, keypt_X, keypt_star

def svg_to_gt(svg, keypt = None, shorter_size = None, re_scale = None, offset = 0, approx = False, compute_udf = False, multi_grid_sizes = False):
    '''
    Given, 
        svg, string as the path to svg file
        shorter_size, int as the shorter length of the target UDF map, if it is not None,
            we will resize the svg path to the target size before we start to compute UDF
    Return,
        dist, numpy array the unsigned distance field
    '''
    DEBUG = False
    print("log:\topening %s"%svg)
    paths, (h, w) = open_svg_flatten(svg)
    if keypt is not None:
        print("log:\topening %s"%keypt)
        keypt_svg = np.load(keypt)
    else:
        print("warning:\tThe keypoint coordinates from SVG is not provided, now the code will still run for debug, however, DO NOT use the output as the ground truth for training!")
        keypt_svg = None

    ## resize and add offset
    '''
    but when generating the ground truth, we should not apply any scaling 
    otherwise there could be a mismatch between the raster/vector image and UDF and edge flags
    '''
    if shorter_size is None and re_scale is None:
        re_scale = 1.0
    elif shorter_size is not None and re_scale is None:
        # this branch is not correct, DON'T USE THIS!
        re_scale = shorter_size / h if h <= w else shorter_size / w
        h = shorter_size if h <= w else int(re_scale * h + 0.5) + 2 * offset
        w = shorter_size if w <= h else int(re_scale * w + 0.5) + 2 * offset
    elif re_scale is None:
        re_scale = 1.0
    else:
        h = int(h * re_scale + 0.5)
        w = int(w * re_scale + 0.5)
    paths = resize_path(paths, [re_scale, re_scale], offset)

    # debug for svg resize code
    '''
    if DEBUG and re_scale != 1:
        wsvg(paths, stroke_widths = [0.5]*len(paths), dimensions = (w, h), filename = svg.replace(".svg", "_%d.svg"%shorter_size))
    '''
    
    ## init edge map
    '''
    h, w here should still be in image domain
    '''
    if multi_grid_sizes:
        h2 = int(h * 2) # sub-pixel
        w2 = int(w * 2)
        hs = [h, h2]
        ws = [w, w2]
        gs = [1, 0.5] # grid size

    dims = np.array([h, w])
    edge_map_x = np.zeros(dims).astype(bool)
    edge_map_y = np.zeros(dims).astype(bool)
    if multi_grid_sizes:
        dims2 = np.array([h2, w2]).astype(int)
        edge_map_x2 = np.zeros(dims2).astype(bool)
        edge_map_y2 = np.zeros(dims2).astype(bool)
        edge_maps_xs = [edge_map_x, edge_map_x2]
        edge_maps_ys = [edge_map_y, edge_map_y2]
    
    ## construct coordinates along x, y axis with multi-resolution
    '''
    xs, ys should be dual contouring points (UDF domain), therefore its dimension should increased
    by 1 based on dimension in the image domain
    '''
    xs = np.linspace(start = 0, stop = w, num = w + 1) 
    ys = np.linspace(start = 0, stop = h, num = h + 1)
    if multi_grid_sizes:
        # 2 x 2 grid
        assert w % 2 == 0
        assert h % 2 == 0
        xs2 = np.linspace(start = 0, stop = w, num = int(w2 + 1)) 
        ys2 = np.linspace(start = 0, stop = h, num = int(h2 + 1))
        xss = [xs, xs2]
        yss = [ys, ys2]

    ## generate edges that are used as UDF computation
    paths_approx, invert_idx, invert_type, edges = get_line_approximation(paths, subdivide = 8)
    # if DEBUG:
    #     wsvg(paths_approx, stroke_widths = [0.5]*len(paths_approx), dimensions = (w, h), filename = svg.replace(".svg", "_approx.svg"))

    bbtree = build_aabbtree_from_path(paths)    
    edge_boxes_x, edge_boxes_y, pts = build_grid_box(xs, ys)
    if multi_grid_sizes:
        edge_boxes_x2, edge_boxes_y2, pts2 = build_grid_box(xss[1], yss[1])
        edge_boxes_xs = [edge_boxes_x, edge_boxes_x2]
        edge_boxes_ys = [edge_boxes_y, edge_boxes_y2]
        ptss = [pts, pts2]

    ## compute UDF, but since we compute UDF on the fly during training, so let's skip this step
    def get_udf(paths, h, w):
        # DON'T use the gradient map in the output result, it is NOT accurate!
        print("Log:\tgenerating accurate UDF")
        def get_acc_distance(distances, p_idx, pt_idx, path, pt):
            dist, _ = path.radialrange(complex(*(pt)))
            distances[p_idx][pt_idx] = dist[0]
        # https://stackoverflow.com/questions/5549190/is-shared-readonly-data-copied-to-different-processes-for-multiprocessing/5550156#5550156
        shared_array_base = multiprocessing.Array(ctypes.c_double, len(paths) * w * h)
        distances = np.ctypeslib.as_array(shared_array_base.get_obj())
        distances = distances.reshape(len(paths), w * h)
        # distances = np.zeros((len(paths), w * h))
        args_acc_udf = []
        for p_idx in range(len(paths)):
            for pt_idx in range(w * h):
                # radialrange returns the tuples (d_min, t_min) and (d_max, t_max)
                args_acc_udf.append((distances, p_idx, pt_idx, paths[p_idx], pts[pt_idx]))
        with Pool(processes=8) as pool:
            pool.starmap(get_acc_distance, args_acc_udf)
        udf = np.sort(distances, axis = 0)[0, :].reshape((h, w))
        gradients = np.zeros((h, w, 2))
        del distances
        return udf, gradients
    def get_udf_fast(edges, pts, h, w, gs, mode = 'v0'):
        # print("Log:\tgenerating UDF")
        '''
        find the closest path of each point
        '''        
        # this really uses crazy mount of RAM!
        # DON'T set resize greater than 512 if you don't have your RAM greater than 32GB!
        if mode == 'v0':
            udf, gradient = lines_to_udf_fast(np.asarray(edges, float).reshape(-1, 2), (h, w), gs)
        else:
            distances, gradients = distances_to_edges(pts.T, edges)
            dist_idx = np.argsort(distances, axis = 0)[0, :]
            gradient = np.take_along_axis(gradients, dist_idx[np.newaxis, np.newaxis, ...], axis = 0).squeeze().T.reshape(h, w, 2)
            udf = np.take_along_axis(distances, dist_idx[np.newaxis, ...], axis = 0).squeeze().T.reshape(h, w)
            if DEBUG:
                udf = np.sort(distances, axis = 0)[0, :].reshape((h, w))
                Image.fromarray(udf.astype(np.uint8)).save(svg.replace(".svg", "_udf_approx.png"))
            num_paths, num_grid_vertices = distances.shape
            assert num_grid_vertices == w * h
            del distances # release the RAM ASAP
        return udf, gradient
    if compute_udf:
        if DEBUG:
            print("log:\tcomputing UDF")
        if approx:
            if multi_grid_sizes:
                udf = []
                for hh, ww in zip(hs, ws):
                    udf_, _ = get_udf(paths, hh, ww)
                    udf.append(udf_)
            else:
                udf, gradient = get_udf(paths, h, w)
        else:    
            if multi_grid_sizes:
                udf = []
                for i, s in enumerate(zip(hs, ws)):
                    hh, ww = s
                    udf_, _ = get_udf_fast(edges, pts, int(hh), int(ww), gs[i])
                    udf_[udf_ >= 5] = 5
                    udf.append(udf_)
            else:
                udf, gradient = get_udf_fast(edges, pts, h, w, 1)
    else:
        udf = None

    # upscale the edge flag mask for better visualization result
    scaling = 5
    edge_flag_mask = np.zeros((h * scaling, w * scaling)).astype(bool)
    if multi_grid_sizes:
        edge_flag_mask2 = np.zeros((h2 * scaling, w2 * scaling)).astype(bool)
        edge_flag_masks = [edge_flag_mask, edge_flag_mask2]

    '''
    compute ground truth edge flags 
    '''
    if multi_grid_sizes:
        pt_map = []
        under_sampled_map = []
        keypt_endpoint = []
        keypt_sharp_turn = []
        keypt_T = []
        keypt_X = []
        keypt_star = []
        for i in range(len(gs)):
            if DEBUG:
                print("log:\tgenerating edge flag ground truth with grid size %d"%gs[i])
            pt_map_, under_sampled_map_, keypt_endpoint_, keypt_sharp_turn_, keypt_T_, keypt_X_, keypt_star_ =\
                get_edge_flags(paths, ptss[i], keypt_svg, (hs[i], ws[i]), edge_boxes_xs[i], edge_boxes_ys[i], 
                    edge_maps_xs[i], edge_maps_ys[i], bbtree, edge_flag_masks[i])
            pt_map.append(pt_map_)
            under_sampled_map.append(under_sampled_map_)
            keypt_endpoint.append(keypt_endpoint_)
            keypt_sharp_turn.append(keypt_sharp_turn_)
            keypt_T.append(keypt_T_)
            keypt_X.append(keypt_X_)
            keypt_star.append(keypt_star_)
        edge_map_x = edge_maps_xs
        edge_map_y = edge_maps_ys
    else:
        if DEBUG:
            print("Log:\tgenerating edge flag ground truth")
        pt_map, under_sampled_map, keypt_endpoint, keypt_sharp_turn, keypt_T, keypt_X, keypt_star =\
            get_edge_flags(paths, pts, keypt_svg, (h, w), edge_boxes_x, edge_boxes_y, edge_map_x, edge_map_y, bbtree, edge_flag_mask)

    if DEBUG and multi_grid_sizes == False:
        Image.fromarray(edge_flag_mask).save(svg.replace(".svg", "_edge_flag.png")) 
        Image.fromarray(under_sampled_map).save(svg.replace(".svg", "_under_sampled_map.png")) 
        # wsvg(grid_lines, stroke_widths = [0.5], colors = ["green"], dimensions = (w, h), filename = svg.replace(".svg", "_grid_lines.svg"))
        pts = [complex(*(pt[0], pt[1])) for pt in keypt_endpoint]
        pt_num = len(pts)
        if pt_num > 0:
            wsvg(node_radii=[0.5]*pt_num, nodes=pts, dimensions = (w, h),
                     node_colors=["red"]*pt_num, filename=svg.replace(".svg", "_1_endpt.svg"))

        pts = [complex(*(pt[0], pt[1])) for pt in keypt_sharp_turn]
        pt_num = len(pts)
        if pt_num > 0:
            wsvg(node_radii=[0.5]*pt_num, nodes=pts, dimensions = (w, h),
                     node_colors=["green"]*pt_num, filename=svg.replace(".svg", "_2_sharp.svg"))

        pts = [complex(*(pt[0], pt[1])) for pt in keypt_T]
        pt_num = len(pts)
        if pt_num > 0:
            wsvg(node_radii=[0.5]*pt_num, nodes=pts, dimensions = (w, h),
                     node_colors=["purple"]*pt_num, filename=svg.replace(".svg", "_3_T.svg"))

        pts = [complex(*(pt[0], pt[1])) for pt in keypt_X]
        pt_num = len(pts)
        if pt_num > 0:
            wsvg(node_radii=[0.5]*pt_num, nodes=pts, dimensions = (w, h),
                     node_colors=["blue"]*pt_num, filename=svg.replace(".svg", "_4_X.svg"))

        pts = [complex(*(pt[0], pt[1])) for pt in keypt_star]
        pt_num = len(pts)
        if pt_num > 0:
            wsvg(node_radii=[0.5]*pt_num, nodes=pts, dimensions = (w, h),
             node_colors=["blue"]*pt_num, filename=svg.replace(".svg", "_5_star.svg"))

    keypt_endpoint = np.array(keypt_endpoint, dtype=object)
    keypt_sharp_turn = np.array(keypt_sharp_turn, dtype=object)
    keypt_T = np.array(keypt_T, dtype=object)
    keypt_X = np.array(keypt_X, dtype=object)
    keypt_star = np.array(keypt_star, dtype=object)

    return udf, edge_map_x, edge_map_y, pt_map, under_sampled_map, keypt_endpoint, keypt_sharp_turn, keypt_T, keypt_X, keypt_star

def update_undersampled_map(under_sampled_map, i, j, h, w, intersect_edge_idx, intersect_per_grid):
    under_sampled_map[i][j] = True
    if 0 in intersect_edge_idx and i > 0:
        if len(intersect_per_grid[i][j][0]) > 1:
            under_sampled_map[i - 1][j] = True
    if 1 in intersect_edge_idx and j > 0:
        if len(intersect_per_grid[i][j][1]) > 1:
            under_sampled_map[i][j - 1] = True
    if 2 in intersect_edge_idx and i < h - 1:
        if len(intersect_per_grid[i][j][2]) > 1:
            under_sampled_map[i + 1][j] = True
    if 3 in intersect_edge_idx and j > 0:
        if len(intersect_per_grid[i][j][3]) > 1:
            under_sampled_map[i][j + 1] = True

    # if this grid is on one of the 4 corner of a cross shape flag, we need to label all left 3 grids
    # cause this will make the refine problem easier
    # at bottom right
    if 0 in intersect_edge_idx and 1 in intersect_edge_idx:
        # get top left gird intersection status
        has_intersect_path = intersect_per_grid[i - 1][j - 1] != False
        intersect_edge_topleft_idx = np.where(has_intersect_path)[0]
        if 2 in intersect_edge_topleft_idx and 3 in intersect_edge_topleft_idx:
            under_sampled_map[i - 1][j] = True
            under_sampled_map[i][j - 1] = True
            under_sampled_map[i - 1][j - 1] = True
    # at bottom left
    if 0 in intersect_edge_idx and 3 in intersect_edge_idx:
        # get top left gird intersection status
        has_intersect_path = intersect_per_grid[i - 1][j + 1] != False
        intersect_edge_topright_idx = np.where(has_intersect_path)[0]
        if 1 in intersect_edge_topright_idx and 2 in intersect_edge_topright_idx:
            under_sampled_map[i - 1][j] = True
            under_sampled_map[i][j + 1] = True
            under_sampled_map[i - 1][j + 1] = True
    # at top left
    if 2 in intersect_edge_idx and 3 in intersect_edge_idx:
        has_intersect_path = intersect_per_grid[i + 1][j + 1] != False
        intersect_edge_bottomright_idx = np.where(has_intersect_path)[0]    
        if 0 in intersect_edge_bottomright_idx and 1 in intersect_edge_bottomright_idx:
            under_sampled_map[i][j + 1] = True
            under_sampled_map[i + 1][j] = True
            under_sampled_map[i + 1][j + 1] = True
    # at top right
    if 1 in intersect_edge_idx and 2 in intersect_edge_idx:
        has_intersect_path = intersect_per_grid[i + 1][j - 1] != False
        intersect_edge_bottomleft_idx = np.where(has_intersect_path)[0]
        if 0 in intersect_edge_bottomleft_idx and 3 in intersect_edge_bottomleft_idx:
            under_sampled_map[i][j - 1] = True
            under_sampled_map[i + 1][j] = True
            under_sampled_map[i + 1][j - 1] = True

def find_center_point(strokes_idx, center_point, paths, bbox):
    res = []
    A, C = bbox
    l, t = A; r, b = C
    for idx in strokes_idx:
        min_dist, _ = paths[idx].radialrange(complex(*center_point))
        dist, t0 = min_dist
        # assert dist < 0.75
        pt = paths[idx].point(t0)
        res.append((pt.real, pt.imag))
    res = np.array(res).mean(axis = 0)
    res[0] = res[0].clip(l, r)
    res[1] = res[1].clip(t, b)
    return res

def get_avg_endpt(intersect_paths, paths, bbox):
    center_points = []
    for pidx, t2 in intersect_paths:
        t2 = 0 if t2 < 0.5 else 1
        center_point = paths[pidx].point(t2)
        center_point = np.array([center_point.real, center_point.imag])
        center_points.append(center_point)
    center_point = np.array(center_points).mean(axis = 0)
    center_point = clip_pt(center_point, bbox)
    return center_point    

def clip_pt(pt, bbox):
    A, C = bbox
    l, t = A; r, b = C
    pt[0] = pt[0].clip(l, r)
    pt[1] = pt[1].clip(t, b)
    return pt

def push_or_add(edges, eidx, pidx):
    if edges[eidx] == False:
        edges[eidx] = [pidx]
    else:
        if pidx not in edges[eidx]:
            edges[eidx].append(pidx)

def build_aabbtree_from_path(paths):
    # Build an axis-aligned bounding box tree for the segments.
    bbtree = AABBTree()
    for path_index, p in enumerate(paths):
        xmin, xmax, ymin, ymax = p.bbox()
        bbtree.add(AABB([(xmin, xmax), (ymin, ymax)]), (path_index, p))
    return bbtree

def build_query_box_edge(xs, ys, edge_size = 0):
    # record the start, end point and aabb box of each edge
    h, w = len(ys) - 1, len(xs) - 1
    edge_boxes_x = np.zeros((h, w, 2)).astype(bool).astype(object)
    for y in range(h):
        for x in range(w):
            # limits = [(xmin, xmax),
            #           (ymin, ymax),
            #           (zmin, zmax),
            #           ...]
            xmin = xs[x]; xmax = xs[x + 1]
            ymin = ys[y] - edge_size / 2; ymax = ys[y] + edge_size / 2
            box = AABB([(xmin, xmax), (ymin, ymax)])
            edge_x = Line(complex(*(xs[x], ys[y])), complex(*(xs[x + 1], ys[y])))
            edge_boxes_x[y][x][0] = box
            edge_boxes_x[y][x][1] = edge_x

    edge_boxes_y = np.zeros((h, w, 2)).astype(bool).astype(object)
    for y in range(h):
        for x in range(w):
            xmin = xs[x] - edge_size / 2; xmax = xs[x] + edge_size / 2
            ymin = ys[y]; ymax = ys[y + 1]
            box = AABB([(xmin, xmax), (ymin, ymax)])
            edge_y = (Line(complex(*(xs[x], ys[y])), complex(*(xs[x], ys[y + 1]))))
            edge_boxes_y[y][x][0] = box
            edge_boxes_y[y][x][1] = edge_y
        
    return edge_boxes_x, edge_boxes_y

def svg_to_gt_batch(svg_path, keypt_path, save_path, size = None, re_scale = None, offset = 0, approx = False, compute_udf = False, multi_grid_sizes = False):
    try:
        p, svg = path.split(svg_path)
        if path.exists(path.join(save_path, svg.replace(".svg", ".npz"))): return
        udf, edge_map_x, edge_map_y, pt_map, under_sampled_map, keypt_endpoint, keypt_sharp_turn, keypt_T, keypt_X, keypt_star \
            = svg_to_gt(svg_path, keypt_path, size, re_scale, offset, approx, compute_udf, multi_grid_sizes)
        
        print("log:\tsaving to %s"%str(path.join(save_path, svg.replace(".svg", ".npz"))))
        np.savez_compressed(path.join(save_path, svg.replace(".svg", ".npz")), 
            udf = udf, edge_x = edge_map_x, edge_y = edge_map_y, pt_map = pt_map, under_sampled = under_sampled_map, 
            end_point = keypt_endpoint, sharp_turn = keypt_sharp_turn, T = keypt_T, X = keypt_X, star = keypt_star)
    except Exception as e:
        print("log:\tunsupport SVG, skip current file.")
    return 0

def lines_to_udf(paths, canvas_size):
    h, w = canvas_size
    # xs = np.linspace(start = 0, stop = w - 1, num = w) + 0.5
    # ys = np.linspace(start = 0, stop = h - 1, num = h) + 0.5
    xs = np.linspace(start = 0, stop = w - 1, num = w)
    ys = np.linspace(start = 0, stop = h - 1, num = h)
    coord_x, coord_y = np.meshgrid(xs, ys, indexing = 'xy')
    pts = np.stack((coord_x, coord_y), axis = -1).reshape(-1, 2)
    # import pdb
    # pdb.set_trace()
    # distances = distances_to_edges_mp(pts.T, paths.reshape(-1, 2, 2))
    # start_time = time.time()
    distances, _ = distances_to_edges(pts.T, paths.reshape(-1, 2, 2))
    # print("%s seconds"%(time.time() - start_time))
    # assert (distances == distances0).all()
    udf = np.sort(distances, axis = 0)[0, :].reshape((h, w))
    del distances
    return udf

def lines_to_udf_fast(pts, canvas_size, grid_size):
    '''
    Given:
        pts, 2N x 2 array, contains start/end point pairs of polylines
        canvas_size, 2 x 1 tuple, (height, width) of the target canvas
    Return:
        udf, h+1 x w+1 array, the unsigned distance field computed based on the given polyline
        grad, h+1 x w+1 x 2 array, the corresponding gradient map
    we assume the vector image and raster image has the same canvas size, so when converting from image domain
    to dual contouring domain we need to expand each pixel/grid into 4 distance sampling points. That is why
    increasing the height and width by 1 and shift the coordinate by half of the grid size when we generate the xs ys 
    '''
    h, w  = canvas_size
    hg = int(h * grid_size)
    wg = int(w * grid_size)
    xs = np.linspace(start = 0, stop = wg, num = w + 1)
    ys = np.linspace(start = 0, stop = hg, num = h + 1)
    pts_grid = np.stack(np.meshgrid(xs, ys, indexing='xy'), axis = -1).reshape(-1, 2)
    pts_grid = np.ascontiguousarray(pts_grid)
    line_segments = np.arange( pts.shape[0] ).reshape(-1,2)
    line_segments = np.ascontiguousarray(line_segments)
    pts = np.ascontiguousarray(pts)
    udf, closest_pt = edge_distance_aabb.AABBDistances( pts_grid, pts, line_segments )
    
    # compute the gradient
    pts_grid = pts_grid.reshape(h + 1, w + 1, 2)
    closest_pt = closest_pt.reshape(h + 1, w + 1, 2)
    grad = (pts_grid - closest_pt)
    norm_len = np.sqrt((grad * grad).sum(axis = -1, keepdims = True))
    eps = 1e-10
    norm_len[norm_len < eps] = eps
    grad = grad / norm_len
    
    return udf.reshape(h + 1, w + 1),  grad
            
def save_pt_to_svg(keypts_dict, output, size):
    w, h = size
    keypts = []
    for key in keypts_dict:
        ptlist = keypts_dict[key]
        if len(ptlist) > 0:
            for i in range(len(ptlist)):
                keypts.append(complex(*ptlist[i]))
    pt_num = len(keypts)
    wsvg(node_radii=[0.5]*pt_num, nodes=keypts, dimensions = (w, h),
         node_colors=["red"]*pt_num, filename=output)

'''
Function adapted from Yotam
'''
def distances_to_edges_mp(pts, edges, pros = 7):
    start_time = time.time()
    split = np.split(pts, np.arange(pts.shape[1] // (pros - 1), pts.shape[1], pts.shape[1] // (pros - 1)), axis = 1)
    assert len(split) == pros
    args = []
    
    for pts_split in split:
        args.append((pts_split, edges))

    with Pool(processes=pros) as pool:
        dist, _ = pool.starmap(distances_to_edges, args)
    print("%s seconds"%(time.time() - start_time))
    return np.concatenate(dist, axis = -1)

def distances_to_edges( pts, edges ):
    '''
    Given:
        pts, array with shape 2 x N, where first dimension means x, y coordinate, N means the number of points that need to compute distance
        edges, array with shape M x 2 x 2, where M means the number of edges, second dimension means 
    Returns:
        a tuple, where the first element is
        an array of distances with dimensions #edges x #pts
        and the second element is
        an array of gradients with respect to 'pts' with dimensions #edges x N coordinates (x,y,...) x #pts.
    '''
    start_time = time.time()
    pts = np.asarray( pts, float )
    ## pts has dimensions N (x,y,...) x #pts
    edges = np.asarray( edges, float )
    ## edges has dimensions ... x #edges x 2 endpoints x N coordinates (x,y,...)
    
    N = pts.shape[0] # N means the coordinate dimension?
    
    assert len( pts.shape ) == 2 and pts.shape[0] == N
    assert edges.shape[-2] == 2 and edges.shape[-1] == N
    
    ## get distance squared to each edge:
    ##   let p = black_pixel_pos, a = endpoint0, b = endpoint1, d = ( b-a ) / sqrt( dot( b-a,b-a ) )
    ##   dot( p-a, d ) < 0 => sqrt( dot( p-a, p-a ) )
    ##   dot( p-a, d ) > 1 => sqrt( dot( p-b, p-b ) )
    ##   else              => sqrt( dot( p - (a + dot( p-a, d ) * d), same ) )
    ## and we can also get the gradient map as: norm( p - a + dot( p-a, d ) * d )
    p_a = pts[np.newaxis,...] - edges[...,:,0,:,np.newaxis]
    p_b = pts[np.newaxis,...] - edges[...,:,1,:,np.newaxis]
    ## p_a and p_b have dimensions ... x #edges x N coordinates (x,y,...) x #pts
    b_a = edges[...,:,1,:] - edges[...,:,0,:]
    ## b_a has dimensions ... x #edges x N coordinates (x,y,...)
    d = b_a / ( b_a**2 ).sum( -1 )[...,np.newaxis]
    d_ = b_a / np.sqrt(( b_a**2 ).sum( -1 )[...,np.newaxis])
    ## d has same dimensions as b_a
    assert b_a.shape == d.shape
    # why? why? why this condition is correct?
    cond = ( p_a * d[...,np.newaxis] ).sum( -2 ) 
    cond_ = ( p_a * d_[...,np.newaxis] ).sum( -2 )
    ## cond has dimensions ... x #edges x #pts
    assert cond.shape[-2:] == (edges.shape[-3], pts.shape[-1])
    
    cond_lt_zero = cond < 0
    cond_gt_one = cond > 1
    cond_else = np.logical_not( np.logical_or( cond_lt_zero, cond_gt_one ) )
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
    # p' = a + dot(p - a, d) * d
    p_prime = edges[...,:,0,:,np.newaxis] + cond_[:,np.newaxis,:] * d_[..., np.newaxis]
    p_prime[ cond_lt_zero[:, np.newaxis, :].repeat(2, axis = 1) ] = edges[...,:,0,:][..., np.newaxis].repeat(pts.shape[1], axis = 2)[cond_lt_zero[:, np.newaxis, :].repeat(2, axis = 1)]
    p_prime[ cond_gt_one[:, np.newaxis, :].repeat(2, axis = 1) ] = edges[...,:,1,:][..., np.newaxis].repeat(pts.shape[1], axis = 2)[cond_gt_one[:, np.newaxis, :].repeat(2, axis = 1)]
    gradients = pts[np.newaxis,...] - p_prime
    distancesSqr = (gradients**2).sum(-2)
    # distancesSqr_ = ( ( p_a - cond[:,np.newaxis,:] * b_a[...,np.newaxis] )**2 ).sum( -2 )
    # <=>
    # distancesSqr[ cond_else ] = ( (
    #    swapaxes( p_a, -1, -2 )[ cond_else ] - swapaxes( cond[:,newaxis,:] * b_a[...,newaxis], -1, -2 )[ cond_else ]
    #    )**2 ).sum( -1 )
    
    ## < 0 case
    # distancesSqr[ cond < 0 ] = ( p_a**2 ).sum( -2 )[ cond < 0 ]
    # <=>ex
    distancesSqr[ cond_lt_zero ] = ( p_a**2 ).sum( -2 )[ cond_lt_zero ]
    # distancesSqr_[ cond_lt_zero ] = ( p_a**2 ).sum( -2 )[ cond_lt_zero ]
    # <=>
    # distancesSqr[ cond_lt_zero ] = ( swapaxes( p_a, -1, -2 )[ cond_lt_zero ]**2 ).sum( -1 )
    
    ## > 1 case
    # distancesSqr[ cond > 1 ] = ( p_b**2 ).sum( -2 )[ cond > 1 ]
    # <=>
    distancesSqr[ cond_gt_one ] = ( p_b**2 ).sum( -2 )[ cond_gt_one ]
    # distancesSqr_[ cond_gt_one ] = ( p_a**2 ).sum( -2 )[ cond_gt_one ]
    # <=>
    # distancesSqr[ cond_gt_one ] = ( swapaxes( p_b, -1, -2 )[ cond_gt_one ]**2 ).sum( -1 )
    
    #print 'distancesSqr:', distancesSqr
    #print 'distances:', sqrt( distancesSqr.min(0) )
    
    ## distancesSqr is now distances
    # we make this inplace to save memory I guess
    np.sqrt( distancesSqr, distancesSqr )
    ## we'll just rename distancesSqr
    distances = distancesSqr
    ## distances has dimensions ... x #edges x #pts
    del distancesSqr
    ## remove nan gradient if necessary
    nan_mask = distances == 0
    if nan_mask.sum() > 0:
        distances_ = distances.copy()
        distances_[nan_mask] = 1
        gradients = (gradients * (1 - nan_mask.astype(float))[:, np.newaxis, :]) / distances_[:, np.newaxis, :]
    else:
        gradients = gradients / distances[:, np.newaxis, :]
    
    # print("log:\tcomputing distance used %s"%(time.time() - start_time))
    return distances, gradients

if __name__ == "__main__":
    __spec__ = None
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", default = -1, help = "decide how many groups to split")
    parser.add_argument("-g", default = -1, help = "current group index")
    arg = parser.parse_args()
    # test if functions work correctly
    '''test differentiable tensor to svg'''
    # test_np = np.load("../experiments/udf/maps/0000008.npz")
    # edge_map = test_np["edge_maps"]
    # keypt_map = test_np["keypt_map"]
    # edge_map = torch.from_numpy(edge_map).float()
    # edge_map.requires_grad = True
    # keypt_map = torch.from_numpy(keypt_map).float()
    # keypt_map.requires_grad = True
    # canvas_h, canvas_w, _ = edge_map.shape
    # canvas_h = canvas_h * 2 + 1
    # canvas_w = canvas_w * 2 + 1
    # lines = pre_to_lines(edge_map, keypt_map)
    # # lines_to_svg(lines, canvas_w, canvas_h, "output.svg")
    # lines_to_svg(lines.detach().numpy(), canvas_w, canvas_h, "output.svg")

    # let's write a function that convert svg paths to udf
    test = False
    if test:
        '''
        generate edge flag for single sample
        '''
        # svg_path = "../data/exp/svg/0000014_300.svg"
        svg_path = "../data/exp/svg/"
        save_path = "../data/exp/gt"
        keypt_path = "../data/exp/keypt/0000014_300.npz"
        gt_path = "../data/exp/gt/0000014_300.npz"
        multi_grid_sizes = True
        def vis_test(gt_path, multi_grid_sizes):
            gts = np.load(gt_path, allow_pickle=True)
            edge_x = gts["edge_x"]
            edge_y = gts["edge_y"]
            pt_map = gts["pt_map"]
            # keypts_dict = {}
            # keypts_dict['end_point'] = gts['end_point']
            # keypts_dict['sharp_turn'] = gts['sharp_turn']
            # keypts_dict['T'] = gts['T']
            # keypts_dict['X'] = gts['X']
            # keypts_dict['star'] = gts['star']
            under_sampled_map = gts["under_sampled"]
            if multi_grid_sizes:
                h, w = edge_x[0].shape
                for i in range(len(edge_x)):
                    edge_map = np.stack((edge_x[i], edge_y[i]), axis = -1)
                    # h, w = edge_x[i].shape # we always has the same canvas size
                    lines, lines_map_x, lines_map_y, _ = map_to_lines(edge_map, pt_map[i])
                    lines_to_svg(lines, w, h, gt_path.replace(".npz", "_rec_%.1f.svg"%i), 'xy')
                    # Image.fromarray(under_sampled_map[i]).save(gt_path.replace(".npz", "_usm_%d.png"%i))
                # # overlay all under sampled map into one image
                # hh, ww = under_sampled_map[0].shape
                # c1 = np.array([[255, 255, 255], [255, 0, 0]]).astype(np.uint8) #red
                # usm1 = c1[under_sampled_map[0].astype(int)]
                # c2 = np.array([[255, 255, 255], [0, 255, 0]]).astype(np.uint8) #green
                # usm2 = c2[under_sampled_map[1].astype(int)]
                # usm2 = cv2.resize(usm2, (ww, hh), interpolation = cv2.INTER_NEAREST)
                # c4 = np.array([[255, 255, 255], [0, 0, 255]]).astype(np.uint8) #blue
                # usm4 = c4[under_sampled_map[2].astype(int)]
                # usm4 = cv2.resize(usm4, (ww, hh), interpolation = cv2.INTER_NEAREST)
                # c8 = np.array([[255, 255, 255], [0, 125, 255]]).astype(np.uint8) #cyan
                # usm8 = c8[under_sampled_map[3].astype(int)]
                # usm8 = cv2.resize(usm8, (ww, hh), interpolation = cv2.INTER_NEAREST)
                # comb = (0.3*usm1 + 0.3*usm2 + 0.3*usm4 + 0.3*usm8).clip(0, 255)
                # Image.fromarray(comb.astype(np.uint8)).save(gt_path.replace(".npz", "_usm_comb.png"))
            else:
                edge_map = np.stack((edge_x, edge_y), axis = -1)
                h, w = edge_x.shape
                lines, lines_map_x, lines_map_y, _ = map_to_lines(edge_map, pt_map)
                lines_to_svg(lines, w, h, gt_path.replace(".npz", "_rec.svg"), 'xy')
        if ".svg" in svg_path:
            svg_to_gt_batch(svg_path, keypt_path, save_path, None, None, 0, compute_udf = False, multi_grid_sizes = True)
            '''
            reconstruct from the generated edge flags
            '''
            
            vis_test(gt_path, multi_grid_sizes)
            # lines = refine_topology(edge_map, pt_map, under_sampled_map, lines_map_x, lines_map_y, keypts_dict)
            # lines_to_svg(lines, w, h, gt_path.replace(".npz", "_rec_refined.svg"), 'xy')
            # save_pt_to_svg(keypts_dict, gt_path.replace(".npz", "_keypt.svg"), (w, h))
            # lines_to_svg(lines, w, h, gt_path.replace(".npz", "_rec.svg"), 'xy')
            # Image.fromarray(under_sampled_map).save(gt_path.replace(".npz", "_under.png"))
        else:
            for f in os.listdir(svg_path):
                if ".svg" not in f: continue
                svg_path_ = os.path.join(svg_path, f)
                keypt_path = os.path.join(svg_path.replace("svg", "keypt"), f.replace(".svg", ".npz"))
                gt_path = os.path.join(save_path, f.replace(".svg", ".npz").replace("svg", "keypt"))
                svg_to_gt_batch(svg_path_, keypt_path, save_path, compute_udf = True, multi_grid_sizes = multi_grid_sizes)
                vis_test(gt_path, multi_grid_sizes)
    else:       
        svg_p = "../data/full/svg"
        keypt_p = "../data/full/keypt"
        save_p = "../data/full/gt"
        svgs = os.listdir(svg_p)
        svgs.sort()
        args = []
        for svg in svgs:
            svg_path = path.join(svg_p, svg)
            keypt_path = path.join(keypt_p, svg.replace(".svg", ".npz"))
            num = int(svg.replace('.svg', ''))
            if arg.s != '-1':
                split = int(arg.s)
                group = int(arg.g)
                assert split > 1 and group >= 0
                assert group < split
                if num % split != group: continue
            save_path = save_p
            args.append((svg_path, keypt_path, save_path, None, None, 0, False, False, True))
            # svg_to_gt_batch(svg_path, keypt_path, save_path, None, None, 0, compute_udf = True, multi_grid_sizes = True)
        with Pool(processes=32) as pool:
            pool.starmap(svg_to_gt_batch, args)

    
