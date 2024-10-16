from pathlib import Path as P
directory = P(__file__)
import sys
mp = str(directory.parent.parent)
if mp not in sys.path: sys.path.append(mp)

import os
import numpy as np
import torch
import cv2

from random import sample, randint, shuffle
from os.path import join, exists
from torchvision import transforms as T
from utils.ndc_tools import map_to_lines, lines_to_svg, lines_to_udf, refine_topology, vis_edge_map, save_pt_to_svg, get_udf_mask, lines_to_udf_fast
from dataset.augmentation import flip_edge, flip_keypt_map, random_bbox, crop_edge, crop_keypt_map
from utils.svg_tools import open_svg_flatten, flip_path, path_to_pts, pts_to_path, crop_path
from utils.dual_contouring import gradient
from utils.ndc_tools import vis_grad

DEBUG = False
if DEBUG:
    from svgpathtools import Path, Line, wsvg

class noisy_udf(torch.utils.data.Dataset):
    def __init__(self, data_dir, device = None, noise_level = 0.01, is_train = True, approx_udf = False,
        dist_clip = 4.5, dist_ndc = 1, crop_size = 256, insert_skel = False, multi_scale = False):
        self.noise_level = noise_level
        self.dist_clip = dist_clip
        self.dist_ndc = dist_ndc
        self.data_dir = data_dir
        self.is_train = is_train
        self.kernel = np.array([[0,1,0],[1,1,1],[0,1,0]], dtype=np.uint8)
        self.crop_size = crop_size
        self.insert_skel = insert_skel
        self.multi_scale = multi_scale
        self.grid_size = [1, 0.5] # pls don't change the size order, this is predefined
        self.approx_udf = approx_udf
        if approx_udf:
            print("warning:\tapprox_udf flag is ON, this training will not lead to a usable model. Unless you are deliberately doing this, TURN OFF this flag!!")
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        # read or construct the file list
        self.idx_path = join(self.data_dir, "noisy_udf_list.txt")
        if exists(self.idx_path) == False:
            print("Log:\tcan't find UDF idx file, scanning the whole dataset")
            self.idx = self.scan()
        else:
            with open(self.idx_path, "r") as f:
                self.idx = [name.strip().strip("\n") for name in f.readlines()]
        
        # split training set and validation set
        if self.is_train:  # 95% as the training set
            self.idx = self.idx[:int(len(self.idx)*0.992)]
            self.idx = self.idx # extend training set
        else:  # 5% as the test set
            self.idx = self.idx[int(len(self.idx)*0.992):]
        
    def scan(self):
        # scan the full dataset and construct the idx
        gt_path = join(self.data_dir, "gt") # path to the flag maps
        gt_list = os.listdir(gt_path)
        gt_list = [join(gt_path, i) for i in gt_list]
        shuffle(gt_list)
        with open(self.idx_path, "w") as f:
            f.write("\n".join(gt_list))
        return gt_list

    def __len__(self):
        return len(self.idx)

    def rand_transpose_s2(self, pts, canvas_size, edge_maps_np, pt_map, keypts_dict, udf, usm, p = 0.5):
        dice = np.random.rand() # roll the dice!
        # if True:
        if dice > p:
            h, w = canvas_size
            canvas_size = (w, h)   
            pts[:, [0, 1]] = pts[:, [1, 0]]
            
            edge_maps_np = edge_maps_np.transpose((1, 0, 2))
            edge_maps_np[..., [0, 1]] = edge_maps_np[..., [1, 0]]
            
            pt_map[..., [0, 1]] = pt_map[..., [1, 0]]
            pt_map = pt_map.transpose((1, 0 , 2))

            if udf is not None:
                udf = udf.T
            usm = usm.T

            for k in keypts_dict:
                if len(keypts_dict[k]) > 0:
                    keypts_dict[k][..., [0, 1]] = keypts_dict[k][..., [1, 0]]
        
        return pts, edge_maps_np, pt_map, keypts_dict, canvas_size, udf, usm
        
    def rand_flip_s2(self, pts, canvas_size, edge_maps_np, pt_map, keypts_dict, udf, usm, p = 0.5):
        dice = np.random.rand() # roll the dice!
        # if True:
        if dice > p: 
            '''
            .→.→.→.→.→.→.→.→
            ↓ ↓ ↓ ↓ ↓ ↓ ↓ ↓  ...
            0. x-axis edge map
            1. y-axis edge map
            '''
            edge_maps_np = flip_edge(edge_maps_np, axis = 'x')
            for key in keypts_dict:
                keypts_dict[key] = flip_path(keypts_dict[key], 'x', canvas_size, 'xy')
            pt_map = flip_keypt_map(pt_map, 'x', canvas_size, 'xy')
            pts = flip_path(pts, 'x', canvas_size, 'xy')
            if udf is not None:
                udf = np.flip(udf, axis = 0)
            usm = np.flip(usm, axis = 0)
            # grad = np.flip(grad, axis = 0)
            # grad[..., 1] = grad[..., 1] * -1
        
        dice = np.random.rand() # roll the dice again!
        # if True:
        if dice > p:
            edge_maps_np = flip_edge(edge_maps_np, axis = 'y')
            for key in keypts_dict:
                keypts_dict[key] = flip_path(keypts_dict[key], 'y', canvas_size, 'xy')
            pt_map = flip_keypt_map(pt_map, 'y', canvas_size, 'xy')
            pts = flip_path(pts, 'y', canvas_size, 'xy')
            if udf is not None:
                udf = np.flip(udf, axis = 1)
            usm = np.flip(usm, axis = 1)
            # grad = np.flip(grad, axis = 1)
            # grad[..., 0] = grad[..., 0] * -1

        return pts, edge_maps_np, pt_map, keypts_dict, udf, usm
    
    def rand_crop_s2(self, pts, canvas_size, edge_maps_np, pt_map, keypts_dict, udf, usm, bbox_size = 256, gsize = 1, junc_pts = None):
        bbox0, bbox1 = random_bbox(canvas_size, gsize, bbox_size, edge_maps_np, junc_pts)
        assert int(bbox_size * gsize) == bbox_size * gsize
        bbox_real = (bbox0[0]*gsize, bbox0[1]*gsize, bbox0[2]*gsize, bbox0[3]*gsize)
        pts, new_end_pt = crop_path(pts, bbox_real, canvas_size, mode = 'path')
        pt_map = crop_keypt_map(pt_map, bbox0, gsize)
        if len(new_end_pt) > 0:
            if len(keypts_dict['end_point']) > 0:
                keypts_dict['end_point'] = np.concatenate((keypts_dict['end_point'], new_end_pt), axis = 0)
            else:
                keypts_dict['end_point'] = new_end_pt
        for key in keypts_dict:
            keypts_dict[key] = crop_path(keypts_dict[key], bbox_real, canvas_size)

        edge_maps_np = crop_edge(edge_maps_np, bbox0)
        if udf is not None:
            udf = crop_edge(udf, bbox1)
        usm = crop_edge(usm, bbox0)
        return pts, edge_maps_np, pt_map, keypts_dict, udf, usm, (int(bbox_size * gsize), int(bbox_size * gsize))
        
    def to_tensor(self, img_np, clip = -1):
        # we don't nee to normalize the input but 
        # a value clipping is still necessary
        if clip > 0:
            img_np = img_np.clip(0, clip) / clip # this is unsigned distance so no negative values
        transforms = T.Compose(
            [
                T.ToTensor()
            ]
        )
        return transforms(img_np).to(self.device)

    def __getitem__(self, index):
        ## reading from GT files
        # open GT numpy array
        gts = np.load(self.idx[index], allow_pickle = True)
        # open svg file
        paths, canvas_size = open_svg_flatten(self.idx[index].replace("gt", 'svg').replace('.npz', '.svg'))
        # convert paths into point list
        pts, _ = path_to_pts(paths)
        # random decide the grid size of current sample
        if self.multi_scale == False:
            gsize_idx = 0
        else:
            gsize_idx = randint(0, len(self.grid_size) - 1)
        # gsize_idx = 1 # force the 0.5 grid size
        gsize = self.grid_size[gsize_idx]
        # read UDF
        # udf_np = gts['udf'][gsize_idx] # we should generate UDF on the fly!
        # read edge flags
        edge_maps_np_x = gts["edge_x"][gsize_idx]
        edge_maps_np_y = gts["edge_y"][gsize_idx]
        edge_maps_np = np.stack((edge_maps_np_x, edge_maps_np_y), axis = -1)
        # read point map
        pt_map = gts["pt_map"][gsize_idx]
        # read keypoints
        keypts_dict = {}
        keypts_dict['end_point'] = np.array(gts['end_point'][gsize_idx])
        keypts_dict['sharp_turn'] = np.array(gts['sharp_turn'][gsize_idx])
        keypts_dict['T'] = np.array(gts['T'][gsize_idx])
        keypts_dict['X'] = np.array(gts['X'][gsize_idx])
        keypts_dict['star'] = np.array(gts['star'][gsize_idx])

        # read undersampled map, we won't use this for training
        # however, it is still necessary for validation our augmentation correctness
        usm_np = gts["under_sampled"][gsize_idx]
        udf_np = None

        '''
        FOR EXPERIMENT ONLY, enable this will bring significant bad reconstruction result!
        '''
        if self.approx_udf:
            assert self.multi_scale == False
            udf_np, _ = lines_to_udf_fast(pts, (int(canvas_size[0]), int(canvas_size[1])), 1)
        '''
        FOR EXPERIMENT ONLY, enable this will bring significant bad reconstruction result!
        '''
        
        # apply augmentation
        if self.is_train:
            pts, edge_maps_np, pt_map, keypts_dict, canvas_size, udf_np, usm_np =\
                self.rand_transpose_s2(pts, canvas_size, edge_maps_np, pt_map, keypts_dict, udf_np, usm_np)
            
            pts, edge_maps_np, pt_map, keypts_dict, udf_np, usm_np =\
                self.rand_flip_s2(pts, canvas_size, edge_maps_np, pt_map, keypts_dict, udf_np, usm_np)
            
            '''
                concatenate sharp turn and junction points, cause bbox
                that contains those points usually means topology complex
                regions
            '''
            keypts = [keypts_dict['sharp_turn'], keypts_dict['T'], keypts_dict['X'], keypts_dict['star']]
            for i in range(len(keypts) - 1, -1, -1):
                if len(keypts[i]) == 0:
                    keypts.pop(i)
            if len(keypts) > 0:
                keypts = np.concatenate(keypts, axis = 0)
            else:
                keypts = None
            pts, edge_maps_np, pt_map, keypts_dict, udf_np, usm_np, canvas_size =\
                self.rand_crop_s2(pts, canvas_size, edge_maps_np, pt_map, keypts_dict, udf_np, usm_np, self.crop_size, gsize, keypts)
            assert int(self.crop_size * gsize) == self.crop_size * gsize

        # re-compute the UDF from svg paths
        if len(pts) == 0:
            print("error:\tempty patch created from sample %s"%self.idx[index])
            from PIL import Image
            Image.fromarray(img).save("prob_img.png")
            Image.fromarray(np.logical_or(edge_maps_np[..., 0], edge_maps_np[..., 1])).save("prob_edge.png")
            # raise ValueError("error:\tempty patch created from sample %s"%self.idx[index])
            udf_np = np.ones((int(canvas_size[0] / gsize) + 1, int(canvas_size[1] / gsize) + 1)) * self.dist_clip
        else:
            if self.approx_udf == False:
                udf_np, _ = lines_to_udf_fast(pts, (int(canvas_size[0] / gsize), int(canvas_size[1] / gsize)), gsize)
            else:
                assert udf_np is not None

        # create sketch skeleton 
        skel_np = np.logical_or(edge_maps_np[..., 0], edge_maps_np[..., 1])

        # create edge flag mask
        edge_mask_np = np.logical_or(edge_maps_np[:, :, 0], edge_maps_np[:, :, 1])
        edge_mask_np = cv2.dilate(edge_mask_np.astype(np.uint8), self.kernel, iterations = 1).astype(bool)
        
        if self.insert_skel:
            udf_np[np.where(skel_np != 0) ] = 0

        # add noise to UDF
        if self.noise_level > 0:
            w = np.random.uniform(high = self.noise_level)
            udf_noise = np.abs(np.random.normal(loc = 0, scale = w * udf_np.std(), size = udf_np.shape))
            udf_np = udf_np + udf_noise

        if DEBUG:
            canvas_h, canvas_w = canvas_size
            # visualize the augmented svg
            lines = pts_to_path(pts)
            wsvg(lines, stroke_widths = [0.5]*len(lines), dimensions = (canvas_w, canvas_h), filename = "%04d_gt.svg"%index)

            # visualize the under sampled map and sketch skeleton from UDF
            from PIL import Image
            Image.fromarray(usm_np).save("%04d_under.png"%index)
            Image.fromarray(udf_np < 1).save("%04d_udf.png"%index)

            # visualize the keypoint
            save_pt_to_svg(keypts_dict, "%04d_keypt.svg"%index, (canvas_w, canvas_h))
            # visualize the reconstruction
            vis_edge_map(edge_maps_np, png = "%04d_edge.png"%index)

            lines_rec, lines_map_x, lines_map_y, _ = map_to_lines(edge_maps_np, pt_map)
            lines_rec = refine_topology(edge_maps_np, pt_map, usm_np, lines_map_x, lines_map_y, keypts_dict)
            lines_to_svg(lines_rec, canvas_w, canvas_h, "%04d_gt_recon.svg"%index, indexing = 'xy')
            
        # create mask we don't need to create keypoint mask cause that could be generated by the edge maps
        edge_maps_x = edge_maps_np[:, :, 0].squeeze()
        edge_maps_y = edge_maps_np[:, :, 1].squeeze()
        # convert edge map to labels
        edge_maps = edge_maps_x + 2 * edge_maps_y

        # convert to tensor
        udf = self.to_tensor(udf_np.copy().astype(float), clip = self.dist_clip)
        edge_maps = self.to_tensor(edge_maps.copy().astype(float))
        edge_mask = self.to_tensor(edge_mask_np.copy().astype(float))
        skel = self.to_tensor(skel_np.copy().astype(float))
        pt_map = self.to_tensor(pt_map.copy().astype(float))
        gsize = torch.Tensor([gsize]).to(self.device)

        # pad keypts dictionary
        for key in keypts_dict:
            keypt = keypts_dict[key]
            keypt_padded = np.ones((500, 2)) * -1
            if len(keypt) > 0:
                keypt_padded[:len(keypt), ...] = keypt
            keypts_dict[key] = keypt_padded

        return udf, edge_maps, pt_map, edge_mask, skel, keypts_dict, gsize

'''for debug'''
def disp_np(img_np):
    from PIL import Image
    Image.fromarray(img_np.squeeze()).show()