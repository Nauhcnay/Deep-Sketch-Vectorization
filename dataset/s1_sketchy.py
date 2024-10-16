import torch
import copy
import random
import cv2
import torchvision.transforms as T
import numpy as np
# import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.io import read_image
from os import listdir
from os.path import exists, join, splitext, isdir
from tqdm import tqdm
from dataset.preprocess import get_coords_from_heatmap, plot_points
from scipy.ndimage import binary_dilation
from scipy.ndimage.interpolation import rotate
from skimage.morphology import skeletonize

import edge_distance_aabb
from utils.ndc_tools import map_to_lines, lines_to_svg, lines_to_udf_fast, refine_topology, vis_edge_map, save_pt_to_svg
from dataset.augmentation import flip_edge, flip_keypt_map, random_bbox, crop_edge, crop_keypt_map
from utils.svg_tools import open_svg_flatten, flip_path, path_to_pts, pts_to_path, crop_path, rotate_pts
from dataset.preprocess import rasterize, svg_to_numpy
from dataset.augmentation import blend_skeletons

from io import BytesIO
from PIL import Image
from random import shuffle
import time

BG_COLOR = 209
BG_SIGMA = 5
MONOCHROME = 1

class SketchyDataset(Dataset):
    '''
    under reconstruction...
    '''
    def __init__(self, img_dir, re_idx = False, patch_size = 256, device = None, 
            insert_skel = False, transform_pixel = None, up_scale = False,
            dist_clip = 8.5, dist_ndc = 3, is_train = True, no_rotation = False, 
            dist_pt = 3, jpg = False, bg = False, approx_udf = False):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        self.img_list = None
        self.is_train = is_train
        self.no_rotation = no_rotation
        self.insert_skel = insert_skel
        self.jpg = jpg
        self.grid_size = [1, 0.5]
        self.up_scale = up_scale
        self.bg = bg
        self.approx_udf = approx_udf
        if approx_udf:
            print("warning:\tapprox_udf flag is ON, this training will not lead to a usable model. Unless you are deliberately doing this, TURN OFF this flag!!")
        
        # initial necessary file pathes
        self.path_index = join(img_dir, "img_index.txt")
        self.path_gt = join(img_dir, "gt") # now we have point instead of images!
        self.path_svg = join(img_dir, "svg")
        self.path_train = join(img_dir, "png")
        brushes = []
        for b in listdir(self.path_train):
            if "_" not in b and isdir(join(self.path_train, b)) and b.isnumeric():
                brushes.append(b)
        self.brushes = brushes
        self.dist_clip = dist_clip
        self.dist_ndc = dist_ndc
        self.dist_pt = dist_pt

        # generate train set index if necessary
        if exists(self.path_index) == False or re_idx:
            print("Log:\tcan't find index file of the dataset or force re-index enabled, start to re-index")
            self.re_index()
        else:
            with open(self.path_index, "r") as f:
                self.img_list = f.readlines()
                self.img_list = [img.strip("\n") for img in self.img_list]
        idx = int(len(self.img_list) * 0.9)
        if is_train:
            self.img_list = self.img_list[:idx]
            self.img_list = self.img_list
        else:
            self.img_list = self.img_list[idx:]
        
        # initial pixel transform method
        self.patch_size = patch_size
        self.transform_pixel = transform_pixel
        self.kernel = np.array([[0,1,0],[1,1,1],[0,1,0]]).astype(np.uint8)
    
    def randomJPEGcompression(self, img_np):
        img = Image.fromarray(img_np).convert('RGB')
        qf = random.randrange(35, 95)
        outputIoStream = BytesIO()
        img.save(outputIoStream, "JPEG", quality=qf, optimice=True)
        outputIoStream.seek(0)
        return np.array(Image.open(outputIoStream).convert('L'))

    def re_index(self):
        print("Log:\tscaning %s..."%self.brushes)
        train_list1 = []
        # 1st scan the full image number form train folder
        # keep the images that exist in all brush folders
        for b in self.brushes:
            scaned_img = listdir(join(self.path_train, b))
            # if the record list has not been initailized
            if len(train_list1) == 0:
                train_list1 = copy.deepcopy(scaned_img)
            # else update the list
            else:
                for i in range(len(train_list1)-1, -1, -1):
                    s = train_list1[i]
                    if s not in scaned_img or ".png" not in s:
                        # pop out images that not exists in any other folders
                        train_list1.pop(i)

        # 2nd scan, scan the gt folder, confirm there exist gt for each train image
        gt_list = listdir(self.path_gt)
        train_list2 = copy.deepcopy(train_list1)
        for s in tqdm(train_list1):
            sn, _ = splitext(s)
            if sn+".npz" not in gt_list: train_list2.remove(s)
        
        # save the scaned result to file
        shuffle(train_list2)
        with open(self.path_index, "w") as f:
            f.write("\n".join(train_list2))
        self.img_list = train_list2


    def add_texture(self, img, bg_color):
        # add a random back ground to the image
        # make sure the returned image only have RGB channels
        h, w, _ = img.shape
        img_rgb = img[:, :, 0:3]
        alpha = img[:, :, 3]
        alpha[alpha == 255] = 0
        alpha = np.expand_dims(alpha, -1).astype(float) / 255
        # generate background and mix it back into image
        dice = np.random.uniform()
        if self.bg and dice < 0.95:
            edge = h if h > w else w
            # change the background color to random value, too
            blank = self.blank_image(edge, edge, background=np.random.randint(bg_color, 245))
            bg = self.texture(blank, sigma=np.random.randint(1, BG_SIGMA), 
                turbulence=np.random.randint(5, 10))[:h, :w, ...]
        else:
            bg = np.ones((h, w, 1)) * 255
                
        
        return (img_rgb * alpha + bg * (1 - alpha)).mean(axis = -1)

    def __len__(self):
        return len(self.img_list)

    def cat_all_pts(self, pts):
        for i in range(len(pts) - 1, -1, -1):
            if len(pts[i]) == 0:
                pts.pop(i)
        if len(pts) > 0:
            return np.concatenate(pts, axis = 0)
        else:
            return pts

    def __getitem__(self, idx):
        # randomly pick up a brush
        DEBUG = False
        if DEBUG:
            start_time = time.time()
            print("log:\tstart data loading")
        brush = random.sample(self.brushes, 1)
        gidx = 1 if self.up_scale else 0
        gsize = self.grid_size[gidx]
        
        # get image path
        img_path = join(self.path_train, brush[0], self.img_list[idx])
        
        # open training image, read as tensor directly
        img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        _, _, c = img.shape
        # this is a trick to allow us to remove white background brought by image rotation later
        if c == 4:
            img[..., 3][img[...,3] == 255] = 254

        # open svg and convert it to points
        svg_path = join(self.path_svg, self.img_list[idx].replace('.png', '.svg'))
        paths, canvas_size = open_svg_flatten(svg_path)
        pts, _  = path_to_pts(paths)
        
        # open ground truths
        gts = self.open_gt(idx)
        if self.approx_udf:
            assert self.up_scale == False
            udf_np, _ = lines_to_udf_fast(pts, canvas_size, 1)
        else:    
            udf_np = None
        end_pt = np.array(gts['end_point'][gidx])
        sharp_pt = np.array(gts['sharp_turn'][gidx])
        t_pt = np.array(gts['T'][gidx])
        x_pt = np.array(gts['X'][gidx])
        star_pt = np.array(gts['star'][gidx])
        usm_np = gts["under_sampled"][gidx]
        edge_maps_np_x = gts["edge_x"][gidx]
        edge_maps_np_y = gts["edge_y"][gidx]
        edge_maps_np = np.stack((edge_maps_np_x, edge_maps_np_y), axis = -1)
        pt_map = gts["pt_map"][gidx]
        skel_np = np.logical_or(edge_maps_np_x, edge_maps_np_y)
        edge_list = [edge_maps_np, pt_map, usm_np, skel_np.astype(int)]

        # prepare the key point list
        pts_all = [end_pt, sharp_pt, t_pt, x_pt, star_pt]
        pts_all = self.cat_all_pts(pts_all)

        pts_junc = [t_pt, x_pt, star_pt]
        pts_junc = self.cat_all_pts(pts_junc)

        pts_list = [pts_all, end_pt, sharp_pt, pts_junc]

        h, w = img.shape[0], img.shape[1]
        assert (h, w) == canvas_size

        if DEBUG:
            end_time = time.time()
            print('log\tloaded with %s seconds'%(end_time - start_time))
            start_time = end_time
            print('log\tstart augment data')
        
        img, pts, pts_list, edge_list = self.transform_pos(img, pts, pts_list, edge_list, canvas_size, gsize, p = 0.5, udf_np = udf_np)
        edge_maps_np, pt_map, usm_np, skel_np, udf_np = edge_list
        edge_maps_x = edge_maps_np[:, :, 0].squeeze()
        edge_maps_y = edge_maps_np[:, :, 1].squeeze()
        xy = np.logical_or(edge_maps_x, edge_maps_y)
        edge_maps_np = edge_maps_x + 2 * edge_maps_y

        # we might get a image without alpha channel
        # so we need to be able to process it without crashing
        if c == 4:
            if int(brush[0]) == 7:
                bg_color = np.random.randint(175, high = BG_COLOR)
            else:
                bg_color = np.random.randint(50, high = BG_COLOR)
            img = self.add_texture(img, bg_color)
        
        if self.jpg:
            dice = np.random.uniform()
            if dice > 0.02:
                img = self.randomJPEGcompression(img)


        # generate UDF
        if len(pts) == 0:
            '''
            it should never goes into this branch!! there might be some bug or something make this happen 
            but I don't know how to resolve, I currently can only make it more robust so that it will not crash
            the training process.
            '''
            print("Warning:\tempty patch is created when reading image %s, pls double check if it is a blank image."%self.img_list[idx])
            udf_np = np.ones((int(self.patch_size / gsize) + 1, int(self.patch_size / gsize) + 1)) * self.dist_clip
        else:
            if self.approx_udf:
                assert udf_np is not None
            else:
                udf_np, _ = lines_to_udf_fast(pts, (int(self.patch_size / gsize), int(self.patch_size / gsize)), gsize)

        # convert usm map to UDF, everything in UDF!
        udf_usm_np = cv2.distanceTransform(1 - usm_np.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE)

        # get traning mask
        stroke_mask_np = udf_np <= self.dist_clip
        edge_maps_xy = np.logical_or(edge_maps_x, edge_maps_y)
        edge_mask_np = cv2.dilate(edge_maps_xy.astype(np.uint8), self.kernel, iterations = 1).astype(bool)
        # pt_mask1_np = cv2.dilate(edge_maps_xy.astype(np.uint8), self.kernel, iterations = 2).astype(bool)
        pt_mask1_np = (udf_np <= self.dist_pt)[:-1, :-1]
        usm_mask_np = udf_usm_np <= self.dist_pt * 1.5
        if DEBUG:
            end_time = time.time()
            print('log\taugmented with %s seconds'%(end_time - start_time))
            start_time = end_time
            print('log\tconvert to tensor')

        # insert skeleton
        if self.insert_skel and self.no_rotation:
            udf_np[np.where(skel_np)] = 0

        # all to tensor
        img = self.img_to_tensor((img / 255).copy()).float()
        
        # normalize all UDFs
        udf_np = (udf_np.clip(0, self.dist_clip) / self.dist_clip).clip(0, 1)
        udf_usm_np = (udf_usm_np.clip(0, self.dist_pt * 1.5) / self.dist_pt * 1.5).clip(0, 1)
        dist = self.dist_pt
        
        udf = self.img_to_tensor(udf_np.copy()).float()
        topo_mask = self.img_to_tensor(stroke_mask_np.copy()).bool() # for topo UDF
        edge_mask = self.img_to_tensor(edge_mask_np.copy()).bool() # for edge flags
        pt_mask1 = self.img_to_tensor(pt_mask1_np.copy()).float()
        usm_mask = self.img_to_tensor(usm_mask_np.copy()).bool() # for USM maps

        edge_map = self.img_to_tensor(edge_maps_np.copy()).float() # edge map
        pt_map = self.img_to_tensor(pt_map.copy()).float() # point map
        udf_usm = self.img_to_tensor(udf_usm_np.copy()).float() # USM map
        skel = self.img_to_tensor(skel_np.copy()).float() 
        
        size_real = (int(self.patch_size / gsize), int(self.patch_size / gsize))
        udf_all_pt, gt_all_pt, coord_all_pt, mask_all_pt  = self.pt_to_tensor(pts_list[0], size_real,
            gsize, mask_clip = dist)
        udf_end_pt, gt_end_pt, coord_end_pt, mask_end_pt = self.pt_to_tensor(pts_list[1], size_real,
            gsize, mask_clip = dist)
        udf_sharp_pt, gt_sharp_pt, coord_sharp_pt, mask_sharp_pt = self.pt_to_tensor(pts_list[2], size_real, 
            gsize, mask_clip = dist)
        udf_junc_pt, gt_junc_pt, coord_junc_pt, mask_junc_pt = self.pt_to_tensor(pts_list[3], size_real,
            gsize, mask_clip = dist)

        # apply the pixel transform
        if self.transform_pixel is not None and self.jpg == False and self.bg == False:
            img = self.transform_pixel(img)

        if DEBUG:
            end_time = time.time()
            print('log\tconverted with %s seconds'%(end_time - start_time))

        res = {}
        res['img'] = img
        res['udfs'] = [udf, udf_all_pt, udf_end_pt, udf_sharp_pt, udf_junc_pt]
        res['udf_masks'] = [topo_mask, pt_mask1, mask_all_pt, mask_end_pt, mask_sharp_pt, mask_junc_pt]
        res['udf_gts'] = [gt_all_pt, gt_end_pt, gt_sharp_pt, gt_junc_pt]

        res['ndc_gts'] = [edge_map, pt_map, udf_usm, skel]
        res['ndc_mask'] = [edge_mask, usm_mask]
        res['coords'] = [coord_all_pt, coord_end_pt, coord_sharp_pt, coord_junc_pt]
        return res

    def img_to_tensor(self, img_np, clip = -1):
        # we don't nee to normalize the input but 
        # a value clipping is still necessary
        if clip > 0:
            img_np = img_np.clip(0, clip) # this is unsigned distance so no negative values
        transforms = T.Compose(
            [
                T.ToTensor()
            ]
        )
        return transforms(img_np).to(self.device)

    def pt_to_tensor(self, keypt, canvas_size, gsize, mask_clip = -1):
        udf_pt, gt, coord = self.pt_to_udf(keypt, canvas_size, self.dist_pt, gsize)
        if (udf_pt == 0).all():
            mask_pt = np.zeros_like(udf_pt)
        elif mask_clip > 0:
            mask_pt = udf_pt < mask_clip
        else:
            mask_clip = 5
            mask_pt = udf_pt < mask_clip
        udf_pt = udf_pt.clip(0, mask_clip) / mask_clip
        udf_pt = torch.FloatTensor(np.expand_dims(udf_pt, axis = -1).copy()).permute(2, 0, 1)
        udf_pt = udf_pt.to(self.device)
        gt = torch.Tensor(np.expand_dims(gt, axis = -1).copy()).permute(2, 0, 1)
        gt = gt.to(self.device)
        coord = torch.FloatTensor(coord.copy()).permute(2, 0, 1)
        coord = coord.to(self.device)
        mask_pt = torch.Tensor(np.expand_dims(mask_pt, axis = -1).copy()).permute(2, 0, 1)
        mask_pt = mask_pt.to(self.device).to(bool)
        return udf_pt, gt, coord, mask_pt

    def open_gt(self, idx):
        gt_path = join(self.path_gt, self.img_list[idx].replace(".png", ".npz"))
        key_pt = np.load(gt_path.strip("\n"), allow_pickle = True)
        return key_pt

    def transform_pos(self, img, pts_path, pts_list, edge_list, canvas_size, gsize, p = 0.5, ang = 30, udf_np = None):
        edge_maps_np, pt_map, usm_np, skel_np = edge_list
        
        ''' transpose'''
        dice = np.random.uniform()
        debug = False
        if dice < p or debug:
            h, w = canvas_size
            canvas_size = [w, h]
            if len(img.shape) == 2:
                img = img.transpose((1, 0))
            else:
                img = img.transpose((1, 0, 2))
            
            if udf_np is not None:
                udf_np = udf_np.transpose((1, 0))

            skel_np = skel_np.transpose((1, 0))
            usm_np = usm_np.transpose((1, 0))

            # transpose edge map
            edge_maps_np = edge_maps_np.transpose((1, 0, 2))
            edge_maps_np[..., [0, 1]] = edge_maps_np[..., [1, 0]]
            
            # transpose points
            pts_path[..., [0, 1]] = pts_path[..., [1, 0]]
            pt_map = pt_map.transpose((1, 0, 2))
            pt_map[..., [0, 1]] = pt_map[..., [1, 0]]
            
            for i in range(len(pts_list)):
                if len(pts_list[i]) > 0:
                    pts_list[i][..., [0, 1]] = pts_list[i][..., [1, 0]]

        def save_for_debug(canvas_size, crop = False):
            canvas_h, canvas_w = canvas_size
            if crop:
                lines_to_svg(pts_path, self.patch_size, self.patch_size, 'temp.svg', indexing = 'xy')
                canvas_size = (self.patch_size, self.patch_size)
            else:
                lines_to_svg(pts_path, canvas_w, canvas_h, 'temp.svg', indexing = 'xy')
            from PIL import Image
            Image.fromarray(img).convert('RGB').save('temp.png')
            skel = svg_to_numpy('temp.svg')
            if crop:
                pts_list[0][pts_list[0] >= 256] = 255
            self.show_pt(skel, pts_list[0], canvas_size)

        '''vertical flipping (along x-axis), comment out this block if you don't need'''
        dice = np.random.uniform()# let's roll the dice!

        if dice < p or debug:
            img = self.flip_img(img, 'x')
            if udf_np is not None:
                udf_np = self.flip_img(udf_np, 'x')
            skel_np = self.flip_img(skel_np, 'x')
            pts_path = flip_path(pts_path, 'x', canvas_size)
            edge_maps_np = flip_edge(edge_maps_np, axis = 'x')
            pt_map = flip_keypt_map(pt_map, 'x', canvas_size, 'xy')
            usm_np = np.flip(usm_np, axis = 0)
            for i in range(len(pts_list)):
                pts_list[i] = flip_path(pts_list[i], 'x', canvas_size, index = 'xy')

        '''horizontal flipping (along x-axis), comment out this block if you don't need'''
        dice = np.random.uniform()# let's roll the dice! x 2
        
        if dice < p or debug:
            img = self.flip_img(img, 'y')
            if udf_np is not None:
                udf_np = self.flip_img(udf_np, 'y')
            skel_np = self.flip_img(skel_np, 'y')
            pts_path = flip_path(pts_path, 'y', canvas_size)
            for i in range(len(pts_list)):
                pts_list[i] = flip_path(pts_list[i], 'y', canvas_size, index = 'xy')
            edge_maps_np = flip_edge(edge_maps_np, axis = 'y')
            pt_map = flip_keypt_map(pt_map, 'y', canvas_size, 'xy')
            usm_np = np.flip(usm_np, axis = 1)

        '''rotation, comment out this block if you don't need'''
        dice = np.random.uniform()# let's roll the dice! x 3
        angd = np.random.randint(-ang, ang+1)# angle in degree

        if dice < p and self.no_rotation == False or debug:
            img = self.rotate_img(img, -angd)
            if udf_np is not None:
                udf_np = self.rotate_img(udf_np, -angd)
            usm_np = self.rotate_img((~usm_np).astype(int) *255, -angd)
            usm_np = usm_np < 200

            skel_np = self.rotate_img(skel_np.astype(float), -angd)
            pts_path = rotate_pts(pts_path, canvas_size, angd, index = 'xy')
            for i in range(len(pts_list)):
                pts_list[i] = rotate_pts(pts_list[i], canvas_size, angd, index = 'xy')
            '''
            DON'T USE edge map comes out from this branch for training!
            it is no longer edge maps after any rotation
            '''
            edge_maps_x = self.rotate_img((~edge_maps_np[..., 0]).astype(int) *255, -angd)
            edge_maps_x = edge_maps_x < 200
            edge_maps_y = self.rotate_img((~edge_maps_np[..., 1]).astype(int) *255, -angd)
            edge_maps_y = edge_maps_y < 200
            edge_maps_np = np.stack((edge_maps_x, edge_maps_y), axis =-1)

        '''cropping, we always need to crop the image and point, so DON'T comment out this block!'''
        bbox, _ = random_bbox(canvas_size, gsize, self.patch_size, edge_maps_np, 
            pts_list[-1], udf_mode = True if gsize == 0.5 else False)
        
        assert int(bbox[0]*gsize) == bbox[0]*gsize
        assert int(bbox[1]*gsize) == bbox[1]*gsize
        assert int(bbox[2]*gsize) == bbox[2]*gsize
        assert int(bbox[3]*gsize) == bbox[3]*gsize

        bbox_real = (int(bbox[0]*gsize), int(bbox[1]*gsize), int(bbox[2]*gsize), int(bbox[3]*gsize))
        pts_path, new_end_pt = crop_path(pts_path, bbox_real, canvas_size, mode = 'path')

        img = crop_edge(img, bbox_real)
        if udf_np is not None:
            udf_np = crop_edge(udf_np, _)
        skel_np = crop_edge(skel_np, bbox)
        edge_maps_np = crop_edge(edge_maps_np, bbox)
        pt_map = crop_keypt_map(pt_map, bbox, gsize)
        usm_np = crop_edge(usm_np, bbox)
        
        if len(new_end_pt) > 0:
            if len(pts_list[1]) == 0:
                pts_list[1] = new_end_pt
            else:
                pts_list[1] = np.concatenate((pts_list[1], new_end_pt), axis = 0)
            pts_list[0] = np.concatenate((pts_list[0], new_end_pt), axis = 0)
        for i in range(len(pts_list)):
            pts_list[i] = crop_path(pts_list[i], bbox_real, canvas_size)

        '''debug block'''
        # import pdb
        # pdb.set_trace()
        # save_for_debug((256, 256))

        return img, pts_path, pts_list, [edge_maps_np, pt_map, usm_np, skel_np, udf_np]

    def flip_img(self, img, axis):
        if axis == 'x':
            axis = 0
        elif axis == 'y':
            axis = 1
        else:
            raise ValueError("Invalid axis format!")
        flipped = np.flip(img, axis = axis)
        return flipped

    def rotate_img(self, img, ang):
        # https://stackoverflow.com/questions/53171057/numpy-matrix-rotation-for-any-degrees
        rotated = rotate(img, ang, reshape = False, cval = 255.0, order = 3, prefilter = False)
        return rotated

    def auto_bright_contrast(self, img, clip_hist_percent = 1):
        # thanks to https://stackoverflow.com/questions/56905592/automatic-contrast-and-brightness-adjustment-of-a-color-photo-of-a-sheet-of-pape
        # Calculate grayscale histogram
        hist = cv2.calcHist([img], [0], None, [256], [0, 256])
        hist_size = len(hist)
        # Calculate cumulative distribution from the histogram
        accumulator = []
        accumulator.append(float(hist[0]))
        for index in range(1, hist_size):
            accumulator.append(accumulator[index -1] + float(hist[index]))
        
        # Locate points to clip
        maximum = accumulator[-1]
        clip_hist_percent *= (maximum/100.0)
        # clip_hist_percent /= 2.0
        
        # Locate left cut
        minimum_gray = 0
        while accumulator[minimum_gray] < clip_hist_percent:
            minimum_gray += 1
        
        # we don't need to cut the right side
        # Locate right cut
        maximum_gray = hist_size -1
        # while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        #     maximum_gray -= 1
        
        # Calculate alpha and beta values
        alpha = 255 / (maximum_gray - minimum_gray)
        beta = -minimum_gray * alpha
        img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
        return img

    def find_new_boundary_pt(self, skeleton, bbox):
        '''
        Find new endpoint created by cropping, 
            return the coordinates of them under the image coord system
        '''
        skel = skeleton.copy()
        h_min, w_min, h_max, w_max = bbox
        mask = np.ones(skel.shape).astype(bool)
        mask[h_min, w_min : w_max] = False # top row
        mask[h_max - 1, w_min : w_max] = False # bottom row
        mask[h_min : h_max, w_min] = False # left column
        mask[h_min : h_max, w_max - 1] = False # right column
        skel[mask] = 255
        return np.array(np.where(skel == 0)).T # (h, w)

    def remove_out_pt(self, pt, bbox, inplace = True):
        '''
        Inplace function, remove points outside of the given bbox
        '''
        y_min, x_min, y_max, x_max = bbox
        mask1 = np.logical_and(pt[:, 0] >= y_min, pt[:, 0] < y_max)
        mask2 = np.logical_and(pt[:, 1] >= x_min, pt[:, 1] < x_max)
        mask =  np.logical_and(mask1, mask2)
        if inplace:
            pt = pt[mask, :]
        else:
            return pt[mask, :]

    def pt_to_udf(self, pt, canvas_size, dist_clip, gsize):
        '''
        the canvas size could be bbox, or the real image size
        return the unsigned distance field for training
        '''
        h, w = canvas_size # y, x
        # init coord
        xs = np.arange(0, w) + 0.5
        ys = np.arange(0, h) + 0.5
        yy, xx = np.meshgrid(ys, xs, indexing = 'ij')
        coord = np.stack((yy, xx), axis = -1)
        # coord = None
        if len(pt) == 0:
            udf = np.ones((h, w)) * dist_clip
            gt = np.zeros((h, w), dtype = bool)
        else:
            # create gt for keypoint coordinate
            pt = pt / gsize
            pt_i = pt.copy().astype(int)
            pt_c = pt.copy()
            pt_i[pt_i[:,0] >= w, 0] = w - 1
            pt_i[pt_i[:,1] >= h, 1] = h - 1
            pt_i[:, [0, 1]] = pt_i[:, [1, 0]]
            pt_idx = tuple(pt_i.T.astype(int))
            gt = np.zeros((h, w), dtype = bool)
            gt[pt_idx] = True
            # update pt map for ndc
            pt_c[:, [0, 1]] = pt_c[:, [1, 0]]
            for idx in range(len(pt_i)):
                i, j = pt_i[idx]
                coord[i, j, ...] = pt_c[idx]
            '''
                we don't care the correct scale of UDF when extracting the keypoint location
                by our current local maximum algorithm, so it is not necessary to re-scale 
                the UDF.
            '''
            udf = cv2.distanceTransform(1 - gt.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE) * gsize
            coord = coord * gsize
        return udf, gt, coord

    def get_rotate_matrix(self, ang):
        '''assume ang are degrees'''
        # convert ang from degree to radian
        assert abs(ang) <= 360
        ang = (ang / 360 ) * np.pi * 2
        return np.array([[np.cos(ang), -np.sin(ang)],[np.sin(ang), np.cos(ang)]])

    def crop_pt(self, pt, bbox):
        '''
        Remove all points outside of the bbox
            and also translate the point coord to bbox coord system
        '''
        y_min, x_min, _, _ = bbox
        res = self.remove_out_pt(pt, bbox, False)
        return res - np.array([[y_min, x_min]])

    # for debug
    def show_img(self, img):
        from PIL import Image
        img = img.detach().cpu().permute(1,2,0).squeeze().numpy()
        img = (img * 255).clip(0, 255).astype(np.uint8)
        Image.fromarray(img).show()

    def show_pt(self, skeleton, pts, canvas_size):
        # pts[pts >= 256] = 255
        h, w = canvas_size
        pts = pts.copy()
        pts[:,[0, 1]] = pts[:, [1, 0]]
        pts[pts[:,0] >= w, 0] = w - 1
        pts[pts[:,1] >= h, 1] = h - 1
        red = np.array([255, 0 , 0])
        from PIL import Image
        res = np.stack((skeleton, skeleton, skeleton), axis = -1)
        res = cv2.resize(res, (w, h), interpolation = cv2.INTER_NEAREST)
        res[tuple((pts + 0.5).T.astype(int))] = red
        Image.fromarray(res).show()


    # let's generate paper texture by opencv
    # https://stackoverflow.com/questions/51646185/how-to-generate-a-paper-like-background-with-opencv
    def blank_image(self, width=1024, height=1024, background=BG_COLOR):
        """
        It creates a blank image of the given background color
        """
        img = np.full((height, width, MONOCHROME), background, np.uint8)
        return img

    def add_noise(self, img, sigma=BG_SIGMA):
        """
        Adds noise to the existing image
        """
        width, height, ch = img.shape
        n = self.noise(width, height, sigma=sigma)
        img = img + n
        return img.clip(0, 255)

    def noise(self, width, height, ratio=1, sigma=BG_SIGMA):
        """
        The function generates an image, filled with gaussian nose. If ratio parameter is specified,
        noise will be generated for a lesser image and then it will be upscaled to the original size.
        In that case noise will generate larger square patterns. To avoid multiple lines, the upscale
        uses interpolation.

        :param ratio: the size of generated noise "pixels"
        :param sigma: defines bounds of noise fluctuations
        """
        mean = 0
        # assert width % ratio == 0, "Can't scale image with of size {} and ratio {}".format(width, ratio)
        # assert height % ratio == 0, "Can't scale image with of size {} and ratio {}".format(height, ratio)

        h = int(height / ratio)
        h = 1 if h == 0 else h
        w = int(width / ratio)
        w = 1 if w == 0 else w

        result = np.random.normal(mean, sigma, (w, h, MONOCHROME))
        if ratio >= 1:
            result = cv2.resize(result, dsize=(width, height), interpolation=cv2.INTER_LINEAR)
        return result.reshape((width, height, MONOCHROME))

    def texture(self, image, sigma=BG_SIGMA, turbulence=10):
        """
        Consequently applies noise patterns to the original image from big to small.

        sigma: defines bounds of noise fluctuations
        turbulence: defines how quickly big patterns will be replaced with the small ones. The lower
        value - the more iterations will be performed during texture generation.
        """
        result = image.astype(float)
        cols, rows, ch = image.shape
        edge = cols if cols < rows else rows
        k = edge ** (1 / (turbulence))
        i = 0
        ratio = edge
        while ratio > 1:
            ratio = int(edge / (k ** i))
            ratio = 1 if ratio == 0 else ratio
            result += self.noise(cols, rows, ratio, sigma=sigma)
            i += 1
        cut = np.clip(result, 0, 255)
        return cut.astype(np.uint8)