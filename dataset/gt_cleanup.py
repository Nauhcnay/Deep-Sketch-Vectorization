# remove udf has no help!
# T_T
from tqdm import tqdm
from os.path import join
import numpy as np
import os

path_to_gt_old = "../data/full/gt"
path_to_gt_new = "../data/full/gt_"

for gt in tqdm(os.listdir(path_to_gt_old)):
    if ".npz" not in gt: continue
    gt_np = np.load(join(path_to_gt_old, gt), allow_pickle = True)
    edge_map_x = gt_np["edge_x"]
    edge_map_y = gt_np["edge_y"]
    pt_map = gt_np['pt_map']
    usm_np = gt_np["under_sampled"]
    end_pt = gt_np["end_point"]
    shrp_pt = gt_np["sharp_turn"]
    t_pt = gt_np["T"]
    x_pt = gt_np["X"]
    star_pt = gt_np["star"]
    np.savez_compressed(join(path_to_gt_new, gt), edge_x = edge_map_x, edge_y = edge_map_y, pt_map = pt_map, 
        under_sampled = usm_np, end_point = end_pt, sharp_turn = shrp_pt, T = t_pt, X = x_pt, star = star_pt)