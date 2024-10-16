import torch
import numpy as np
from scipy.spatial import KDTree
from torch.nn.functional import normalize

def dist_loss(pre, gt, mask, pt_mode = False, dist_mode = 'l1', loss_mode = 'idx'):
    # compute loss map
    if dist_mode == 'l1':
        weights = [0.1, 0.3, 1] if loss_mode == 'idx' else [0.01, 1, 10]
        loss_map = torch.abs(gt - pre)
    elif dist_mode == 'l2':
        weights = [0.1, 0.5, 1] if loss_mode == 'idx' else [0.1, 5, 10]
        loss_map = torch.square(gt - pre)
    else:
        raise ValueError("distance type could only be 'l1' or 'l2'.")
    # prepear loss mask
    '''
        mask1 should always contains mask2, mask2 gives more accurate
        location where the ground truth label is
    '''
    mask1, mask2 = mask
    assert mask1.shape == mask2.shape
    mask1_out = torch.logical_not(mask1)
    mask12 = torch.logical_xor(mask1, mask2)
    mask1_out = mask1_out.to(bool)
    mask12 = mask12.to(bool)
    mask2 = mask2.to(bool)
    mask1 = mask1.to(bool)
    if pt_mode:
        masks = [mask1_out, mask12, mask2]
    else: 
        weights = [0.1, 1] if loss_mode == 'idx' else [0.01, 1]
        masks = [mask1_out, mask1]
    # compute loss
    return apply_mask_loss(loss_map, masks, weights, loss_mode)

def apply_mask_loss(loss_map, masks, weights, loss_mode = 'idx'):
    mask_num = 0
    loss = 0
    if loss_mode == 'idx':
        for i in range(len(masks)):
            if masks[i].sum() > 0:
                loss = loss + loss_map[masks[i]].mean() * weights[i]
                mask_num += 1
        return loss / min(1, mask_num)
    elif loss_mode == 'multi':
        mask = 0
        for i in range(len(masks)):
            mask = mask + masks[i] * weights[i]
        return (loss_map * mask).mean()
    else:
        raise ValueError("invailid loss mode, only 'idx' and 'multi' is supported")

def chamfer_distance_vec(pts_res, pts_gt):
    pts_res = np.array([[pt.real, pt.imag] for pt in pts_res])
    pts_gt = np.array([[pt.real, pt.imag] for pt in pts_gt])
    
    KDtree_for_res = KDTree(pts_gt)
    KDtree_for_gt = KDTree(pts_res)
    
    # compute chamfer distance from results to gt
    dd_res, _ = KDtree_for_res.query(pts_res, k=1, workers = -1)
    dist_res_to_gt = dd_res.sum()

    # compute chamfer distance from gt to results
    dd_res, _ = KDtree_for_gt.query(pts_gt, k=1, workers = -1)
    dist_gt_to_res = dd_res.sum()    
    
    return dist_gt_to_res + dist_res_to_gt
    