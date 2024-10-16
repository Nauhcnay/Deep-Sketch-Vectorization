import numpy as np
import matplotlib as plt
import cv2
from dataset.preprocess import get_coords_from_heatmap

def compute_pt_pr_all(pres, gts):
    precision = []
    recall = []
    assert len(pres) == len(gts)
    for i in range(len(pres)):
        assert len(pres[i]) == len(gts[i])
        p_batch = []
        r_batch = []
        for j in range(len(pres[i])):
            p, r = compute_pt_pr(pres[i][j], gts[i][j])
            p_batch.append(p)
            r_batch.append(r)
        precision.append(np.array(p_batch).mean())
        recall.append(np.array(r_batch).mean())
    return np.array(precision), np.array(recall)

def compute_pt_pr(pre, gt):
    pre = pre.squeeze()
    gt = gt.squeeze()
    cd_pre = to_img_coord(255 - (pre * 255))
    grid_pre = np.zeros(pre.shape, dtype = bool)
    if len(cd_pre) > 0:
        cd_pre[:, [0, 1]] = cd_pre[:, [1, 0]]
        grid_pre[tuple(cd_pre.T)] = True

    cd_gt = to_img_coord(255 - (gt * 255))
    cd_gt[:, [0, 1]] = cd_gt[:, [1, 0]]
    grid_gt = np.zeros(gt.shape, dtype = bool)
    if len(cd_gt) > 0:
        cd_gt[:, [0, 1]] = cd_gt[:, [1, 0]]
        grid_gt[tuple(cd_gt.T)] = True
    
    # compute precision
    if grid_gt.sum() > 0:
        p = np.logical_and(grid_pre, grid_gt).sum() / grid_gt.sum()
    else:
        p = 1.0

    # compute recall
    grid_neg_gt = np.logical_not(grid_gt)
    if grid_neg_gt.sum() > 0:
        r = np.logical_and(np.logical_not(grid_pre), grid_neg_gt).sum() / grid_neg_gt.sum()
    else:
        r = 1.0

    return p, r

def to_img_coord(hm, dist = 2, pt_num = 50, thr_rel = 50):
    hm = hm.clip(0, 255)
    ### this output is xy ###
    cd = get_coords_from_heatmap(hm, dist, pt_num, thr_rel)
    return cd    

def plot_hm_coord(sketch, hm, coord):
    plt.xticks([])
    plt.yticks([])
    plt.imshow(sketch, cmap='gray')
    plt.imshow(hm, cmap='hot_r', alpha=0.8)
    if len(coord) > 0: 
        plt.scatter(coord[:, 0], coord[:, 1], marker='+', c='lime', label='end', linewidths=0.15)

def udf_filter(udf, thr = 0.99):
    udf = udf.clip(0, 1)
    udf_mask = udf < thr
    res_mask = np.zeros(udf_mask.shape).astype(bool)
    _, regs = cv2.connectedComponents(udf_mask.astype(np.uint8), connectivity = 4)
    for r in np.unique(regs):
        if r == 0: continue
        m = regs == r
        if udf[m].min() > 0.1:
            res_mask[m] = True
    udf[res_mask] = 1
    return udf

def plot_udf_coord_numpy(sketch, udfs, udf_all_pt = None, alpha = 0.3, pt_num = 1000):
    '''
    Given,
        sketch, a numpy array as the input skech
        udfs, a list of numpy arrays as the input UDF(s)
    Return,
        a color image as numpy array that blends the sketch as colorized UDF and the detected keypoint from the UDF
    '''
    assert type(udfs) == list
    assert len(udfs) <= 6
    res = sketch.copy()
    # blend sketch with udf blend
    if udf_all_pt is not None:
        hm_all_pt = udf_to_hm(udf_all_pt)
        res = blend_heatmaps(res, hm_all_pt)
    # draw keypoint on the sketch
    res, keypts_dict = extract_keypts(udfs, pt_num, res)
    return res, keypts_dict

def vis_pt_single(sketch, udf, keypt_name = ['none'], alpha = 0.3, pt_num = 1000):
    res = sketch.copy()
    # blend sketch with udf blend
    hm = udf_to_hm(udf)
    res = blend_heatmaps(res, hm)
    # draw keypoint on the sketch
    res, keypts_dict = extract_keypts([udf], pt_num, res, keypt_name)
    return res, keypts_dict

def udf_to_hm(udf):
    return 255 - udf.clip(0, 1).squeeze() * 255

def blend_heatmaps(img, hm, alpha = 0.3):
    # visualize the heatmap
    hm = cv2.applyColorMap(hm.astype(np.uint8), cv2.COLORMAP_HOT)
    hm = cv2.cvtColor(hm, cv2.COLOR_BGR2RGB)
    # blend sketch and heatmap
    res = (img * (1 - alpha) + hm * alpha)
    return res.astype(np.uint8)

def extract_keypts(udfs, pt_numm, img = None, key_list = ["end_point", "sharp_turn", "junc"], 
        thr_list = [0.8, 0.8, 0.5]):
    assert len(udfs) <= len(key_list)
    # init colors
    green = np.array([0, 255, 0]) # end points, 0
    red = np.array([255, 0, 0]) # sharp turn, 1
    blue = np.array([0, 0, 255]) # T junctions, 2
    cyan = np.array([255, 255, 0]) # X junctions, 3
    yellow = np.array([0, 255, 255]) # star junctions, 4
    colors = [green, red, blue, cyan, yellow]
    # extract keypoint and overlay it on given image if possible
    keypts_dict = {}
    for i in range(len(udfs)):
        udf = udfs[i].clip(0, 1)
        # detect keypoint coordinates
        cd = to_img_coord((255 - udf * 255), pt_num = pt_numm, thr_rel = thr_list[i])
        # draw on the sketch
        if img is not None:
            img = draw_cross(img, cd, colors[i])
        keypts_dict[key_list[i]] = cd
    return img, keypts_dict

def draw_cross(res, cd, color):
    if len(cd) > 0:
        h, w = res.shape[0], res.shape[1]
        cd = cd.copy()
        # this function need hw format!
        cd[:, (0, 1)] = cd[:, (1, 0)]
        cd_center = tuple(cd.T)
        res[cd_center] = color
    return res

def vis_grad_batch(pre, gt, iter_counter, save_path):
    assert len(pre) == len(gt)
    pre = pre.permute(0, 2, 3, 1).detach().cpu().numpy()
    gt = gt.permute(0, 2, 3, 1).detach().cpu().numpy()
    for i in range(len(pre)):
        svg_path = join(save_path, "train_%d_%d_grad_pre.svg"%(iter_counter, i))
        vis_grad(pre[i], svg_path)
        svg_path = join(save_path, "train_%d_%d_grad_gt.svg"%(iter_counter, i))
        vis_grad(gt[i], svg_path)

def vis_batch(img, pre, gt, gsize, mask = None, blend = None, img_num = 8, add_all_pts = False):
    _, _, h, w = img.shape
    size_real = (int(w / gsize), int(h / gsize))
    assert len(pre) == len(gt)
    # assert len(img) == len(pre)
    if len(pre) > 1:
        usms_pre = pre[-1]
        usms_gt = gt[-1]
        pre = pre[:-1]
        gt = gt[:-1]

    imgs_np = img.detach().cpu().numpy()
    # tensor to numpy for all
    # for i in range(len(pre)):
    #     pre[i] = pre[i].squeeze(1).detach().cpu().numpy()
    #     gt[i] = gt[i].squeeze(1).detach().cpu().numpy()
    # select the first 8 samples to visualize
    # for each sample in this batch
    res = []
    keypts_pre = {}
    keypts_gt = {}
    for i in range(img_num if img_num <= (len(img)) else len(img)):
        # get img as numpy array
        img_np = imgs_np[i].squeeze()
        img_np = img_np * 255
        img_np = cv2.resize(img_np, size_real, interpolation = cv2.INTER_AREA)
        img_np = np.expand_dims(img_np, axis = -1).repeat(3, axis = -1)
        img_org_np = img_np.copy()
        if mask is not None:
            mask_np = mask[i].squeeze().astype(float)
            mask_np[mask_np == 0] = 0.5
            mask_np = mask_np * 255
            mask_np = np.expand_dims(mask_np, axis = -1).repeat(3, axis = -1)
            img_np = img_np * 0.7 + mask_np * 0.3

        # get input udfs
        udfs_pre = []
        udfs_gt = []
        # for each type udf 
        for j in range(len(pre)):
            udfs_pre.append(pre[j][i].squeeze())
            udfs_gt.append(gt[j][i].squeeze())
        pre_udf = blend[0][i].squeeze() # topology or all pt prediction
        gt_udf = blend[1][i].squeeze() # topology or all pt ground truth
        if add_all_pts:
            usm_pre = usms_pre[i].squeeze()
            usm_gt = usms_gt[i].squeeze()
            gt_np, keypt_gt = plot_udf_coord_numpy(img_org_np, udfs_gt, usm_gt)
            pre_np, keypt_pre = plot_udf_coord_numpy(img_org_np, udfs_pre, usm_pre)
            all_pre = pre_udf
            all_np, _ = plot_udf_coord_numpy(img_org_np, [all_pre], all_pre)
            img_np = add_title(img_np, "Input", org = (0, 20), fontScale = 0.5, thickness = 1)
            gt_np = add_title(gt_np, "GT Keypoint + GT USM", org = (0, 20), fontScale = 0.5, thickness = 1)
            all_np = add_title(all_np, "Pre keypt + Heatmap", org = (0, 20), fontScale = 0.5, thickness = 1)
            pre_np = add_title(pre_np, "Pre keypt/cls + Pre USM", org = (0, 20), fontScale = 0.5, thickness = 1)
            res.append(np.concatenate((img_np, gt_np, all_np, pre_np), axis = 0))    
        else:
            gt_np, keypt_gt = plot_udf_coord_numpy(img_org_np, udfs_gt, gt_udf)
            pre_np, keypt_pre = plot_udf_coord_numpy(img_org_np, udfs_pre, pre_udf)
            img_np = add_title(img_np, "Input", org = (0, 20), fontScale = 0.5, thickness = 1)
            gt_np = add_title(gt_np, "GT UDF", org = (0, 20), fontScale = 0.5, thickness = 1)
            pre_np = add_title(pre_np, "Pre UDF", org = (0, 20), fontScale = 0.5, thickness = 1)
            res.append(np.concatenate((img_np, gt_np, pre_np), axis = 0))
        for key in keypt_pre:
            if key not in keypts_pre:
                keypts_pre[key] = [keypt_pre[key]]
            else:
                keypts_pre[key].append(keypt_pre[key])
            if key not in keypts_gt:
                keypts_gt[key] = [keypt_gt[key]]
            else:
                keypts_gt[key].append(keypt_gt[key])
    res = np.concatenate(res, axis = 1)
    return res, keypts_pre, keypts_gt

def add_title(img, text, org = (0, 100), fontScale = 2, color = (0, 0, 255), thickness = 1):
    font = cv2.FONT_HERSHEY_SIMPLEX
    img = cv2.putText(img, text, org, font, fontScale, color, thickness, cv2.LINE_AA) 
    return img
