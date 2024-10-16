import torch
from torch import Tensor
from torch.nn.functional import mse_loss, binary_cross_entropy_with_logits, conv2d, softmax, relu, logsigmoid
from utils.focal_loss import softmax_focal_loss, sigmoid_loss, apply_weights
from torchvision.transforms import GaussianBlur as G
from torch import sigmoid
from utils.ndc_tools import pre_to_map


def loss_sparsity(pre, gt):
    device = pre.device

    weights = torch.ones((1, 1, 3, 3), requires_grad = False)
    weights[:, :, 1, 1] = 0 # we should not consider the center of the kernel
    weights = weights.to(device)
    b, c, h, w = pre.shape # record the shape
    pre = pre.permute(0, 2, 3, 1).reshape(-1, 4)
    gt = gt.long()
    
    pre = softmax(pre, dim = -1)
    pre_none = pre[torch.arange(pre.size(0)), 0].reshape((b, h, w, 1)).permute(0, 3, 2, 1)
    gt_none = (gt == 0).float() / 2

    # get prediction mask 
    mask_has_edge = (pre.argmax(dim = -1) != 0).reshape((b, h, w, 1)).permute(0, 3, 2, 1).bool()
    
    # compute the sparsity value
    pre_none = conv2d(pre_none, weights, padding = 'same')
    gt_none = conv2d(gt_none, weights, padding = 'same')

    if mask_has_edge.sum() == 0:
        loss = 0
    else:
        loss = relu(gt_none[mask_has_edge] - pre_none[mask_has_edge], inplace = True).mean()
    return loss

def loss_edge(edge_maps_pre, edge_maps, edge_mask, focal = True, alpha = False, review = False, review_all = False):
    '''
    Given:
        edge_maps_pre:  A float (b, 4, h, w) tensor, the edge flag prediction output from UDC network
        edge_maps:      A boolean (b, 2, h, w) tensor, the edge flag ground truth
        focal:          A boolean variable, compute focal loss based on the recross entropy loss if it is true
        alpha:          A boolean variable, apply adaptive loss weight per edge flag type
    Return:
        loss:           A float tensor that contains the loss value
        accs:           float numbers about the prediction accuracy, they should always between 0 to 1
    '''
    
    c = edge_maps.shape[1] # get the length of label channel
    assert c == 1 or c ==2
    mask_ndc = edge_mask.flatten()

    # compute CE loss based focal loss
    if c == 1:
        # construct edge map GT
        edge_maps_gt_xy = edge_maps.permute(0, 2, 3, 1).flatten().long()
        edge_maps_gt_z = edge_maps_gt_xy != 0

        # get predition labels
        edge_maps_pre = edge_maps_pre.permute(0, 2, 3, 1).reshape(-1, 5)
        edge_maps_pre_xy = edge_maps_pre[..., :-1]
        edge_maps_pre_z = edge_maps_pre[..., 4]

        loss_ce = softmax_focal_loss(edge_maps_pre_xy, edge_maps_gt_xy, mask = mask_ndc, focal = focal, alpha = alpha, 
            gamma = 5, review = review, review_all = review_all)
        acc_x_pos, acc_x_neg, acc_y_pos, acc_y_neg, acc_a_pos, acc_a_neg = get_prediction_acc(edge_maps_pre_xy, edge_maps_gt_xy)
        weights = [0.02, 1, 1]
        loss_z = sigmoid_loss(edge_maps_pre_z, edge_maps_gt_z, weights = weights, mask = mask_ndc, 
            review = review, review_all = review_all, focal = focal)
        loss = loss_ce
        
    # BCE loss based focal loss
    else:
        mask_ndc = edge_mask.flatten().bool()
        edge_maps_gt_x = edge_maps[:, 0, ...].flatten()
        edge_maps_gt_y = edge_maps[:, 1, ...].flatten()
        edge_maps_gt_z = torch.logical_or(edge_maps_gt_x.bool(), edge_maps_gt_y.bool()).float()
        
        edge_maps_pre_x = edge_maps_pre[:, 0, ...].flatten()
        edge_maps_pre_y = edge_maps_pre[:, 1, ...].flatten()
        edge_maps_pre_z = edge_maps_pre[:, 2, ...].flatten()
        
        
        loss = sigmoid_loss(torch.cat((edge_maps_pre_x, edge_maps_pre_y)), torch.cat((edge_maps_gt_x, edge_maps_gt_y)), mask = torch.cat((mask_ndc, mask_ndc)), review = review)

        loss_z = sigmoid_loss(edge_maps_pre_z, edge_maps_gt_z, mask_usm, review= review)

        edge_maps_pre = edge_maps_pre[:, :2, ...]
        acc_x_pos, acc_x_neg, acc_y_pos, acc_y_neg, acc_a_pos, acc_a_neg = get_prediction_acc(edge_maps_pre, edge_maps, True)

    return loss, loss_z, acc_x_pos, acc_x_neg, acc_y_pos, acc_y_neg, acc_a_pos, acc_a_neg

def loss_usm(usm_pre, usm_gt):
    usm_pre = sigmoid(usm_pre)
    # generate the mask by dilate the ground truth
    # but we use gaussain blur instead
    blur = G(kernel_size = 3, sigma = 1.0)
    usm_mask_pos = (blur(usm_gt) != 0).float()
    usm_mask_neg = (blur(usm_gt) == 0).float() / 1e3
    loss_pos = torch.square((usm_pre - usm_gt) * usm_mask_pos).sum() / usm_gt.sum()
    loss_neg = torch.square((usm_pre - usm_gt) * usm_mask_neg).sum() / (1 - usm_gt).sum()
    return loss_pos + loss_neg

def loss_soft(input, target, mask):
    loss = sigmoid(target) * logsigmoid(input) * mask * -1
    return loss.mean()

def loss_skel(skel_pre, skel_gt, edge_mask):
    # this is infact similar to a curve regularization

    bce_loss = binary_cross_entropy_with_logits(skel_pre, skel_gt, reduction="none")
    
    prediction = sigmoid(skel_pre)
    p_t = prediction * skel_gt + (1 - prediction) * (1 - skel_gt)
    p_w = (1 - p_t) ** 2 # gamma is 2 as the focal loss does

    mask_pos_gt = skel_gt != 0
    mask_neg_gt = skel_gt == 0
    
    mask_pos = prediction > 0.5
    mask_pos_correct = torch.logical_and(mask_pos, mask_pos_gt) # reduce the loss if it is very confident
    mask_pos_wrong = torch.logical_xor(mask_pos_gt, mask_pos_correct) # increase the loss anyway

    mask_neg = prediction <= 0.5
    mask_neg_correct = torch.logical_and(torch.logical_and(mask_neg, mask_neg_gt), edge_mask)
    mask_neg_wrong = torch.logical_and(torch.logical_xor(mask_neg_gt, mask_neg_correct), edge_mask)
    
    masks = [mask_pos_correct, mask_pos_wrong, mask_neg_correct, mask_neg_wrong]

    # weights = [1, 1, 1, 1]
    weights = [1, 2, 1, 2] # this stage is refinement, and seems correction for false positive is much more important than the others
    losses = []
    loss = 0

    for i in range(len(masks)):
        if masks[i].sum() > 0:
            losses.append(bce_loss[masks[i]])
        else:
            losses.append(None)

    for i in range(len(losses)):
        if losses[i] is not None:
            loss = loss + (losses[i] * p_w[masks[i]]).mean() * weights[i] 

    return loss * 0.01


def get_prediction_acc(edge_maps_pre, edge_maps_gt, binary_mode = False):
    '''
    Given:
        edge_map_pre:   A float tensor with shape (k, 4), the prediction value of edge flag
        edge_map_gt:    A int tensor with shape (k, 1), the ground truth of edge flag 
                        (0: none, 1: x-edge, 2: y-edge, 3: both-edges)
    Return:
        the accruacy metrics
    '''
    if binary_mode:
        assert edge_maps_gt.shape[1] == 2
        assert edge_maps_pre.shape[1] == 2
        edge_maps_pre = torch.sigmoid(edge_maps_pre)
        edge_maps_x_pre = edge_maps_pre[:, 0, ...] > 0.5 # x-axis edge
        edge_maps_y_pre = edge_maps_pre[:, 1, ...] > 0.5 # y-axis edge
        edge_maps_a_pre = torch.logical_and(edge_maps_x_pre, edge_maps_y_pre).float() # both axis edge
        edge_maps_x_pre = edge_maps_x_pre.float()
        edge_maps_y_pre = edge_maps_y_pre.float()

        edge_maps_x_gt = edge_maps_gt[:, 0, ...] # x-axis edge
        edge_maps_y_gt = edge_maps_gt[:, 1, ...] # y-axis edge
        edge_maps_a_gt = torch.logical_and(edge_maps_x_gt, edge_maps_y_gt).float() # both axis edge
        edge_maps_x_gt = edge_maps_x_gt.float()
        edge_maps_y_gt = edge_maps_y_gt.float()
    else:
        softmax = torch.nn.Softmax(dim = -1)
        edge_maps_pre = softmax(edge_maps_pre)
        temp = torch.zeros_like(edge_maps_pre)
        temp[torch.arange(temp.shape[0]), edge_maps_pre.argmax(dim=1)] = 1
        
        edge_maps_x_pre = temp[:, 1, ...] # x-axis edge
        edge_maps_y_pre = temp[:, 2, ...] # y-axis edge
        edge_maps_a_pre = temp[:, 3, ...] # both axis edge
        
        edge_maps_x_gt = (edge_maps_gt == 1).float()
        edge_maps_y_gt = (edge_maps_gt == 2).float()
        edge_maps_a_gt = (edge_maps_gt == 3).float()
    
    acc_x_pos = torch.sum(edge_maps_x_gt * (edge_maps_x_pre).float()) / torch.clamp(torch.sum(edge_maps_x_gt), min=1)
    acc_x_neg = torch.sum((1 - edge_maps_x_gt) * (1 - edge_maps_x_pre).float()) / torch.clamp(torch.sum(1 - edge_maps_x_gt), min=1)
    acc_y_pos = torch.sum(edge_maps_y_gt * (edge_maps_y_pre).float()) / torch.clamp(torch.sum(edge_maps_y_gt), min=1)
    acc_y_neg = torch.sum((1 - edge_maps_y_gt) * (1 - edge_maps_y_pre).float()) / torch.clamp(torch.sum(1 - edge_maps_y_gt), min=1)
    acc_a_pos = torch.sum(edge_maps_a_gt * (edge_maps_a_pre).float()) / torch.clamp(torch.sum(edge_maps_a_gt), min=1)
    acc_a_neg = torch.sum((1 - edge_maps_a_gt) * (1 - edge_maps_a_pre).float()) / torch.clamp(torch.sum(1 - edge_maps_a_gt), min=1)

    return acc_x_pos, acc_x_neg, acc_y_pos, acc_y_neg, acc_a_pos, acc_a_neg


def get_grid_mask(edge_map_x, edge_map_y):
    '''
    Given:
        edge_map_x:     A int tensor with shape (b, h, w), contains 
                        ground truth edge flag that intersect with grid
                        edges along x-axis
        edge_map_y:     A int tensor with shape (b, h, w), contains 
                        ground truth edge flag that intersect with grid
                        edges along y-axis
    Return:
        grid_mask_x:    A int tensor with shape (b, h, w), contains the 
                        flag that indicate the current grid (set as 1 if 
                        it is labelled, else 0) contains a stroke that intersect
                        with grid edges along x-axis
        grid_mask_y:    A int tensor with shape (b, h, w), contains the 
                        flag that indicate the current grid (set as 1 if 
                        it is labelled, else 0) contains a stroke that intersect
                        with grid edges along y-axis
        grid_mask:      A int tensor with shape (b, h, w), it contains the 
                        flag that indicate the current grid has stroke inside
    '''
    # find the addtional x-valence from the bottom neighbour
    mask_x = torch.ones(edge_map_x.shape).to(edge_map_x.device)
    mask_x[:, 0, :] = 0
    grid_mask_x = torch.roll(edge_map_x, shifts=-1, dims=1) * mask_x

    # find the addtional y-valence from the right neighbour
    mask_y = torch.ones(edge_map_y.shape).to(edge_map_y.device)
    mask_y[:, :, 0] = 0
    grid_mask_y = torch.roll(edge_map_y, shifts=-1, dims=2) * mask_y

    # get mask for all grids that contains at least one polylines
    x_ = torch.logical_or(grid_mask_x.bool(), edge_map_x.bool())
    y_ = torch.logical_or(grid_mask_y.bool(), edge_map_y.bool())
    grid_mask_all = torch.logical_or(x_, y_).float()

    return grid_mask_x, grid_mask_y, grid_mask_all

def get_valence(edge_map_x, edge_map_y, edge_mask):
    '''
    Given:
        edge_map_x:     A int tensor with shape (b, h, w, 1), contains 
                        ground truth edge flag that intersect with grid
                        edges along x-axis
        edge_map_y:     A int tensor with shape (b, h, w, 1), contains 
                        ground truth edge flag that intersect with grid
                        edges along y-axis
    Return:
        valence_map:    A int tensor with shape (b, h, w, 1), contains the
                        valence of each grid
        valences:       A single int tensor which counts the number of grids
                        that has a specific type of valence (0 to 4)
    
    we need to compute the valence per grid, but remember the edge flag is defined as:
         X           
        .→ .→ .→ .→
      Y ↓  ↓  ↓  ↓
    so the valence of each grid should consider its left and top neighbour
    '''
    valence_both = edge_map_x.squeeze() + edge_map_y.squeeze()
    valence_x, valence_y, _ = get_grid_mask(edge_map_x, edge_map_y)

    # sum up all valences for each grid
    valence_map = (valence_both + valence_x.squeeze() + valence_y.squeeze()) * edge_mask.squeeze()

    # count the total grid numbers of each type of valence
    edge_mask = edge_mask.squeeze().bool()
    valence_0 = (valence_map[torch.logical_and(valence_map == 0, edge_mask)]).sum()
    valence_1 = (valence_map[torch.logical_and(valence_map == 1, edge_mask)]).sum()
    valence_2 = (valence_map[torch.logical_and(valence_map == 2, edge_mask)]).sum()
    valence_3 = (valence_map[torch.logical_and(valence_map == 3, edge_mask)]).sum()
    valence_4 = (valence_map[torch.logical_and(valence_map == 4, edge_mask)]).sum()

    # should we adjust the counter numbers accrodingly?
    return valence_map, (valence_0, valence_1, valence_2, valence_3, valence_4)

def loss_keypt(pt_map_pre, pt_map_gt, edge_maps_gt):
    '''
    Given:
        pt_map_pre:
        pt_map_gt:
        edge_map_gt:
    Return:
        loss
    '''
    # get grid mask
    edge_map_x, edge_map_y, _ = pre_to_map(edge_maps_gt.permute(0, 2, 3, 1))
    _, _, grid_mask = get_grid_mask(edge_map_x, edge_map_y)
    keypt_pos_mask = grid_mask.bool().unsqueeze(1)
    keypt_neg_mask = ~keypt_pos_mask

    # loss_map = torch.abs(pt_map_pre - pt_map_gt)
    loss_map = torch.square(pt_map_pre - pt_map_gt)
    
    # compute the loss
    masks = [keypt_pos_mask.repeat(1,2,1,1), keypt_neg_mask.repeat(1,2,1,1)]
    weights = [1, 0.05]
    return apply_weights(loss_map, masks, weights)

# from https://github.com/milesial/Pytorch-UNet/blob/23c14e6f908720c2d6ff4f828f494ab17539869e/utils/dice_score.py#L36
def dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all batches, or for a single mask
    assert input.size() == target.size()
    if input.dim() == 2 and reduce_batch_first:
        raise ValueError(f'Dice: asked to reduce batch but got tensor without batch dimension (shape {input.shape})')

    if input.dim() == 2 or reduce_batch_first:
        inter = torch.dot(input.reshape(-1), target.reshape(-1))
        sets_sum = torch.sum(input) + torch.sum(target)
        if sets_sum.item() == 0:
            sets_sum = 2 * inter

        return (2 * inter + epsilon) / (sets_sum + epsilon)
    else:
        # compute and average metric for each batch element
        dice = 0
        for i in range(input.shape[0]):
            dice += dice_coeff(input[i, ...], target[i, ...])
        return dice / input.shape[0]


def multiclass_dice_coeff(input: Tensor, target: Tensor, reduce_batch_first: bool = False, epsilon=1e-6):
    # Average of Dice coefficient for all classes
    assert input.size() == target.size()
    dice = 0
    for channel in range(input.shape[1]):
        dice += dice_coeff(input[:, channel, ...], target[:, channel, ...], reduce_batch_first, epsilon)

    return dice / input.shape[1]

def dice_loss(input: Tensor, target: Tensor, multiclass: bool = False):
    # Dice loss (objective to minimize) between 0 and 1
    assert input.size() == target.size()
    fn = multiclass_dice_coeff if multiclass else dice_coeff
    return 1 - fn(input, target, reduce_batch_first=True)

def valence_loss(edge_pre, edge_gt, edge_mask, method = "valence"):
    '''
    Given:
        edge_pre:   A float (b, 4, h, w) tensor, the edge flag prediction output from UDC network
        edge_gt:    A boolean (b, 2, h, w) tensor, the edge flag ground truth
        method:     string, should be one of "valence", "per-grid", "all"
    Return:
        loss:       valence loss
    '''
    edge_pre = edge_pre.permute(0,2,3,1)
    edge_gt = edge_gt.permute(0,2,3,1)
    c = edge_pre.shape[-1]
    assert c == 5
    
    # prediction float to flag maps
    edge_pre_x, edge_pre_y, edge_map_pre = pre_to_map(edge_pre[..., :-1])
    valence_pre, valence_sums_pre = get_valence(edge_pre_x, edge_pre_y, edge_mask)
    edge_gt_x, edge_gt_y = split_edge_map_per_axis(edge_gt)
    valence_gt, valence_sums_gt = get_valence(edge_gt_x, edge_gt_y, edge_mask)
    if method == "valence":
        loss = 0
        for i in range(len(valence_sums_pre)):
            loss = loss + torch.abs(valence_sums_pre[i] - valence_sums_gt[i]) / edge_pre.shape[0]
        return loss / len(valence_sums_pre)
        
    elif method == "per-grid":
        edge_mask = edge_mask.squeeze().bool()
        loss_map = torch.square(valence_pre - valence_gt)
        valence_mask_0 = torch.logical_and(valence_gt == 0, edge_mask)
        valence_mask_1 = torch.logical_and(valence_gt == 1, edge_mask)
        # valence_mask_1_wrong = torch.logical_and(torch.logical_xor(valence_pre == 1, valence_gt == 1), edge_mask)
        
        valence_mask_2 = torch.logical_and(valence_gt == 2, edge_mask)
        valence_mask_2_wrong = torch.logical_and(torch.logical_xor(valence_pre == 2, valence_gt == 2), edge_mask)
        
        valence_mask_3 = torch.logical_and(valence_gt == 3, edge_mask)
        valence_mask_3_wrong = torch.logical_and(torch.logical_xor(valence_pre == 3, valence_gt == 3), edge_mask)
        
        valence_mask_4 = torch.logical_and(valence_gt == 4, edge_mask)
        # valence_mask_4_wrong = torch.logical_and(torch.logical_xor(valence_pre == 4, valence_gt == 4), edge_mask)

        valence_masks = [valence_mask_0, valence_mask_1, valence_mask_2, valence_mask_3, valence_mask_4]
        weights = [1, 1, 1, 1, 1]
        valence_count = 0
        loss = 0
        for i in range(len(valence_masks)):
            if valence_masks[i].sum() > 0:
                loss = loss + loss_map[valence_masks[i]].mean() * weights[i]
                valence_count += 1
        return loss / valence_count

    elif method == "all":
        print("warning:\tthis method has not been implemented")
        return 0
    else:
        # if invalid method is given, then skip the valence loss 
        return 0

def split_edge_map_per_axis(edge_maps):
    '''
    Given:
        edge_maps:  A int tensor with shape (b, h, w, 1), contains 
                    all ground truth edge flag
    Return:
        edge_map_x: A int tensor with shape (b, h, w, 1), contains 
                    ground truth edge flag that intersect with grid
                    edges along x-axis
        edge_map_y: A int tensor with shape (b, h, w, 1), contains 
                    ground truth edge flag that intersect with grid
                    edges along y-axis
    '''
    edge_map_x = (edge_maps == 1).long() + (edge_maps == 3).long()
    edge_map_y = (edge_maps == 2).long() + (edge_maps == 3).long()
    return edge_map_x, edge_map_y