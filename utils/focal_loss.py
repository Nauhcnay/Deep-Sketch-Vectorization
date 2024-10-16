# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import numpy as np
from torch.nn import functional as F
from torchvision.transforms import GaussianBlur as G

def softmax_focal_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask = None,
    alpha = True,
    gamma: float = 2,
    reduction: str = "mean",
    focal = True,
    review = False,
    review_all = False
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: Adaptive loss weight for each class.
        gamma: Exponent of the modulating factor (1 - p_t) to
               balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    p = F.softmax(inputs, dim = -1)
    ce_loss = F.cross_entropy(inputs, targets, reduction="none")

    # get the prediction value at the ground truth position
    p_target = p[torch.arange(p.size(0)), p.argmax(dim = -1)]
    # find the correct prediction mask
    mask_correct = p.argmax(dim = -1) == targets
    mask_wrong = p.argmax(dim = -1) != targets
    mask_all = targets != 0
    mask_x = targets == 1
    mask_y = targets == 2
    mask_xy = targets == 3
    assert (mask_all == torch.logical_or(torch.logical_or(mask_x, mask_y), mask_xy)).all()

    mask_correct_f = mask_correct.float()
    p_t = p_target * mask_correct_f + (1 - p_target) * (1 - mask_correct_f)
    p_w = (1 - p_t) ** gamma
    if focal:
        loss = ce_loss * p_w
    else:
        loss = ce_loss

    if alpha:
        assert mask is not None
        '''
        Create mask by the GT, and decide the weights by number of different labels
        '''
        mask_none_outmask = torch.logical_not(mask)
        mask_none_inmask = torch.logical_xor(mask, mask_all)

        masks = [mask_none_outmask, mask_none_inmask, mask_all]
        weights = [0.02, 1, 1]

        # penalize all false positives
        if review_all:
            mask_wrong_outmask = torch.logical_and(torch.logical_not(mask), mask_wrong)
            masks.append(mask_wrong_outmask)
            weights.append(0.005)
        
        if review:
            mask_stroke_wrong = torch.logical_and(mask, mask_wrong) # false negative
            masks.append(mask_stroke_wrong)
            weights.append(0.01)

        return apply_weights(loss, masks, weights, reduction = reduction)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss

def sigmoid_loss(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mask,
    review = False,
    review_all = False,
    reduction: str = "mean",
    weights = [0.02, 1, 1],
    focal = True
) -> torch.Tensor:
    """
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    Args:
            positive vs negative examples. Default = -1 (no weighting).
    gamma: Exponent of the modulating factor (1 - p_t) to
           balance easy vs hard examples.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
    p = torch.sigmoid(inputs)
    if focal:
        p_t = p * targets + (1 - p) * (1 - targets)
        p_w = (1 - p_t) ** 2
        ce_loss = ce_loss * p_w

    mask_pos_gt = targets.bool()
    mask_neg_gt = torch.logical_not(targets)
    mask = torch.logical_or(mask, mask_pos_gt)
    mask_neg_outmask = torch.logical_not(mask)
    mask_neg_inmask = torch.logical_xor(mask, mask_pos_gt)
    assert (torch.logical_and(mask, mask_neg_gt) == mask_neg_inmask).all()
    masks = [mask_neg_outmask, mask_neg_inmask, mask_pos_gt]

    mask_pos_pre = p > 0.5 # all positive prediction
    mask_neg_pre = p <= 0.5 # all negative prediction
    mask_pos_correct = torch.logical_and(mask_pos_pre, mask_pos_gt) # true positive
    mask_pos_wrong = torch.logical_xor(mask_pos_gt, mask_pos_correct) # false positive
    mask_pos_wrong_ = torch.logical_and(mask_pos_gt, mask_neg_pre)
    assert (mask_pos_wrong_ == mask_pos_wrong).all()
    out_mask = torch.logical_not(mask)

    if review_all:
        mask_neg_wrong_outmask = torch.logical_and(torch.logical_and(mask_pos_pre, mask_neg_gt), mask_neg_outmask)
        masks.append(mask_neg_wrong_outmask)
        weights.append(0.005)
    
    if review: 
        mask_neg_wrong_inmask = torch.logical_and(torch.logical_and(mask_pos_pre, mask_neg_gt), mask) # false postive
        masks.append(torch.logical_or(mask_neg_wrong_inmask, mask_pos_wrong))
        weights.append(0.01)
            

    return apply_weights(ce_loss, masks, weights, reduction = reduction)

# sigmoid_focal_loss_jit: "torch.jit.ScriptModule" = torch.jit.script(sigmoid_loss)

def apply_weights(ce_loss, masks, weights, focal_weights = None, reduction = 'mean'):
    assert len(masks) == len(weights)
    losses = []
    loss = 0
    avg_counter = 0
    if focal_weights is None:
        focal_weights = torch.ones(ce_loss.shape).to(ce_loss.device).float()

    for i in range(len(masks)):
        if masks[i].sum() > 0:
            losses.append(ce_loss[masks[i]])
        else:
            losses.append(None)
    for i in range(len(masks)):
        if losses[i] is not None:
            if reduction == 'mean':
                loss =  loss + (losses[i] * weights[i] * focal_weights[masks[i]]).mean()
            elif reduction == 'sum':
                loss =  loss + (losses[i] * weights[i] * focal_weights[masks[i]]).sum()
            else:
                raise ValueError("Unsupported reduction method %s"%reduction)
            avg_counter += 1
    return loss / avg_counter

def sigmoid_focal_loss_star(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    alpha: float = -1,
    gamma: float = 1,
    reduction: str = "none",
) -> torch.Tensor:
    """
    FL* described in RetinaNet paper Appendix: https://arxiv.org/abs/1708.02002.
    Args:
        inputs: A float tensor of arbitrary shape.
                The predictions for each example.
        targets: A float tensor with the same shape as inputs. Stores the binary
                 classification label for each element in inputs
                (0 for the negative class and 1 for the positive class).
        alpha: (optional) Weighting factor in range (0,1) to balance
                positive vs negative examples. Default = -1 (no weighting).
        gamma: Gamma parameter described in FL*. Default = 1 (no weighting).
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        Loss tensor with the reduction option applied.
    """
    inputs = inputs.float()
    targets = targets.float()
    shifted_inputs = gamma * (inputs * (2 * targets - 1))
    loss = -(F.logsigmoid(shifted_inputs)) / gamma

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss *= alpha_t

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


sigmoid_focal_loss_star_jit: "torch.jit.ScriptModule" = torch.jit.script(
    sigmoid_focal_loss_star
)