import torch
import numpy as np
import argparse
import os

from utils.ndc_tools import pre_to_lines, lines_to_svg, map_to_lines, pre_to_map, refine_topology
from utils.ndc_loss import get_valence
from os.path import join, exists, split
from network.udc import CNN_2d_resnet
from types import SimpleNamespace
from train_s2 import init_base_coord
from pathlib import Path


def load_udf(path_to_npz, dist, device, gts, skel=False, noise_level=-1):
    udf = np.load(path_to_npz)["udf"]
    edge_x = gts['edge_x']
    edge_y = gts['edge_y']
    s = edge_x + 2 * edge_y
    if skel:
        udf[np.where(s)] = 0

    if noise_level != -1:
        # add noise onto the sketch
        w = np.random.uniform(high=noise_level)
        udf_noise = np.abs(
            np.random.normal(
                loc=0,
                scale=w *
                udf.std(),
                size=udf.shape))
        udf = udf + udf_noise

    udf = udf[np.newaxis, np.newaxis, ...].clip(0, dist) / dist
    return torch.FloatTensor(udf).to(device), torch.FloatTensor(s)


def load_keypt(path_to_npz):
    data = np.load(path_to_npz)
    wanted_keys = ['end_point', 'sharp_turn', 'T', 'X', 'star']
    return {key: data[key] for key in wanted_keys}


def load_gts(path_to_npz):
    data = np.load(path_to_npz)
    wanted_keys = ['edge_x', 'edge_y', 'pt_map', 'under_sampled', 'udf']
    res = {}
    for key in wanted_keys:
        res[key] = data[key][:-1, :-1]
    return res


def udf_to_svg(net, udf):
    pass


def vis_ndc(canvas_h, canvas_w, path_to_npz):
    ndc_npz = np.load(path_to_npz)
    path_to_npz, name = split(path_to_npz)
    edge_ndc = ndc_npz['edge'].transpose(1, 0, 2)
    w, h, _ = edge_ndc.shape
    pt_ndc = ndc_npz['pt']

    for i in range(h):
        for j in range(w):
            pt_ndc[i, j, 0] += i
            pt_ndc[i, j, 1] += j
    # pt_ndc[..., [0, 1]] = pt_ij   _ndc[..., [1, 0]]
    pt_ndc = pt_ndc.transpose(1, 0, 2)
    lines_ndc, _, _ = map_to_lines(edge_ndc, pt_ndc)
    path_to_npz = Path(path_to_npz)
    lines_to_svg(
        lines_ndc,
        canvas_w,
        canvas_h,
        join(
            path_to_npz.parent,
            'svg',
            name.replace(
                '.npz',
                '.svg')))


def vis_gt(canvas_h, canvas_w, edge_map, pt_map):
    lines_ndc, _, _ = map_to_lines(edge_map, pt_map)
    lines_to_svg(lines_ndc, canvas_w, canvas_h, "sketchvg_gt.svg")


def get_acc(edge_maps_pre_xy, edge_maps_gt, edge_mask):

    edge_maps_x_pre = (edge_maps_pre_xy == 1).float() * \
        edge_mask  # x-axis edge
    edge_maps_y_pre = (edge_maps_pre_xy == 2).float() * \
        edge_mask  # y-axis edge
    edge_maps_a_pre = (edge_maps_pre_xy == 3).float() * \
        edge_mask  # both axis edge

    edge_maps_x_gt = (edge_maps_gt == 1).float()
    edge_maps_y_gt = (edge_maps_gt == 2).float()
    edge_maps_a_gt = (edge_maps_gt == 3).float()

    acc_x_pos = torch.sum(edge_maps_x_gt * (edge_maps_x_pre).float()) / \
        torch.clamp(torch.sum(edge_maps_x_gt), min=1)
    acc_x_neg = torch.sum((1 - edge_maps_x_gt) * (1 - edge_maps_x_pre).float()
                          ) / torch.clamp(torch.sum(1 - edge_maps_x_gt), min=1)
    acc_x_recall = torch.sum(edge_maps_x_gt * (edge_maps_x_pre).float()) / \
        torch.clamp(torch.sum(edge_maps_x_pre), min=1)

    acc_y_pos = torch.sum(edge_maps_y_gt * (edge_maps_y_pre).float()) / \
        torch.clamp(torch.sum(edge_maps_y_gt), min=1)
    acc_y_neg = torch.sum((1 - edge_maps_y_gt) * (1 - edge_maps_y_pre).float()
                          ) / torch.clamp(torch.sum(1 - edge_maps_y_gt), min=1)
    acc_y_recall = torch.sum(edge_maps_y_gt * (edge_maps_y_pre).float()) / \
        torch.clamp(torch.sum(edge_maps_y_pre), min=1)

    acc_a_pos = torch.sum(edge_maps_a_gt * (edge_maps_a_pre).float()) / \
        torch.clamp(torch.sum(edge_maps_a_gt), min=1)
    acc_a_neg = torch.sum((1 - edge_maps_a_gt) * (1 - edge_maps_a_pre).float()
                          ) / torch.clamp(torch.sum(1 - edge_maps_a_gt), min=1)
    acc_a_recall = torch.sum(edge_maps_a_gt * (edge_maps_a_pre).float()) / \
        torch.clamp(torch.sum(edge_maps_a_pre), min=1)

    return acc_x_pos, acc_x_neg, acc_x_recall, acc_y_pos, acc_y_neg, acc_y_recall, acc_a_pos, acc_a_neg, acc_a_recall


if __name__ == "__main__":
    # we will output 3 results:
    # 1. direct output
    # 2. results after stroke refinement
    # 3. results after topology refinement
    parser = argparse.ArgumentParser(
        description="Testing code for NDC Sketch Network")
    parser.add_argument("--input", "-i", default='./data/benchmark/gt')
    parser.add_argument(
        "--model",
        "-m",
        default="./pertrained/ablation/ndc_noisy_wo_skel.pth")
    parser.add_argument(
        "--output",
        "-o",
        default='./experiments/08.exp_compairison/08.ndc_skel')
    args = parser.parse_args()
    path_3d_ndc = './experiments/08.exp_compairison/04.3D NDC'

    # init
    assert (os.path.exists(args.model))
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ckpt = torch.load(args.model)
    try:
        args_ndc = SimpleNamespace(**ckpt['param'])
    except BaseException:
        args_ndc = ckpt['param']
    net_ndc = CNN_2d_resnet(
        device,
        channels=args_ndc.dims,
        layers=args_ndc.layers,
        batch_norm=args_ndc.bn,
        multi_scale=args_ndc.ms,
        grad=args_ndc.grad)
    # Modify the state dictionary to drop specific layer weights
    ckpt['model_state_dict'] = {
        k: v for k,
        v in ckpt['model_state_dict'].items() if 'usm' not in k}
    # we will skip optimizer loading
    net_ndc.load_state_dict(ckpt['model_state_dict'])
    net_ndc = net_ndc.to(device)

    logs = []

    for npz in os.listdir(args.input):
        # load data
        print("log:\topening %s" % npz)
        if os.path.exists(join(args.output, npz.replace('.npz', '_org.svg'))):
            continue
        line = npz
        if exists(join(args.output, 'svg', npz.replace('.npz', '.svg'))):
            continue
        input_path = join(args.input, npz)
        gts = load_gts(input_path)
        keypts_dict = load_keypt(input_path)
        udf, edge_maps_gt_xy = load_udf(
            input_path, args_ndc.dist_clip, device, gts, False, noise_level=0.1)

        usm_gt = torch.Tensor(gts['under_sampled'])
        pt_map_gt = gts['pt_map']
        # edge_mask = ((gts['udf'] <= args_ndc.dist_ndc)).astype(bool)
        edge_mask = ((gts['udf'] <= 5)).astype(bool)

        # predict
        base_coord = init_base_coord(udf, udf.device)
        net_ndc.eval()
        with torch.no_grad():
            edge_maps_pre, pt_map_pre, _ = net_ndc.forward(udf)

        pt_map_pre = pt_map_pre + base_coord
        pt_map_pre = pt_map_pre.permute(0, 2, 3, 1).squeeze()
        edge_maps_pre = edge_maps_pre.permute(
            0, 2, 3, 1)[..., :-1]  # we discard skeleton layer
        _, canvas_h, canvas_w, _ = edge_maps_pre.shape
        # lines_pre = pre_to_lines(edge_maps_pre[0], pt_map_pre) # remove batch
        # dimension

        # reconstruct original svg from prediction
        if True:
            edge_map_x_pre, edge_map_y_pre, _ = pre_to_map(edge_maps_pre[0])
            edge_map_x_pre = edge_map_x_pre.cpu() * torch.FloatTensor(edge_mask)
            edge_map_y_pre = edge_map_y_pre.cpu() * torch.FloatTensor(edge_mask)
            edge_maps_pre_xy = (
                edge_map_x_pre + 2 * edge_map_y_pre).squeeze().cpu()
            # edge_maps_pre_xy = edge_maps_pre_xy * torch.FloatTensor(edge_mask)
            edge_map_pre_ = torch.stack(
                (edge_map_x_pre, edge_map_y_pre), dim=-1)
            lines_pre, lines_map_x_pre, lines_map_y_pre, _ = map_to_lines(
                edge_map_pre_, pt_map_pre, True)
            lines_to_svg(
                lines_pre, canvas_w, canvas_h, join(
                    args.output, npz.replace(
                        '.npz', '_org.svg')))
            lines_pre = refine_topology(
                edge_map_pre_,
                pt_map_pre,
                usm_gt,
                lines_map_x_pre,
                lines_map_y_pre,
                keypts_dict,
                True)
            # lines_pre = pre_to_lines(edge_map_pre_, pt_map_pre[i])
            lines_to_svg(
                lines_pre, canvas_w, canvas_h, join(
                    args.output, npz.replace(
                        '.npz', '_refined.svg')))
            acc_x_pos, acc_x_neg, acc_x_recall, acc_y_pos, acc_y_neg, acc_y_recall, acc_a_pos, acc_a_neg, acc_a_recall = get_acc(
                edge_maps_pre_xy, edge_maps_gt_xy, edge_mask)
            # compute valence loss
            line = line + \
                " " + str(((acc_x_pos + acc_x_neg) / 2).item()) + " " + str(acc_x_recall.item()) + \
                " " + str(((acc_y_pos + acc_y_neg) / 2).item()) + " " + str(acc_y_recall.item()) + \
                " " + str(((acc_a_pos + acc_a_neg) / 2).item()) + " " + str(acc_a_recall.item())

            valence_pre, valence_sums_pre = get_valence(
                edge_map_x_pre.unsqueeze(-1), edge_map_y_pre.unsqueeze(-1), torch.Tensor(edge_mask))
            valence_gt, valence_sums_gt = get_valence((edge_maps_gt_xy == 1).unsqueeze(
                -1), (edge_maps_gt_xy == 2).unsqueeze(-1), torch.Tensor(edge_mask))
            loss = 0
            for i in range(len(valence_sums_pre)):
                loss = loss + \
                    torch.abs(valence_sums_pre[i] - valence_sums_gt[i])
            loss = loss / len(valence_sums_pre)
            line = line + " " + str(loss.item())

        # reconstruct gt svg
        if False:
            edge_map_x_gt, edge_map_y_gt, _ = pre_to_map(
                edge_maps_gt_xy.unsqueeze(-1))
            edge_maps_gt_xy = (
                edge_map_x_gt +
                2 *
                edge_map_y_gt).squeeze().cpu().numpy()
            edge_map_gt_ = np.stack((edge_map_x_gt, edge_map_y_gt), axis=-1)
            lines_gt, lines_map_x_gt, lines_map_y_gt, _ = map_to_lines(
                edge_map_gt_, pt_map_gt, False)
            lines_to_svg(
                lines_gt, canvas_w, canvas_h, join(
                    args.output, npz.replace(
                        '.npz', '_org_gt.svg')))
            lines_gt = refine_topology(
                edge_map_gt_,
                pt_map_gt,
                usm_gt.numpy(),
                lines_map_x_gt,
                lines_map_y_gt,
                keypts_dict,
                False)
            lines_to_svg(
                lines_gt, canvas_w, canvas_h, join(
                    args.output, npz.replace(
                        '.npz', '_refined_gt.svg')))

        # reconstruct svg from 3D NDC
        if False:
            res_ndc = np.load(join(path_3d_ndc, npz))
            edge_maps_ndc_pre = res_ndc['edge']
            pt_map_ndc_pre = res_ndc['pt']
            edge_maps_ndc_pre = edge_maps_ndc_pre.transpose(1, 0, 2)
            h, w, _ = edge_maps_ndc_pre.shape
            # recover pt map
            for j in range(h):
                for i in range(w):
                    pt_map_ndc_pre[i, j, 0] += i
                    pt_map_ndc_pre[i, j, 1] += j

            pt_map_ndc_pre = pt_map_ndc_pre.transpose(1, 0, 2)
            lines_ndc, _, _ = map_to_lines(edge_maps_ndc_pre, pt_map_ndc_pre)
            lines_to_svg(
                lines_ndc, canvas_w, canvas_h, join(
                    args.output, npz.replace(
                        '.npz', '_3d_ndc.svg')))
            edge_maps_ndc_pre_xy = edge_maps_ndc_pre[...,
                                                     0] + 2 * edge_maps_ndc_pre[..., 1]
            acc_x_pos, acc_x_neg, acc_x_recall, acc_y_pos, acc_y_neg, acc_y_recall, acc_a_pos, acc_a_neg, acc_a_recall = get_acc(
                torch.Tensor(edge_maps_ndc_pre_xy), torch.Tensor(edge_maps_gt_xy)[3:h + 3, 3:w + 3], edge_mask[3:h + 3, 3:w + 3])

            line = line + \
                " " + str(acc_x_pos_ndc.item()) + " " + str(acc_x_neg_ndc.item()) + \
                " " + str(acc_y_pos_ndc.item()) + " " + str(acc_y_neg_ndc.item()) + \
                " " + str(acc_a_pos_ndc.item()) + " " + str(acc_a_neg_ndc.item())

            edge_maps_ndc_pre = torch.Tensor(edge_maps_ndc_pre)
            valence_pre, valence_sums_pre = get_valence(edge_maps_ndc_pre[..., 0].unsqueeze(
                -1), edge_maps_ndc_pre[..., 1].unsqueeze(-1), torch.Tensor(edge_mask)[3:h + 3, 3:w + 3])
            edge_maps_gt_xy = torch.Tensor(edge_maps_gt_xy[3:h + 3, 3:w + 3])
            edge_mask = edge_mask[3:h + 3, 3:w + 3]
            valence_gt, valence_sums_gt = get_valence((edge_maps_gt_xy == 1).unsqueeze(
                -1), (edge_maps_gt_xy == 2).unsqueeze(-1), torch.Tensor(edge_mask))
            loss = 0
            for i in range(len(valence_sums_pre)):
                loss = loss + \
                    torch.abs(valence_sums_pre[i] - valence_sums_gt[i])
            loss = loss / len(valence_sums_pre)
            line = line + " " + str(loss.item())
        logs.append(line)

    with open("ndc_exp_log_woskel_noisy.txt", 'w') as f:
        f.write('\n'.join(logs))
