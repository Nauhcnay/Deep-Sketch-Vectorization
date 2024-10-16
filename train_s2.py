import torch
import wandb
import os
import argparse
import cv2

import numpy as np
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import time

from torch import sigmoid
from torch import Tensor
from os.path import join, exists
from network.udc import CNN_2d_resnet
from dataset.s2_udf import noisy_udf
from tqdm import tqdm
from PIL import Image
from datetime import datetime

from utils.ndc_tools import pre_to_lines, lines_to_svg, refine_topology, pre_to_map, map_to_lines, init_base_coord
from utils.ndc_loss import loss_edge, loss_keypt, valence_loss, loss_usm, loss_skel, loss_sparsity, loss_soft
from dataset.augmentation import blend_skeletons
from dataset.preprocess import rasterize, svg_to_numpy
from utils.focal_loss import sigmoid_loss
from utils.losses import dist_loss
from utils.keypt_tools import add_title

DEBUG = True

# https://github.com/fastai/fastbook/issues/85
# when test on windows worker number should always be 0


def load_data(
        root,
        batch_size,
        num_workers=0,
        noise_level=0.01,
        dist_clip=8.5,
        dist_ndc=1,
        insert_skel=False,
        multi_scale=False,
        approx_udf=False):

    train_set = noisy_udf(
        root,
        noise_level=noise_level,
        is_train=True,
        dist_clip=dist_clip,
        dist_ndc=dist_ndc,
        insert_skel=insert_skel,
        multi_scale=multi_scale,
        approx_udf=approx_udf)
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers)

    val_set = noisy_udf(
        root,
        noise_level=noise_level,
        is_train=False,
        dist_clip=dist_clip,
        dist_ndc=dist_ndc,
        insert_skel=insert_skel,
        multi_scale=multi_scale,
        approx_udf=approx_udf)
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=1, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader


def train():
    args = parse()
    # load data
    train_loader, val_loader = load_data(args.ds, args.bs, num_workers=args.workers, noise_level=args.nl, approx_udf=args.approx_udf,
                                         dist_clip=args.dist_clip, insert_skel=args.insert_skel, dist_ndc=args.dist_ndc, multi_scale=args.mgs)
    # load network, let's predict both map at the same time
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    net = CNN_2d_resnet(
        device,
        channels=args.dims,
        layers=args.layers,
        multi_scale=args.msb,
        batch_norm=args.bn,
        drop_out=args.do,
        coord_conv=args.coord,
        noisy=args.noisy,
        resnext_input=args.resnext_i,
        resnext_feature=args.resnext_f)
    milestones = [30, 60]
    net = net.to(device)

    # init optimizer, scheduler, loss layer
    optimizer = optim.Adam(net.parameters(), lr=args.lr, weight_decay=1e-8)

    # init optimizer scheduler
    if args.sch:
        # scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epoch)
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=milestones, gamma=0.5)

    # load model if possible
    if args.cp == "./models_ndc":
        now = datetime.now()
        dt_formatted = now.strftime("D%Y-%m-%dT%H-%M-%S")
        if args.name is not None:
            dt_formatted = dt_formatted + "-" + args.name
        model_folder = join(args.cp, dt_formatted)
        os.makedirs(model_folder)
    else:
        model_folder = args.cp

    epoch_start = 0
    model_path = join(model_folder, "last_epoch.pth")
    model_path_keypt = join(model_folder, "last_epoch_keypt.pth")
    if exists(model_path):
        print("Log\tloading %s" % model_path)
        checkpoint = torch.load(model_path)
        net.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch_start = checkpoint['epoch']
        args = checkpoint['param']

    if exists(model_path_keypt) and args.keypt:
        print("Log\tloading %s" % model_path_keypt)
        checkpoint_keypt = torch.load(model_path_keypt)
        net_keypt.load_state_dict(checkpoint_keypt['model_state_dict'])
        optimizer_keypt.load_state_dict(
            checkpoint_keypt['optimizer_state_dict'])

    # inital wandb
    if args.log:
        wandb.init(
            project="Neural Dual Contouring 2D",
            entity="waterheater",
            config=args,
            name=args.name)
        config = wandb.config
    else:
        config = args

    # train
    iter_counter = 0  # we use a global iter_counter for iterations
    iter_counter_val = 0

    def start_epoches(
            iter_counter,
            iter_counter_val,
            config,
            epoch_start,
            train_loader,
            val_loader,
            model_folder):
        for i in range(config.epoch - epoch_start):
            # record the prediction accuracy
            train_acc_pos_x = 0
            train_acc_pos_y = 0
            train_acc_pos_a = 0
            train_acc_neg_x = 0
            train_acc_neg_y = 0
            train_acc_neg_a = 0
            train_avg_acc_count = 0
            epoch = i + epoch_start

            print("Log:\tstarting epoch %d" % epoch)
            net.train()
            if args.keypt_dual:
                net_keypt.train()

            # set a counter to check if it is the last batch
            pcounter = 0
            last_batch = len(train_loader)
            pbar = tqdm(train_loader, ncols=175)
            stime = time.time()
            for data in pbar:
                # get training data from dataset
                udf, edge_maps_gt, pt_map_gt, edge_mask, skel_gt, keypts_dicts, gsizes = data
                base_coord = init_base_coord(pt_map_gt, device, gsizes)
                pt_map_gt = pt_map_gt - base_coord
                if args.debug:
                    etime = time.time()
                    print("log:\tload data in %s seconds" % (etime - stime))
                    stime = etime

                # start train
                edge_maps_pre, pt_map_pre = net.forward(udf.float(), gsizes)
                if args.debug:
                    etime = time.time()
                    print(
                        "log:\tpass through network in %s seconds" %
                        (etime - stime))
                    stime = etime

                # compute loss
                acc_list = [
                    train_acc_pos_x,
                    train_acc_pos_y,
                    train_acc_pos_a,
                    train_acc_neg_x,
                    train_acc_neg_y,
                    train_acc_neg_a,
                    train_avg_acc_count]
                loss_dict, acc_list, acc_res_list = compute_ndc_loss_all(
                    args, edge_maps_pre, edge_maps_gt, edge_mask, pt_map_pre, pt_map_gt, acc_list)
                train_acc_pos_x, train_acc_pos_y, train_acc_pos_a, train_acc_neg_x, train_acc_neg_y, train_acc_neg_a, train_avg_acc_count = acc_list
                acc_pos_x, acc_pos_y, acc_pos_xy, acc_neg_x, acc_neg_y, acc_neg_xy = acc_res_list
                if args.debug:
                    etime = time.time()
                    print("log:\tcompute loss in %s seconds" % (etime - stime))
                    stime = etime

                # back propagate gradient
                optimizer.zero_grad()
                loss_dict['all'].backward()
                if args.keypt_dual:
                    loss_dict['keypt'].backward()
                optimizer.step()
                if args.keypt_dual:
                    optimizer_keypt.step()
                    scheduler_keypt.step()
                if config.sch:
                    scheduler.step()
                if args.debug:
                    etime = time.time()
                    print(
                        "log:\tback propagate gradient in %s seconds" %
                        (etime - stime))
                    stime = etime
                pbar.set_description(
                    "Loss: %.4f, ACC X: %.2f(P)-%.2f(N), ACC Y: %.2f(P)-%.2f(N), ACC XY: %.2f(P)-%.2f(N)" %
                    (loss_dict['edge'].item(),
                     acc_pos_x,
                     acc_neg_x,
                     acc_pos_y,
                     acc_neg_y,
                     acc_pos_xy,
                     acc_neg_xy))

                # visualize result
                if iter_counter % 3000 == 0:
                    pres = (edge_maps_pre, pt_map_pre + base_coord, None)
                    gts = (edge_maps_gt, pt_map_gt + base_coord, skel_gt, None)
                    train_res = vis_train_ndc(
                        config,
                        pres,
                        gts,
                        model_folder,
                        iter_counter,
                        gsizes,
                        keypts_dicts_gt=keypts_dicts,
                        keypts_dicts_pre=keypts_dicts)
                    if args.debug:
                        etime = time.time()
                        print(
                            "log:\tsave visualization results in %s seconds" %
                            (etime - stime))
                        stime = etime

                if config.log:
                    # record loss
                    if iter_counter % 500 == 0:
                        log_loss_ndc(
                            config, loss_dict, iter_counter, acc_res_list)

                iter_counter += 1
                pcounter += 1
                if args.debug:
                    if pcounter % 10 == 0:
                        break

            # update save model every epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'param': vars(args)
            }, model_path)

            # save model every 10 epoch
            if epoch % 10 == 0:
                model_iter_path = join(model_folder, "epoch_%07d.pth" % epoch)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'param': vars(args)
                }, model_iter_path)

            # start validation
            if epoch % 3 == 0:
                print("Log:\tstarting evaluation")
                net.eval()
                if args.keypt_dual:
                    net_keypt.eval()
                val_acc_pos_x = 0
                val_acc_pos_y = 0
                val_acc_pos_a = 0
                val_acc_neg_x = 0
                val_acc_neg_y = 0
                val_acc_neg_a = 0
                val_avg_acc_count = 0

                edge_loss_val_total = 0
                sparsity_loss_val_total = 0
                skel_loss_val_total = 0
                keypt_loss_val_total = 0
                loss_val_v_total = 0

                save_counter = 0
                val_pngs = []
                pbar_val = tqdm(val_loader, ncols=175)
                for data in pbar_val:
                    iter_counter_val += 1
                    with torch.no_grad():
                        udf_val, edge_maps_val, pt_map_val, edge_mask_val, skel_val, keypt_dicts_val, gsizes_val = data
                        base_coord_val = init_base_coord(
                            pt_map_val, device, gsizes_val)
                        pt_map_val = pt_map_val - base_coord_val

                        edge_maps_pre_val, pt_map_pre_val = net.forward(
                            udf_val.float(), gsizes_val)

                        acc_list_val = (
                            val_acc_pos_x,
                            val_acc_pos_y,
                            val_acc_pos_a,
                            val_acc_neg_x,
                            val_acc_neg_y,
                            val_acc_neg_a,
                            val_avg_acc_count)
                        loss_dict, acc_list_val, acc_res_list = compute_ndc_loss_all(
                            args, edge_maps_pre_val, edge_maps_val, edge_mask_val, pt_map_pre_val, pt_map_val, acc_list_val)
                        val_acc_pos_x, val_acc_pos_y, val_acc_pos_a, val_acc_neg_x, val_acc_neg_y, val_acc_neg_a, val_avg_acc_count = acc_list_val
                        acc_pos_x_val, acc_pos_y_val, acc_pos_xy_val, acc_neg_x_val, acc_neg_y_val, acc_neg_xy_val = acc_res_list

                        edge_loss_val_total += loss_dict['edge']
                        skel_loss_val_total += loss_dict.get('skel', 0)
                        keypt_loss_val_total += loss_dict.get('keypt', 0)
                        loss_val_v_total += loss_dict.get('valence', 0)

                        pbar_val.set_description(
                            "Loss: %.4f, ACC X: %.2f(P)-%.2f(N), ACC Y: %.2f(P)-%.2f(N), ACC XY: %.2f(P)-%.2f(N)" %
                            (loss_dict['edge'].item(),
                             acc_pos_x_val,
                             acc_neg_x_val,
                             acc_pos_y_val,
                             acc_neg_y_val,
                             acc_pos_xy_val,
                             acc_neg_xy_val))

                        # we just need 10 visualization results for each
                        # validation
                        if args.debug:
                            print("log:\tsave validation result to images")
                        if save_counter < 9:
                            # generate the svg from gt and prediction
                            pt_map_val = pt_map_val + base_coord_val
                            pt_map_pre_val = pt_map_pre_val + base_coord_val

                            edge_maps_val = edge_maps_val.permute(0, 2, 3, 1)
                            pt_map_val = pt_map_val.permute(0, 2, 3, 1)
                            edge_maps_pre_xy_val = edge_maps_pre_val.permute(
                                0, 2, 3, 1)[..., :-1]
                            edge_maps_pre_z_val = edge_maps_pre_val.permute(
                                0, 2, 3, 1)[..., -1]
                            pt_map_pre_val = pt_map_pre_val.permute(0, 2, 3, 1)
                            skel_pre_val = sigmoid(edge_maps_pre_z_val) > 0.5
                            usm_pre_val = None

                            # let's try to refine the reconstruction, use this
                            # swith to turn on and off the refinement
                            if False:
                                keypts_dict_val = unpad_keypt_dict(
                                    keypt_dicts_val, 0)
                                edge_map_x_val, edge_map_y_val, _ = pre_to_map(
                                    edge_maps_val[0])
                                edge_map_val = torch.stack(
                                    (edge_map_x_val, edge_map_y_val), dim=-1)
                                _, lines_map_x, lines_map_y, _ = map_to_lines(
                                    edge_map_val, pt_map_val[0], True)
                                try:
                                    lines = refine_topology(
                                        edge_map_val,
                                        pt_map_val[0],
                                        usm_gt_val[0],
                                        lines_map_x,
                                        lines_map_y,
                                        keypts_dict_val,
                                        True)
                                except BaseException:
                                    lines = pre_to_lines(
                                        edge_maps_val[0], pt_map_val[0])
                            else:
                                lines = pre_to_lines(
                                    edge_maps_val[0], pt_map_val[0])

                            if config.keypt_only:
                                edge_maps_pre_xy_val = edge_maps_val

                            if False:
                                edge_map_x_val_pre, edge_map_y_val_pre, _ = pre_to_map(
                                    edge_maps_pre_xy_val[0])
                                edge_map_val_pre = torch.stack(
                                    (edge_map_x_val_pre, edge_map_y_val_pre), dim=-1)
                                _, lines_map_x_pre, lines_map_y_pre, _ = map_to_lines(
                                    edge_map_val_pre, pt_map_pre_val[0], True)
                                try:
                                    lines_pre = refine_topology(
                                        edge_map_val_pre,
                                        pt_map_pre_val[0],
                                        usm_pre_val[0],
                                        lines_map_x_pre,
                                        lines_map_y_pre,
                                        keypts_dict_val,
                                        True)
                                except BaseException:
                                    lines_pre = pre_to_lines(
                                        edge_maps_pre_xy_val[0], pt_map_pre_val[0])
                            else:
                                lines_pre = pre_to_lines(
                                    edge_maps_pre_xy_val[0], pt_map_pre_val[0])
                            _, canvas_h, canvas_w, _ = edge_maps_val.shape
                            canvas_h = int(canvas_h * float(gsizes_val[0]))
                            canvas_w = int(canvas_w * float(gsizes_val[0]))
                            if exists(join(model_folder, "results")) == False:
                                os.makedirs(join(model_folder, "results"))

                            svg_pre = join(
                                model_folder,
                                "results",
                                "val_iter%07d_pre.svg" %
                                (iter_counter_val))
                            if lines_pre is not None:
                                lines_pre_flag = lines_to_svg(
                                    lines_pre, canvas_w, canvas_h, svg_pre)
                            else:
                                lines_pre_flag = False

                            svg_gt = join(
                                model_folder,
                                "results",
                                "val_iter%07d_gt.svg" %
                                (iter_counter_val))
                            if lines is not None:
                                lines_flag = lines_to_svg(
                                    lines, canvas_w, canvas_h, svg_gt)
                            else:
                                lines_flag = False

                            # record all validation results
                            skel_pre_val_np = skel_pre_val.detach().cpu().squeeze().numpy().astype(int)
                            skel_val_np = skel_val.detach().cpu().squeeze().numpy().astype(int)
                            skel_np = blend_skeletons(
                                skel_val_np, skel_pre_val_np, alpha=0.5)
                            figure = cv2.resize(
                                skel_np, to_target_wh(skel_np), interpolation=cv2.INTER_AREA)
                            val_pngs.append(figure)
                            if lines_pre_flag:
                                res_pre = svg_to_numpy(svg_pre)
                            else:
                                res_pre = None
                            if lines_pre_flag and res_pre is not None:
                                res_pre = cv2.resize(
                                    res_pre, to_target_wh(res_pre), interpolation=cv2.INTER_AREA)
                                val_pngs.append(res_pre)
                            else:
                                val_pngs.append(
                                    (np.ones(
                                        to_target_wh(
                                            figure,
                                            reverse=True)) *
                                        255).astype(
                                        np.uint8))
                            if lines_flag:
                                res_gt = svg_to_numpy(svg_gt)
                                res_gt = np.repeat(
                                    res_gt[..., np.newaxis], 3, axis=-1)
                                # res_gt = blend_skeletons(res_gt, (usm_pre_val_np, usm_gt_val_np), usm_mode = True)
                            else:
                                res_gt = None
                            if lines_flag and res_gt is not None:
                                res_gt = cv2.resize(
                                    res_gt, to_target_wh(res_gt), interpolation=cv2.INTER_AREA)
                                val_pngs.append(res_gt)
                            else:
                                h_, w_ = to_target_wh(figure, reverse=True)
                                val_pngs.append(
                                    (np.ones(
                                        h_,
                                        w_,
                                        3) *
                                        255).astype(
                                        np.uint8))
                        save_counter += 1
                        if args.debug:
                            import pdb
                            pdb.set_trace()

                # write the validation result, each row will have 3 samples
                samples_per_row = 3
                val_res = []
                for m in range(0, len(val_pngs), samples_per_row * 3):
                    val_row = []
                    for n in range(samples_per_row):
                        idx = m + n * 3
                        if idx >= len(val_pngs):
                            val_row.append(np.ones(val_pngs[0].shape) * 255)
                            val_row.append(np.ones(val_pngs[0].shape) * 255)
                            val_row.append(np.ones(val_pngs[0].shape) * 255)
                        else:
                            val_row.append(val_pngs[idx])
                            val_row.append(
                                val_pngs[idx + 1][..., np.newaxis].repeat(3, axis=-1))
                            val_row.append(val_pngs[idx + 2])
                    val_res.append(np.concatenate(val_row, axis=1))
                wh = None
                for m in range(samples_per_row):
                    if m == 0:
                        wh = (val_res[m].shape[1], val_res[0].shape[0])
                    else:
                        assert wh is not None
                        val_res[m] = cv2.resize(
                            val_res[m], wh, interpolation=cv2.INTER_AREA)
                val_res = np.concatenate(val_res, axis=0)
                png_v = join(
                    model_folder,
                    "results",
                    "val_iter%07d_res.png" %
                    iter_counter)
                Image.fromarray(val_res.astype(np.uint8)).save(png_v)

                if config.log:
                    wandb.log({"Edge map Val loss": edge_loss_val_total /
                              val_avg_acc_count}, step=iter_counter)
                    wandb.log({"Edge Val ACC left (pos)": val_acc_pos_x /
                              val_avg_acc_count}, step=iter_counter)
                    wandb.log({"Edge Val ACC up (pos)": val_acc_pos_y /
                              val_avg_acc_count}, step=iter_counter)
                    wandb.log({"Edge Val ACC both (pos)": val_acc_pos_a /
                              val_avg_acc_count}, step=iter_counter)
                    wandb.log({"Edge Val ACC left (neg)": val_acc_neg_x /
                              val_avg_acc_count}, step=iter_counter)
                    wandb.log({"Edge Val ACC up (neg)": val_acc_neg_y /
                              val_avg_acc_count}, step=iter_counter)
                    wandb.log({"Edge Val ACC both (neg)": val_acc_neg_a /
                              val_avg_acc_count}, step=iter_counter)
                    if args.keypt_dual or args.keypt:
                        wandb.log(
                            {
                                "Keypoint map Val loss (dist)": keypt_loss_val_total /
                                val_avg_acc_count},
                            step=iter_counter)
                    if config.valence:
                        wandb.log(
                            {"Valence Val loss": loss_val_v_total / val_avg_acc_count}, step=iter_counter)
                    # if config.usm:
                    #     wandb.log({"USM Val loss": loss_val_usm_total / val_avg_acc_count}, step = iter_counter)
                    if config.skel:
                        wandb.log(
                            {"Skel Val loss": skel_loss_val_total / val_avg_acc_count}, step=iter_counter)

                    wandb.log({"Validation Result": wandb.Image(
                        val_res)}, step=iter_counter)
                    wandb.log({"Train Result": wandb.Image(
                        train_res)}, step=iter_counter)

    if DEBUG:
        start_epoches(
            iter_counter,
            iter_counter_val,
            config,
            epoch_start,
            train_loader,
            val_loader,
            model_folder)
    else:
        # for real training
        try:
            start_epoches(
                iter_counter,
                iter_counter_val,
                config,
                epoch_start,
                train_loader,
                val_loader,
                model_folder)
        except Exception as e:
            print(e)
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss_dict['all'],
                'param': vars(args)}, join(model_folder, "exception_saved.pth"))


def log_loss_ndc(config, loss_dict, iter_counter, acc_res_list):
    acc_pos_x, acc_neg_x, acc_pos_y, acc_neg_y, acc_pos_xy, acc_neg_xy = acc_res_list
    wandb.log({"Edge map loss": loss_dict['edge']}, step=iter_counter)
    wandb.log({"Edge ACC left (pos)": acc_pos_x}, step=iter_counter)
    wandb.log({"Edge ACC up (pos)": acc_pos_y}, step=iter_counter)
    wandb.log({"Edge ACC both (pos)": acc_pos_xy}, step=iter_counter)
    wandb.log({"Edge ACC left (neg)": acc_neg_x}, step=iter_counter)
    wandb.log({"Edge ACC up (neg)": acc_neg_y}, step=iter_counter)
    wandb.log({"Edge ACC both (neg)": acc_neg_xy}, step=iter_counter)
    if config.keypt or config.keypt_dual:
        wandb.log(
            {"Keypoint map loss (dist)": loss_dict['keypt']}, step=iter_counter)
    if config.valence:
        wandb.log({"Valence loss": loss_dict['valence']}, step=iter_counter)
    # if config.usm:
    #     wandb.log({"USM loss": loss_dict['usm']}, step = iter_counter)
    if config.skel:
        wandb.log({"Skeleton loss": loss_dict['skel']}, step=iter_counter)


def unpad_keypt_dict(keypts_dicts, idx):
    unpadded_dict = {}
    for key in keypts_dicts:
        valide_nums = (keypts_dicts[key][idx] != -1).sum()
        assert valide_nums % 2 == 0
        valide_nums = int(valide_nums / 2)
        try:
            unpad_keypt = keypts_dicts[key][idx][:valide_nums, ...].numpy()
        except BaseException:
            unpad_keypt = keypts_dicts[key][idx][:valide_nums, ...]
        unpadded_dict[key] = unpad_keypt
    return unpadded_dict


def vis_train_ndc(
        config,
        pres,
        gts,
        model_folder,
        iter_counter,
        gsizes,
        save=True,
        img_num_total=16,
        img_num_row=8,
        keypts_dicts_gt=None,
        keypts_dicts_pre=None):
    train_pngs = [[], [], [], []]
    edge_maps_pre, pt_map_pre, usm_pre = pres
    edge_maps_gt, pt_map_gt, skel_gt, usm_gt = gts
    # edge_maps_pre, pt_map_pre = pres
    # edge_maps_gt, pt_map_gt, skel_gt = gts

    # generate the svg from gt and prediction
    edge_maps_gt = edge_maps_gt.permute(0, 2, 3, 1)
    pt_map_gt = pt_map_gt.permute(0, 2, 3, 1)
    edge_maps_pre_xy = edge_maps_pre.permute(0, 2, 3, 1)[..., :-1]
    edge_maps_pre_z = edge_maps_pre.permute(0, 2, 3, 1)[..., -1]
    pt_map_pre = pt_map_pre.permute(0, 2, 3, 1)
    skel_pre = torch.sigmoid(edge_maps_pre_z) > 0.5
    # usm_pre = torch.sigmoid(usm_pre) > 0.5

    # get image ready and push them into stack
    for i in range(min(img_num_total, len(edge_maps_gt))):
        # get lines from GT
        # if keypts_dicts_gt is not None:
        gsize = float(gsizes[i])
        if usm_gt is not None:
            keypts_dict = unpad_keypt_dict(keypts_dicts_gt, i)
            edge_map_x, edge_map_y, _ = pre_to_map(edge_maps_gt[i])
            edge_map_gt = torch.stack((edge_map_x, edge_map_y), dim=-1)
            _, lines_map_x, lines_map_y, _ = map_to_lines(
                edge_map_gt, pt_map_gt[i], True)
            try:
                lines = refine_topology(
                    edge_map_gt,
                    pt_map_gt[i],
                    usm_gt[i],
                    lines_map_x,
                    lines_map_y,
                    keypts_dict,
                    True)
            except BaseException:
                lines = pre_to_lines(edge_maps_gt[i], pt_map_gt[i])
        else:
            lines = pre_to_lines(edge_maps_gt[i], pt_map_gt[i])

        # get lines from Pre
        if config.keypt_only:
            edge_map_pre = edge_maps_gt[i]
        else:
            edge_map_pre = edge_maps_pre_xy[i]

        # This branch is for the end to end stage, DO NOT remove it
        if usm_pre is not None:
            keypts_dict = unpad_keypt_dict(keypts_dicts_pre, i)
            edge_map_x_pre, edge_map_y_pre, _ = pre_to_map(edge_map_pre)
            edge_map_pre_ = torch.stack(
                (edge_map_x_pre, edge_map_y_pre), dim=-1)
            _, lines_map_x_pre, lines_map_y_pre, _ = map_to_lines(
                edge_map_pre_, pt_map_pre[i], True)
            try:
                lines_pre = refine_topology(
                    edge_map_pre_,
                    pt_map_pre[i],
                    usm_pre[i],
                    lines_map_x_pre,
                    lines_map_y_pre,
                    keypts_dict,
                    True)
            except BaseException:
                lines_pre = pre_to_lines(edge_map_pre_, pt_map_pre[i])
        else:
            lines_pre = pre_to_lines(edge_map_pre, pt_map_pre[i])

        _, canvas_h, canvas_w, _ = edge_maps_gt.shape
        canvas_h = canvas_h * gsize
        canvas_w = canvas_w * gsize

        # debug
        # h, w = edge_map_gt.shape[0], edge_map_gt.shape[1]
        # lines_to_svg(lines_pre, w, h, "test.svg", 'xy')

        if exists(join(model_folder, "results")) == False:
            os.makedirs(join(model_folder, "results"))
        svg_pre = join(
            model_folder, "results", "train_iter%07d_%02d_pre.svg" %
            (iter_counter, i))

        if lines_pre is not None:
            lines_pre_flag = lines_to_svg(
                lines_pre, canvas_w, canvas_h, svg_pre)
        else:
            lines_pre_flag = False

        svg_gt = join(
            model_folder, "results", "train_iter%07d_%02d_gt.svg" %
            (iter_counter, i))

        if lines is not None:
            lines_flag = lines_to_svg(lines, canvas_w, canvas_h, svg_gt)
        else:
            lines_flag = False
        # skel_pre_np = (sigmoid(edge_maps_pre[i][..., -1]) > 0.5).detach().cpu().squeeze().numpy().astype(int)
        skel_pre_np = skel_pre[i].detach().cpu().squeeze().numpy().astype(int)
        skel_gt_np = skel_gt[i].detach().cpu().squeeze().numpy().astype(int)
        skel_np = blend_skeletons(skel_gt_np, skel_pre_np, alpha=0.5)
        # usm_pre_np = usm_pre[i].detach().cpu().squeeze().numpy().astype(int)
        # usm_gt_np = usm_gt[i].detach().cpu().squeeze().numpy().astype(int)
        train_pngs[0].append(skel_np)
        if lines_pre_flag:
            png_np = svg_to_numpy(svg_pre)
        else:
            png_np = None
        if lines_pre_flag and png_np is not None:
            train_pngs[1].append(png_np)
        else:
            train_pngs[1].append(
                (np.ones(
                    (skel_np.shape[0],
                     skel_np.shape[1])) *
                    255).astype(
                    np.uint8))
        if lines_flag:
            png_np = svg_to_numpy(svg_gt)
            if usm_pre is not None and usm_gt is not None:
                png_np = blend_skeletons(
                    png_np,
                    (usm_pre[i].astype(int).squeeze(),
                     usm_gt[i].astype(int).squeeze()),
                    usm_mode=True)
            else:
                png_np = np.repeat(png_np[..., np.newaxis], 3, axis=-1)
        else:
            png_np = None
        if lines_flag and png_np is not None:
            train_pngs[2].append(png_np)
        else:
            train_pngs[2].append(
                (np.ones(
                    (skel_np.shape[0],
                     skel_np.shape[1])) *
                    255).astype(
                    np.uint8))

    # write training result
    png_t = join(
        model_folder,
        "results",
        "train_iter%07d_res.png" %
        iter_counter)
    t_nums = len(train_pngs[0]) // img_num_row

    if (len(train_pngs[0]) % img_num_row) != 0:
        t_nums += 1
    wh = to_target_wh(train_pngs[0][0])
    train_res = []

    # concatnate images for final visualization
    for i in range(0, int(t_nums * img_num_row), img_num_row):
        udfs = []
        usms = []
        gts = []
        pres = []
        for j in range(img_num_row):
            idx = i + j
            if idx < len(train_pngs[0]):
                skel_np = train_pngs[0][idx]
                skel_np = cv2.resize(
                    skel_np, wh, interpolation=cv2.INTER_NEAREST)
                skel_np = add_title(skel_np, "Skeleton map")
                udfs.append(skel_np)
                gts_np = train_pngs[2][idx]
                gts.append(
                    add_title(
                        cv2.resize(
                            gts_np,
                            wh,
                            interpolation=cv2.INTER_AREA),
                        "GT vector + USM pre"))
                pres_np = train_pngs[1][idx]
                pres.append(
                    add_title(
                        cv2.resize(
                            pres_np,
                            wh,
                            interpolation=cv2.INTER_AREA),
                        "Pre vector"))
                # usm_np = train_pngs[3][idx]
                # usms.append(cv2.resize(usm_np, wh, interpolation = cv2.INTER_NEAREST))
            else:
                wh_udf = (wh[1], wh[0], 3)
                udfs.append(np.ones(wh_udf) * 255)
                gts.append(np.ones(wh_udf) * 255)
                pres.append(np.ones(wh) * 255)
                # usms.append(np.ones(wh_udf) * 255)
        udfs = np.concatenate(udfs, axis=1)
        # usms = np.concatenate(usms, axis = 1)
        gts = np.concatenate(gts, axis=1)
        pres = np.concatenate(pres, axis=1)[..., np.newaxis].repeat(3, axis=-1)
        train_res.append(np.concatenate((udfs, gts, pres), axis=0))
    train_res = np.concatenate(train_res, axis=0)

    if save:
        Image.fromarray(train_res.astype(np.uint8)).save(png_t)
    else:
        if exists(svg_pre):
            os.remove(svg_pre)
        if exists(svg_gt):
            os.remove(svg_gt)
    return train_res


def compute_ndc_loss_all(
        config,
        edge_maps_pre,
        edge_maps_gt,
        edge_mask,
        pt_map_pre,
        pt_map_gt,
        acc_list=None):
    if acc_list is None:
        skip_acc = True
    else:
        train_acc_pos_x, train_acc_pos_y, train_acc_pos_a, train_acc_neg_x, train_acc_neg_y, train_acc_neg_a, train_avg_acc_count = acc_list
        skip_acc = False

    loss_dict = {}
    # compute the loss
    if config.keypt_only:
        loss_all = 0
        edge_loss = 0
        edge_loss_z = 0
        acc_x_pos = Tensor([1.])
        acc_x_neg = Tensor([1.])
        acc_y_pos = Tensor([1.])
        acc_y_neg = Tensor([1.])
        acc_a_pos = Tensor([1.])
        acc_a_neg = Tensor([1.])
    else:
        edge_loss, edge_loss_z, acc_x_pos, acc_x_neg, acc_y_pos, acc_y_neg, acc_a_pos, acc_a_neg =\
            loss_edge(edge_maps_pre, edge_maps_gt, edge_mask, focal=config.focal, alpha=True,
                      review=config.review, review_all=config.review_all)
        loss_all = edge_loss
        loss_dict['edge'] = edge_loss

    if config.skel:
        skel_loss = edge_loss_z
        loss_all = loss_all + 0.001 * skel_loss
        loss_dict['skel'] = skel_loss
    else:
        skel_loss = 0

    # compute the valence loss
    # we will record the valence loss value anyway
    loss_v = valence_loss(
        edge_maps_pre,
        edge_maps_gt,
        edge_mask,
        method=config.method)
    if config.valence:
        if config.method == 'per-grid':
            loss_all = loss_all + 0.005 * loss_v
        else:
            loss_all = loss_all + 1e-5 * loss_v
        loss_dict['valence'] = loss_v
    else:
        loss_dict['valence'] = loss_v

    # compute the key point loss (l2 loss)
    keypt_loss = loss_keypt(pt_map_pre, pt_map_gt, edge_maps_gt)

    if config.keypt:
        loss_dict['keypt'] = keypt_loss
        loss_all = loss_all + 0.5 * keypt_loss
    else:
        pt_map_pre.copy_(pt_map_gt)

    loss_dict['all'] = loss_all

    if skip_acc:
        acc_list = None
        acc_res_list = None
    else:
        # record loss and acc
        train_acc_pos_x += acc_x_pos.data
        train_acc_neg_x += acc_x_neg.data
        train_acc_pos_y += acc_y_pos.data
        train_acc_neg_y += acc_y_neg.data
        train_acc_pos_a += acc_a_pos.data
        train_acc_neg_a += acc_a_neg.data
        train_avg_acc_count += 1

        acc_pos_x = train_acc_pos_x / train_avg_acc_count
        acc_pos_y = train_acc_pos_y / train_avg_acc_count
        acc_pos_xy = train_acc_pos_a / train_avg_acc_count
        acc_neg_x = train_acc_neg_x / train_avg_acc_count
        acc_neg_y = train_acc_neg_y / train_avg_acc_count
        acc_neg_xy = train_acc_neg_a / train_avg_acc_count

        acc_list = [
            train_acc_pos_x,
            train_acc_pos_y,
            train_acc_pos_a,
            train_acc_neg_x,
            train_acc_neg_y,
            train_acc_neg_a,
            train_avg_acc_count]
        acc_res_list = [
            acc_pos_x,
            acc_pos_y,
            acc_pos_xy,
            acc_neg_x,
            acc_neg_y,
            acc_neg_xy]

    return loss_dict, acc_list, acc_res_list


def to_target_wh(img, width=None, height=1500, reverse=False):
    # compute the target width and height for image resize
    h, w = img.shape[0], img.shape[1]
    if width is None and height is not None:
        ratio = height / h
    elif width is not None and height is None:
        ratio = width / h
    else:
        ratio = 1
    if reverse:
        return (int(h * ratio + 0.5), int(w * ratio + 0.5))
    else:
        return (int(w * ratio + 0.5), int(h * ratio + 0.5))


def parse():
    parser = argparse.ArgumentParser(description='Neural Dual Contouring 2D')
    '''
    Data parameters
    '''
    parser.add_argument('--ds', metavar='dataset', type=str,
                        help='root path of dataset', default="./data/full")
    parser.add_argument('--bs', metavar='batch size', type=int,
                        help='batch size', default=1)
    parser.add_argument(
        '--workers',
        metavar='dataloader workers',
        type=int,
        help='the number of train dataloader worker number',
        default=0)
    parser.add_argument('--nl', metavar='noise level', type=float,
                        help='the noise level for self test mode', default=0.0)
    parser.add_argument('--mgs', action="store_true",
                        help="enable multiple grid size for data augmentation")
    parser.add_argument('--approx_udf', action="store_true",
                        help="flag for paper experiment need, DON'T enable it")
    '''
    Model
    '''
    parser.add_argument('--cp', metavar='checkpoint', type=str,
                        help='path to checkpoint', default="./models_ndc")
    '''
    Training
    '''
    parser.add_argument('--debug', action="store_true",
                        help="enable debug mode")
    parser.add_argument(
        '--epoch',
        metavar='total epoches',
        type=int,
        help='the epoch numbers for the whole training',
        default=100)
    parser.add_argument(
        '--lr',
        metavar='initial learning rate',
        type=float,
        help='the start learning rate for training',
        default=1e-4)
    parser.add_argument('--log', action="store_true", help="enable wandb log")
    parser.add_argument(
        '--sch',
        action="store_true",
        help="enable optimizer scheduler")
    parser.add_argument('--keypt', action="store_true",
                        help="enable predicting the key point output in the same model")
    parser.add_argument('--keypt_dual', action="store_true",
                        help="enable predicting the key point output in a separate model")
    parser.add_argument('--mask', action="store_true",
                        help="enable mask weighting on the loss computation")
    parser.add_argument(
        '--review',
        action="store_true",
        help="review wrong predictions and learn them again in each iteration")
    parser.add_argument(
        '--review_all',
        action="store_true",
        help="review wrong predictions and learn them again in each iteration")
    parser.add_argument('--keypt_only', action="store_true",
                        help="only train the key point loss")
    parser.add_argument('--skel', action="store_true",
                        help="enable skeleton loss")
    parser.add_argument('--insert_skel', action="store_true",
                        help="insert skel to UDF")
    parser.add_argument(
        '--name',
        metavar='train task name',
        type=str,
        help='the name that is shown in the Wandb UI',
        default=None)
    parser.add_argument(
        '--valence',
        action="store_true",
        help="use valence loss to improve the stroke prediction quality")
    parser.add_argument(
        '--method',
        metavar='valence loss method',
        type=str,
        help='define the way of how to compute the valence loss',
        default="valence")
    parser.add_argument('--focal', action="store_true",
                        help="enable focal loss")
    parser.add_argument('--adptive', action="store_true",
                        help="enable adpative masked loss")
    parser.add_argument('--noisy', action="store_true",
                        help="use larger dual contouring grid")
    '''
    Network
    '''
    parser.add_argument(
        '--dist_clip',
        metavar='threshold distance',
        type=float,
        help='the threshold for distance values',
        default=4.5)
    parser.add_argument('--dist_ndc', metavar='threshold distance', type=float,
                        help='the threshold for generating masks', default=1.5)
    parser.add_argument('--ln', action="store_true",
                        help="use larger net for training")
    parser.add_argument('--bn', action="store_true",
                        help="enable batch normalization for training")
    parser.add_argument('--do', action="store_true",
                        help="add drop out layer in the network")
    parser.add_argument('--lk', action="store_true",
                        help="use large kernel for the convolution layers")
    parser.add_argument('--coord', action="store_true",
                        help="enable coodr convlution")
    parser.add_argument('--dims', metavar='dataloader workers', type=int,
                        help='the channel number of the network', default=64)
    parser.add_argument(
        '--layers',
        metavar='network structure',
        nargs="+",
        type=int,
        help='define the number of layers of the network (the default should be 2, 6, 2)',
        default=[
            3,
            6,
            2])
    parser.add_argument('--msb', action="store_true",
                        help="enable multiple scale UDF branch ")
    parser.add_argument('--resnext_i', action="store_true",
                        help="enable multiple scale UDF branch ")
    parser.add_argument('--resnext_f', action="store_true",
                        help="enable multiple scale UDF branch ")
    args = parser.parse_args()

    return args


if __name__ == "__main__":
    __spec__ = None  # we need this line when using pdb
    torch.multiprocessing.set_start_method('spawn')
    train()
