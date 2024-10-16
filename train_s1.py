
import cv2
import torch
import wandb
import os
import argparse
import numpy as np
import torch.optim as optim
import torchvision.transforms as T

from tqdm import tqdm
from datetime import datetime
from PIL import Image
from types import SimpleNamespace
from torch.nn.functional import normalize
from os.path import join, exists

from network.keypoint import PyramidalNet
from dataset.s1_sketchy import SketchyDataset
from network.udc import CNN_2d_resnet
from train_s2 import vis_train_ndc
from train_s2 import compute_ndc_loss_all, init_base_coord
from utils.losses import dist_loss
from utils.keypt_tools import plot_udf_coord_numpy, vis_batch

DEBUG = True

# create dataloader
# https://github.com/fastai/fastbook/issues/85
# when test on windows worker number should always be 0


class AddGaussianNoise(object):
    def __init__(self, mean=0., std=1., device="cpu"):
        self.std = std
        self.mean = mean
        self.device = device

    def __call__(self, tensor):
        return tensor + (torch.randn(tensor.size()) *
                         self.std + self.mean).to(tensor.device)

    def __repr__(self):
        return self.__class__.__name__ + \
            '(mean={0}, std={1})'.format(self.mean, self.std)


def load_data(
        root,
        batch_size,
        num_workers=0,
        patch_size=256,
        is_train=True,
        dist_pt=3,
        approx_udf=False,
        no_rotation=False,
        insert_skel=False,
        dist_clip=8.5,
        jpg=False,
        up_scale=False,
        bg=False,
        device=None):
    # https://pytorch.org/vision/stable/auto_examples/plot_transforms.html#sphx-glr-auto-examples-plot-transforms-py
    transform_pixel = T.Compose([
        T.RandomAutocontrast(),
        T.ColorJitter(brightness=0.15),
    ])

    data_set = SketchyDataset(
        root,
        patch_size=patch_size,
        is_train=is_train,
        bg=bg,
        insert_skel=insert_skel,
        transform_pixel=transform_pixel,
        dist_pt=dist_pt,
        no_rotation=no_rotation,
        dist_clip=dist_clip,
        up_scale=up_scale,
        jpg=jpg,
        approx_udf=approx_udf,
        device=device)
    data_loader = torch.utils.data.DataLoader(
        data_set,
        batch_size=batch_size,
        shuffle=True if is_train else False,
        num_workers=num_workers)
    return data_loader


def train():
    args = parse()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dist_pt = 6 if args.up_scale else args.dist_pt
    # add NDC network for end-to-end training
    if args.ndc_add:
        # adding NDC network means we should have a pretrained UDF network
        # load pretrained UDF network
        assert args.udf is not None
        model_path_udf = join(args.fcp, args.udf)
        assert exists(model_path_udf)
        print("Log\tloading %s" % model_path_udf)
        ckpt_udf = torch.load(model_path_udf, map_location="cpu")
        args_udf = vars(ckpt_udf['param'])
        if "hourglass_input_channels" not in args_udf:
            args_udf["hourglass_input_channels"] = args.hourglass_input_channels
        if "hourglass_channels" not in args_udf:
            args_udf["hourglass_channels"] = args.hourglass_channels
        if "downsample_depth" not in args_udf:
            args_udf["downsample_depth"] = args.downsample_depth
        if "cardinality" not in args_udf:
            args_udf["cardinality"] = args.cardinality
        if "backbone" not in args_udf:
            args_udf["backbone"] = args.backbone
        if "width" not in args_udf:
            args_udf["width"] = args.width
        args_udf = SimpleNamespace(**args_udf)

        # load NDC pretrained model
        '''
            we still always load NDC model first
            but it doesn't mean we will use its parameters. By loading
            args from a NDC model, we can only read the network configs
            for NDC intialization then train it from scratch
        '''
        assert args.ndc is not None
        model_path_ndc = join(args.fcp, args.ndc)
        assert exists(model_path_ndc)
        print("Log\tloading %s" % model_path_ndc)
        ckpt_ndc = torch.load(model_path_ndc, map_location="cpu")
        try:
            args_ndc = vars(ckpt_ndc['param'])
        except BaseException:
            args_ndc = ckpt_ndc['param']
            assert isinstance(args_ndc, dict)
        args_ndc['review_all'] = args.review_all
        if "noisy" not in args_ndc:
            args_ndc['noisy'] = False
        if "resnext_f" not in args_ndc:
            args_ndc['resnext_f'] = False
        if "resnext_i" not in args_ndc:
            args_ndc['resnext_i'] = False
        if isinstance(args_ndc, dict):
            args_ndc = SimpleNamespace(**args_ndc)
        # initial NDC network
        net_ndc = CNN_2d_resnet(
            device,
            channels=args_ndc.dims,
            layers=args_ndc.layers,
            multi_scale=args_ndc.msb,
            batch_norm=args_ndc.bn,
            noisy=args_ndc.noisy,
            resnext_feature=args_ndc.resnext_f,
            resnext_input=args_ndc.resnext_i)
        if not args.ndc_scratch:
            print("log:\tloading pretrained parameters for NDC network")
            net_ndc.load_state_dict(ckpt_ndc['model_state_dict'])
        net_ndc = net_ndc.to(device)
        # overwrite some parameters from loaded model to overwrite current
        # args, or vice versa
        assert args_udf.dist_clip >= args_ndc.dist_clip
        try:
            # overwrite this option if we need to enable review mode
            if args.review:
                args_ndc.review = args.review
            args.dist_mode = args_udf.dist_mode
            args.dist_clip = args_udf.dist_clip
            args.insert_skel = args_udf.insert_skel
            args.dist_pt = args_udf.dist_pt
            args.lr = args_udf.lr
            args.up_scale = args_udf.up_scale
            args.jpg = args_udf.jpg
            args.hourglass_channels = args_udf.hourglass_channels
            args.hourglass_input_channels = args_udf.hourglass_input_channels
            args.downsample_depth = args_udf.downsample_depth
            args.cardinality = args_udf.cardinality
            args.paper_background = args_udf.paper_background
            args.backbone = args_udf.backbone
            args.width = args_udf.width
            args_ndc.review_all = args.review_all
        except Exception as e:
            print("Warning:\tgot error when transfering the network configurations, please double check if everything goes right")
            print(str(e))

    # load data
    no_rotation = args.ndc_add or args.no_rotation
    train_loader = load_data(
        args.ds,
        args.bs,
        patch_size=args.ps,
        insert_skel=args.insert_skel,
        dist_pt=args.dist_pt,
        dist_clip=args.dist_clip,
        num_workers=args.workers,
        up_scale=args.up_scale,
        bg=args.paper_background,
        is_train=True,
        no_rotation=no_rotation,
        jpg=args.jpg,
        approx_udf=args.approx_udf,
        device=device)
    val_loader = load_data(
        args.ds,
        args.bs,
        patch_size=args.ps,
        insert_skel=args.insert_skel,
        dist_pt=args.dist_pt,
        dist_clip=args.dist_clip,
        num_workers=args.workers,
        up_scale=args.up_scale,
        bg=args.paper_background,
        is_train=False,
        no_rotation=no_rotation,
        jpg=args.jpg,
        approx_udf=args.approx_udf,
        device=device)

    # init network
    n_classes = 6  # topology, all points, end point, sharp turn, junctions, usm
    net = PyramidalNet(
        n_classes=n_classes,
        up_scale=args.up_scale,
        hourglass_channels=args.hourglass_channels,
        hourglass_input_channels=args.hourglass_input_channels,
        downsample_depth=args.downsample_depth,
        cardinality=args.cardinality,
        back_bone=args.backbone,
        width=args.width)
    net = net.to(device)

    # load pretrained UDF network
    if args.udf_scratch == False and args.ndc_add:
        print("log:\tloading pretrained parameters for UDF network")
        net.load_state_dict(ckpt_udf['model_state_dict'])

    # init optimizer, scheduler
    '''
        We probably don't need to decay the learning rate when using Adam
        https://stackoverflow.com/questions/39517431/should-we-do-learning-rate-decay-for-adam-optimizer_udf
    '''
    if args.ndc_add:
        udf_lr = args.lr
        optimizer_udf = optim.Adam(
            net.parameters(), lr=udf_lr, weight_decay=1e-8)
        if not args.udf_scratch:
            optimizer_udf.load_state_dict(ckpt_udf["optimizer_state_dict"])
        ndc_lr = args_ndc.lr
        optimizer_ndc = optim.Adam(
            net_ndc.parameters(),
            lr=ndc_lr,
            weight_decay=1e-8)
        if not args.ndc_scratch:
            optimizer_ndc.load_state_dict(ckpt_ndc["optimizer_state_dict"])
    else:
        optimizer_udf = optim.Adam(
            net.parameters(),
            lr=args.lr,
            weight_decay=1e-8)

    # load last check point if exists
    model_path_base = "./models"
    now = datetime.now()
    dt_formatted = now.strftime("D%Y-%m-%dT%H-%M-%S")
    model_folder = join(model_path_base, dt_formatted)
    if args.name is not None:
        model_folder = model_folder + "-" + args.name
    os.makedirs(model_folder)
    if args.cp == "":
        args.cp = model_path_base
    model_path = join(model_folder, "last_epoch_udf.pth")

    epoch_start = 0
    # try to load last checkpoint
    model_path_load = join(args.cp, "last_epoch_udf.pth")
    model_path_ndc_load = model_path_load.replace('udf.pth', 'ndc.pth')
    if exists(model_path_load):
        print("Log\tloading %s" % model_path_load)
        checkpoint = torch.load(model_path_load, map_location="cpu")
        net.load_state_dict(checkpoint['model_state_dict'])
        epoch_start = checkpoint['epoch']
        optimizer_udf.load_state_dict(checkpoint["optimizer_state_dict"])
    if args.ndc_add and exists(model_path_ndc_load):
        print("Log\tloading %s" % model_path_ndc_load)
        checkpoint = torch.load(model_path_ndc_load, map_location="cpu")
        net_ndc.load_state_dict(checkpoint['model_state_dict'])
        optimizer_ndc.load_state_dict(checkpoint["optimizer_state_dict"])

    # inital wandb
    if args.log:
        if args.usr == "your_wandb_usr_name":
            raise ValueError("please set up your wandb user name by --usr first!")
        wandb.init(
            project="Deep Sketch Vectorization",
            # entity="waterheater",
            entity = args.usr,
            name=args.name)
        wandb.config = {
            "learning_rate": args.lr,
            "epochs": args.epoch,
            "batch_size": args.bs
        }

    # start train
    iter_counter = 0  # we use a global iter_counter for iterations

    def start_epoches(
            iter_counter,
            args,
            epoch_start,
            train_loader,
            model_folder):
        xs = []
        ls_topo = []
        ls_p_all = []
        ls_p_end = []
        ls_p_sharp = []
        ls_p_T = []
        ls_p_X = []
        ls_p_star = []

        ys_p_end = []
        ys_r_end = []
        ys_p_sharp = []
        ys_r_sharp = []
        ys_p_T = []
        ys_r_T = []
        ys_p_X = []
        ys_r_X = []
        ys_p_star = []
        ys_r_star = []

        ls_topo_val = []
        ls_p_all_val = []
        ls_p_end_val = []
        ls_p_sharp_val = []
        ls_p_T_val = []
        ls_p_X_val = []
        ls_p_star_val = []

        ys_p_end_val = []
        ys_r_end_val = []
        ys_p_sharp_val = []
        ys_r_sharp_val = []
        ys_p_T_val = []
        ys_r_T_val = []
        ys_p_X_val = []
        ys_r_X_val = []
        ys_p_star_val = []
        ys_r_star_val = []

        vis_every_iter = 3000
        img_vis_total = 8 if args.bs > 8 else args.bs

        # start training epoch
        for i in range(args.epoch - epoch_start):
            epoch = i + epoch_start
            print("Log:\tstarting epoch %d" % epoch)

            if args.ndc_add:
                if args.ndc_fix:
                    print("log:\tfreezing NDC network")
                    for para in net_ndc.parameters():
                        para.requires_grad = False
                    net_ndc.eval()
                else:
                    for para in net_ndc.parameters():
                        para.requires_grad = True
                    if args.eval:
                        print("log:\ttraining NDC network with eval() mode")
                        net_ndc.eval()
                    else:
                        print("log:\ttraining NDC network")
                        net_ndc.train()

            if args.udf_fix:
                print("log:\tfreezing UDF network")
                for para in net.parameters():
                    para.requires_grad = False
                net.eval()
            else:
                for para in net.parameters():
                    para.requires_grad = True
                if args.ndc_add or args.eval:
                    print("log:\ttraining UDF network with eval() mode")
                    net.eval()  # for debug, this branch is NOT normal and should be a logical bug
                else:
                    print("log:\ttraining UDF network")
                    net.train()

            p_bar = tqdm(train_loader, ncols=175)
            for data in p_bar:
                # read data
                img = data['img']
                udf, udf_all_pt, udf_end_pt, udf_sharp_pt, udf_junc_pt = data['udfs']
                mask_topo, pt_mask1, mask_all_pt, mask_end_pt, mask_sharp_pt, mask_junc_pt = data[
                    'udf_masks']
                gt_all_pt, gt_end_pt, gt_sharp_pt, gt_junc_pt = data['udf_gts']
                edge_maps_gt, pt_map_gt, udf_usm_gt, skel_gt = data['ndc_gts']
                edge_mask, usm_mask = data['ndc_mask']
                gsize = 0.5 if args.up_scale else 1
                # all image should have the same grid size
                gsizes = [gsize] * len(img)
                # predict UDF
                if args.udf_fix:
                    with torch.no_grad():
                        udf_topo_pre, udfs_pre = net.forward_with_downscale_outs(
                            img)
                else:
                    udf_topo_pre, udfs_pre = net.forward_with_downscale_outs(
                        img)

                # compute loss
                # topology, all points, end point, sharp turn, junctions, usm
                udf_all_pre, udf_end_pre, udf_sharp_pre, udf_junc_pre, udf_usm_pre = udfs_pre
                pres_train = [
                    udf_topo_pre,
                    udf_all_pre,
                    udf_end_pre,
                    udf_sharp_pre,
                    udf_junc_pre,
                    udf_usm_pre]
                gts_train = [
                    udf,
                    udf_all_pt,
                    udf_end_pt,
                    udf_sharp_pt,
                    udf_junc_pt,
                    udf_usm_gt]
                masks_train = [
                    mask_topo,
                    mask_all_pt,
                    mask_end_pt,
                    mask_sharp_pt,
                    mask_junc_pt,
                    usm_mask,
                    pt_mask1]
                if iter_counter % vis_every_iter == 0:
                    loss_udf, losses = loss_total(
                        pres_train, gts_train, masks_train, to_numpy_inplace=True, dist_mode=args.dist_mode)
                else:
                    loss_udf, losses = loss_total(
                        pres_train, gts_train, masks_train, to_numpy_inplace=False, dist_mode=args.dist_mode)

                # add second network training or testing
                if args.ndc_add:
                    base_coord = init_base_coord(pt_map_gt, device, gsizes)
                    pt_map_gt = pt_map_gt - base_coord

                    if args_udf.dist_clip != args_ndc.dist_clip:
                        udf_topo_pre = (
                            udf_topo_pre * args_udf.dist_clip).clip(0, args_ndc.dist_clip) / args_ndc.dist_clip

                    # predict edge maps
                    if args.ndc_fix:
                        with torch.no_grad():
                            edge_maps_pre, pt_map_pre = net_ndc.forward(
                                udf_topo_pre, gsizes)
                    else:
                        edge_maps_pre, pt_map_pre = net_ndc.forward(
                            udf_topo_pre, gsizes)

                    # compute loss
                    loss_dict, _, _ = compute_ndc_loss_all(
                        args_ndc, edge_maps_pre, edge_maps_gt, edge_mask, pt_map_pre, pt_map_gt)
                    loss_ndc = loss_dict['all']
                    pt_map_pre = pt_map_pre + base_coord
                    pt_map_gt = pt_map_gt + base_coord

                # backpropagate the gradient
                if args.udf_fix and args.ndc_fix == False:
                    assert args.ndc_add
                    loss = loss_ndc
                    optimizer_ndc.zero_grad()
                    loss.backward()
                    optimizer_ndc.step()
                elif args.udf_fix == False and args.ndc_fix:
                    assert args.ndc_add
                    optimizer_udf.zero_grad()
                    loss = loss_udf + 0.1 * loss_ndc
                    loss.backward()
                    optimizer_udf.step()
                elif args.udf_fix == False and args.ndc_fix == False and args.ndc_add:
                    optimizer_ndc.zero_grad()
                    optimizer_udf.zero_grad()
                    loss = loss_udf + loss_ndc
                    loss.backward()
                    optimizer_udf.step()
                    optimizer_ndc.step()
                else:
                    optimizer_udf.zero_grad()
                    loss = loss_udf
                    loss.backward()
                    optimizer_udf.step()

                # display the loss value on progress bar
                if args.ndc_add:
                    p_bar.set_description(
                        "UDF: %.4f, NDC: %.4f, Topo: %.4f, Edge: %.4f, Keypt: %.4f, Valence: %.4f, USM: %.4f" %
                        (loss_udf.item(),
                         loss_ndc.item(),
                            losses[0],
                            loss_dict['edge'].item(),
                            loss_dict['keypt'].item(),
                            loss_dict['valence'].item(),
                            losses[5].item()))
                else:
                    p_bar.set_description(
                        "Topo: %.4f, All: %.4f, End: %.4f, Sharp: %.4f, Junc: %.4f, USM: %.4f" %
                        (losses[0], losses[1], losses[2], losses[3], losses[4], losses[5]))

                # visualize the training
                if iter_counter % vis_every_iter == 0:
                    dist_clip = args.dist_clip
                    # let's visulize all prediction results
                    pres_train[0] = pres_train[0][:, :, :-1, :-1]
                    gts_train[0] = gts_train[0][:, :, :-1, :-1]
                    masks_train[0] = masks_train[0][..., :-1, :-1]
                    prediction = pres_train[:1]

                    gts = gts_train[:1]
                    if args.ndc_add:
                        res_topo, _, _ = vis_batch(
                            img, prediction, gts, gsize, blend=[
                                pres_train[0], gts_train[0]], img_num=img_vis_total)
                    else:
                        res_topo, _, _ = vis_batch(
                            img, prediction, gts, gsize, masks_train[0], blend=[
                                pres_train[0], gts_train[0]], img_num=img_vis_total)
                    prediction = pres_train[2:]
                    gts = gts_train[2:]
                    # res_pt = vis_batch(img, prediction, gts, masks_train[0], blend = [pres_train[1], gts_train[1]], img_num = img_vis_total, add_all_pts = True)
                    res_pt, keypts_pre, keypts_gt = vis_batch(
                        img, prediction, gts, gsize, blend=[
                            pres_train[1], gts_train[1]], img_num=img_vis_total, add_all_pts=True)
                    if exists(join(model_folder, 'results')) == False:
                        os.makedirs(join(model_folder, 'results'))

                    if args.ndc_add:
                        usm_pre = (
                            (pres_train[-1] * args_udf.dist_pt) < args_udf.dist_pt / 1.5)
                        usm_gt = (gts_train[-1] == 0)
                        pres_ndc = (edge_maps_pre, pt_map_pre, usm_pre)
                        gts_ndc = (edge_maps_gt, pt_map_gt, skel_gt, usm_gt)

                        res_ndc = vis_train_ndc(
                            args_ndc,
                            pres_ndc,
                            gts_ndc,
                            model_folder,
                            iter_counter,
                            gsizes,
                            save=False,
                            img_num_total=img_vis_total,
                            img_num_row=img_vis_total,
                            keypts_dicts_gt=keypts_gt,
                            keypts_dicts_pre=keypts_pre)

                        h, w = res_ndc.shape[0], res_ndc.shape[1]
                        res_topo = cv2.resize(
                            res_topo, (w, h), interpolation=cv2.INTER_AREA)
                        res_topo = np.concatenate((res_topo, res_ndc), axis=0)

                    Image.fromarray(
                        res_pt.astype(
                            np.uint8)).save(
                        join(
                            model_folder,
                            'results',
                            "train_%d_pt.png" %
                            iter_counter))
                    Image.fromarray(
                        res_topo.astype(
                            np.uint8)).save(
                        join(
                            model_folder,
                            'results',
                            "train_%d_topo.png" %
                            iter_counter))

                    # also visualize grad
                    if args.log:
                        fig_pt = wandb.Image(res_pt.astype(np.uint8))
                        fig_topo = wandb.Image(res_topo.astype(np.uint8))
                        wandb.log({'keypoint': fig_pt}, step=iter_counter)
                        wandb.log({'Topology': fig_topo}, step=iter_counter)

                # record loss value
                if iter_counter % 500 == 0 and args.log:
                    wandb.log({'Topo': losses[0]}, step=iter_counter)
                    wandb.log({'All': losses[1]}, step=iter_counter)
                    wandb.log({'End': losses[2]}, step=iter_counter)
                    wandb.log({'Sharp': losses[3]}, step=iter_counter)
                    wandb.log({'Junction': losses[4]}, step=iter_counter)
                    wandb.log({'USM': losses[5]}, step=iter_counter)

                    if args.ndc_add:
                        wandb.log(
                            {'Edge': loss_dict['edge']}, step=iter_counter)
                        wandb.log(
                            {'Keypt': loss_dict['keypt']}, step=iter_counter)
                        wandb.log(
                            {'Valence': loss_dict['valence']}, step=iter_counter)
                if args.debug:
                    if (iter_counter + 1) % 10 == 0:
                        break
                iter_counter += 1

            # save model at current time
            if epoch % 5 == 0:
                model_path_epoch = join(
                    model_folder,
                    "epoch%04d_udf.pth" %
                    epoch)
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer_udf.state_dict(),
                    'param': args,
                }, model_path_epoch)

                if args.ndc_add:
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': net_ndc.state_dict(),
                        'optimizer_state_dict': optimizer_ndc.state_dict(),
                        'param': args_ndc
                    }, model_path_epoch.replace("udf.pth", 'ndc.pth'))
            # save model every epoch
            torch.save({
                'epoch': epoch,
                'model_state_dict': net.state_dict(),
                'optimizer_state_dict': optimizer_udf.state_dict(),
                'param': args,
            }, model_path)

            if args.ndc_add:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': net_ndc.state_dict(),
                    'optimizer_state_dict': optimizer_ndc.state_dict(),
                    'param': args_ndc
                }, model_path.replace("udf.pth", 'ndc.pth'))

            # validation
            if epoch % 3 == 0:
                print("Log:\tstarting validation")
                iter_counter_val = 0
                loss_topo_val_total = 0
                loss_all_val_total = 0
                loss_sharp_val_total = 0
                loss_junc_val_total = 0
                loss_edge_val_total = 0
                loss_keypt_val_total = 0
                loss_valence_val_total = 0
                loss_usm_val_total = 0
                loss_end_val_total = 0

                net.eval()
                if args.ndc_add:
                    net_ndc.eval()
                p_bar_val = tqdm(val_loader, ncols=175)
                # p_avg = np.zeros(5)
                # r_avg = np.zeros(5)
                for data_val in p_bar_val:
                    img = data_val['img']
                    udf_val, udf_all_val, udf_end_val, udf_sharp_val, udf_junc_val = data_val[
                        'udfs']
                    mask_topo_val, pt_mask1_val, mask_all_val, mask_end_val, mask_sharp_val, mask_junc_val = data_val[
                        'udf_masks']
                    edge_maps_gt_val, pt_map_gt_val, usm_gt_val, skel_gt_val = data_val['ndc_gts']
                    edge_mask, usm_mask = data_val['ndc_mask']
                    gsize = 0.5 if args.up_scale else 1
                    gsizes = [gsize] * len(img)
                    with torch.no_grad():
                        udf_topo_pre, udfs_pre = net.forward_with_downscale_outs(
                            img)
                        udf_all_pre, udf_end_pre, udf_sharp_pre, udf_junc_pre, udf_usm_pre = udfs_pre
                        # compute the loss
                        pres_val = [
                            udf_topo_pre,
                            udf_all_pre,
                            udf_end_pre,
                            udf_sharp_pre,
                            udf_junc_pre,
                            udf_usm_pre]
                        gts_val = [
                            udf_val,
                            udf_all_val,
                            udf_end_val,
                            udf_sharp_val,
                            udf_junc_val,
                            usm_gt_val]
                        masks_val = [
                            mask_topo_val,
                            mask_all_val,
                            mask_end_val,
                            mask_sharp_val,
                            mask_junc_val,
                            usm_mask,
                            pt_mask1_val]
                        if iter_counter_val == 0:
                            loss_udf, losses = loss_total(
                                pres_val, gts_val, masks_val, to_numpy_inplace=True, dist_mode=args.dist_mode)
                        else:
                            loss_udf, losses = loss_total(
                                pres_val, gts_val, masks_val, to_numpy_inplace=False, dist_mode=args.dist_mode)

                        loss_topo_val_total += losses[0].item()
                        loss_all_val_total += losses[1].item()
                        loss_end_val_total += losses[2].item()
                        loss_sharp_val_total += losses[3].item()
                        loss_junc_val_total += losses[4].item()
                        loss_usm_val_total += losses[5].item()

                        if args.ndc_add:
                            base_coord = init_base_coord(
                                pt_map_gt_val, device, gsizes)
                            pt_map_gt_val = pt_map_gt_val - base_coord
                            if args_udf.dist_clip != args_ndc.dist_clip:
                                udf_topo_pre = (
                                    udf_topo_pre * args_udf.dist_clip).clip(
                                    0, args_ndc.dist_clip) / args_ndc.dist_clip

                            edge_maps_pre_val, pt_map_pre_val = net_ndc.forward(
                                udf_topo_pre.clip(0, 1), gsizes)

                            loss_dict, _, _ = compute_ndc_loss_all(
                                args_ndc, edge_maps_pre_val, edge_maps_gt_val, edge_mask, pt_map_pre_val, pt_map_gt_val)
                            pt_map_pre_val = pt_map_pre_val + base_coord
                            pt_map_gt_val = pt_map_gt_val + base_coord
                            loss_edge_val_total += loss_dict['edge']
                            loss_keypt_val_total += loss_dict['keypt']
                            loss_valence_val_total += loss_dict['valence']

                    if args.ndc_add:
                        p_bar_val.set_description(
                            "UDF: %.4f, NDC: %.4f, Topo: %.4f, Edge: %.4f, Keypt: %.4f, Valence: %.4f, USM: %.4f" %
                            (loss_udf.item(),
                             loss_ndc.item(),
                                losses[0],
                                loss_dict['edge'].item(),
                                loss_dict['keypt'].item(),
                                loss_dict['valence'].item(),
                                losses[5].item()))
                    else:
                        p_bar_val.set_description(
                            "Topo: %.4f, All: %.4f, End: %.4f, Sharp: %.4f, Junc: %.4f, USM: %.4f" %
                            (losses[0], losses[1], losses[2], losses[3], losses[4], losses[5]))

                    # visualize the first batch
                    if iter_counter_val == 0:
                        dist_clip = args.dist_clip
                        pres_val[0] = pres_val[0][:, :, :-1, :-1]
                        gts_val[0] = gts_val[0][:, :, :-1, :-1]
                        masks_val[0] = masks_val[0][:, :, :-1, :-1]
                        prediction = pres_val[:1]
                        gts = gts_val[:1]
                        if args.ndc_add:
                            res_topo, _, _ = vis_batch(
                                img, prediction, gts, gsize, blend=[
                                    pres_val[0], gts_val[0]], img_num=img_vis_total)
                        else:
                            res_topo, _, _ = vis_batch(
                                img, prediction, gts, gsize, masks_val[0], blend=[
                                    pres_val[0], gts_val[0]], img_num=img_vis_total)
                        prediction = pres_val[2:]
                        gts = gts_val[2:]
                        res_pt, keypts_pre, keypts_gt = vis_batch(
                            img, prediction, gts, gsize, blend=[
                                pres_val[1], gts_val[1]], img_num=img_vis_total, add_all_pts=True)

                        if args.ndc_add:
                            usm_pre_val = (
                                (udf_usm_pre.clip(
                                    0,
                                    1) *
                                    args_udf.dist_clip) < args_udf.dist_clip /
                                2).detach().cpu().numpy()
                            usm_gt_val = (
                                usm_gt_val == 0).detach().cpu().numpy()
                            pres_ndc = (
                                edge_maps_pre_val, pt_map_pre_val, usm_pre_val)
                            gts_ndc = (
                                edge_maps_gt_val,
                                pt_map_gt_val,
                                skel_gt_val,
                                usm_gt_val)

                            res_ndc = vis_train_ndc(
                                args_ndc,
                                pres_ndc,
                                gts_ndc,
                                model_folder,
                                iter_counter,
                                gsizes,
                                save=False,
                                img_num_total=img_vis_total,
                                img_num_row=img_vis_total,
                                keypts_dicts_gt=keypts_gt,
                                keypts_dicts_pre=keypts_pre)

                            val_res = vis_train_ndc(
                                args_ndc,
                                pres_ndc,
                                gts_ndc,
                                model_folder,
                                iter_counter,
                                gsizes,
                                False,
                                img_vis_total,
                                img_vis_total,
                                keypts_gt,
                                keypts_pre)
                            h, w = val_res.shape[0], val_res.shape[1]
                            res_topo = cv2.resize(
                                res_topo, (w, h), interpolation=cv2.INTER_AREA)
                            res_topo = np.concatenate(
                                (res_topo, val_res), axis=0)
                            # Image.fromarray(val_res.astype(np.uint8)).save(join(model_folder, 'results', "val_%d_ndc.png"%iter_counter))
                        Image.fromarray(
                            res_pt.astype(
                                np.uint8)).save(
                            join(
                                model_folder,
                                'results',
                                "val_%d_pt.png" %
                                iter_counter))
                        Image.fromarray(
                            res_topo.astype(
                                np.uint8)).save(
                            join(
                                model_folder,
                                'results',
                                "val_%d_topo.png" %
                                iter_counter))
                    iter_counter_val += 1
                    if args.debug:
                        import pdb
                        pdb.set_trace()

                if args.log:
                    fig_pt = wandb.Image(res_pt.astype(np.uint8))
                    fig_topo = wandb.Image(res_topo.astype(np.uint8))
                    wandb.log({'Keypoint Val': fig_pt,
                              'Topology Val': fig_topo}, step=iter_counter)
                    wandb.log({'Topo Val': loss_topo_val_total /
                              iter_counter_val}, step=iter_counter)
                    wandb.log({'All Val': loss_all_val_total /
                              iter_counter_val}, step=iter_counter)
                    wandb.log({'End Val': loss_end_val_total /
                              iter_counter_val}, step=iter_counter)
                    wandb.log({'Sharp Val': loss_sharp_val_total /
                              iter_counter_val}, step=iter_counter)
                    wandb.log({'Junction Val': loss_junc_val_total /
                              iter_counter_val}, step=iter_counter)
                    wandb.log({'USM Val': loss_usm_val_total /
                              iter_counter_val}, step=iter_counter)

                    if args.ndc_add:
                        wandb.log({'Edge Val': loss_edge_val_total /
                                  iter_counter_val}, step=iter_counter)
                        wandb.log({'Keypt Val': loss_keypt_val_total /
                                  iter_counter_val}, step=iter_counter)
                        wandb.log(
                            {'Valence Val': loss_valence_val_total / iter_counter_val}, step=iter_counter)
    if DEBUG:
        start_epoches(
            iter_counter,
            args,
            epoch_start,
            train_loader,
            model_folder)
    else:
        # for real training
        try:
            start_epoches(
                iter_counter,
                args,
                epoch_start,
                train_loader,
                model_folder)
        except Exception as e:
            print(e)
            torch.save(
                net.state_dict(),
                join(
                    model_folder,
                    "exception_saved.pth"))

def loss_total(
        pre,
        gt,
        masks,
        to_numpy_inplace=False,
        dist_mode='l2',
        loss_mode='idx'):
    losses = []
    udf_topo = gt[0].clone()
    mask_topo = masks[0].clone()
    mask_edge = masks[-1]
    assert len(pre) == len(gt)
    assert len(pre) == len(masks) - 1
    for i in range(len(pre)):
        # compute topolgy UDF loss
        if i == 0:
            losses.append((dist_loss(pre[i],
                                     gt[i],
                                     [mask_topo,
                                      mask_topo],
                                     pt_mode=False,
                                     dist_mode=dist_mode,
                                     loss_mode=loss_mode)).unsqueeze(0))
        # else compute point UDF loss
        else:
            losses.append(0.1 * (dist_loss(pre[i],
                                           gt[i],
                                           [masks[-1],
                                            masks[i]],
                                           pt_mode=True,
                                           dist_mode=dist_mode,
                                           loss_mode=loss_mode)).unsqueeze(0))
        if to_numpy_inplace:
            pre[i] = pre[i].detach().cpu().numpy()
            gt[i] = gt[i].detach().cpu().numpy()
            masks[i] = masks[i].detach().cpu().numpy()
    masks[-1] = masks[-1].detach().cpu().numpy()
    return torch.sum(torch.cat(losses, dim=0)) / len(losses), losses


def data_loader_testing():
    '''
    A function that test if the data augmentation is correct
    '''
    args = parse()
    train_loader = load_data(args.ds, 50, patch_size=256, mask_decay=args.md)
    for data in tqdm(train_loader):
        # get training data from dataset
        img = data['img']
        udf, udf_all_pt, udf_end_pt, udf_sharp_pt, udf_t_pt, udf_x_pt, udf_star_pt = data[
            'udfs']
        s_mask, mask_all_pt, mask_end_pt, mask_sharp_pt, mask_t_pt, mask_x_pt, mask_star_pt = data[
            'masks']
        gt_all_pt, gt_end_pt, gt_sharp_pt, gt_t_pt, gt_x_pt, gt_star_pt = data['gts']
        coord_all_pt, coord_end_pt, coord_sharp_pt, coord_t_pt, coord_x_pt, coord_star_pt = data[
            'coords']

        udf_all_pt = udf_all_pt.cpu().numpy().squeeze()
        udf_end_pt = udf_end_pt.cpu().numpy().squeeze()
        udf_sharp_pt = udf_sharp_pt.cpu().numpy().squeeze()
        udf_t_pt = udf_t_pt.cpu().numpy().squeeze()
        udf_x_pt = udf_x_pt.cpu().numpy().squeeze()
        udf_star_pt = udf_star_pt.cpu().numpy().squeeze()
        udf = udf.cpu().numpy().squeeze()
        s_mask = s_mask.cpu().numpy().squeeze()
        res_pt = []
        res_topo = []
        for i in range(len(img)):
            udfs = [
                udf_end_pt[i],
                udf_sharp_pt[i],
                udf_t_pt[i],
                udf_x_pt[i],
                udf_star_pt[i]]
            # convert img tensor to 3 channel gray numpy image
            sketch = img[i].cpu().detach().squeeze().numpy()
            sketch = sketch * 255
            sketch = np.expand_dims(sketch, axis=-1).repeat(3, axis=-1)
            fig_pt_gt = plot_udf_coord_numpy(sketch, udfs)
            fig_topo_gt = plot_udf_coord_numpy(sketch, [udf[i]])
            # concate all images as a large numpy array along y-axis
            mk = np.expand_dims(s_mask[i], axis=-1).repeat(3, axis=-1)
            mk = (mk * 255).clip(0, 255)
            res_pt.append(
                np.concatenate(
                    (sketch, mk, fig_pt_gt, fig_topo_gt), axis=0))
            # res_topo.append(fig_topo_gt)
        res = np.concatenate(res_pt, axis=1)
        # res2 = np.concatenate(res_topo, axis = 1)
        # res = np.concatenate((res1, res2), axis = 0)
        from PIL import Image
        Image.fromarray(res.astype(np.uint8)).save("res.png")
        # break


def parse():
    parser = argparse.ArgumentParser(
        description='Rough Sketch Junction Detection Network')
    '''
    Training
    '''
    parser.add_argument(
        '--name',
        metavar="name",
        type=str,
        help='the log name of the current training',
        default=None)
    parser.add_argument(
        '--udf',
        metavar="udf",
        type=str,
        help='pretrained udf model file name',
        default="udf_4.5.pth")
    parser.add_argument(
        '--ndc',
        metavar="ndc",
        type=str,
        help='pretrained udf model file name',
        default="ndc_4.5.pth")
    parser.add_argument(
        '--loss_mode',
        type=str,
        help="loss mode for UDF, it could be 'idx' or 'multi', where the first mode has better performance but is less stable",
        default='idx')
    parser.add_argument(
        '--lr',
        metavar='initial learning rate',
        type=float,
        help='the start learning rate for training',
        default=1e-4)
    parser.add_argument('--log', action="store_true", help="enable wandb log")
    parser.add_argument(
        '--usr', 
        action="store_true", 
        help="wandb user name", 
        default = "your_wandb_usr_name")
    parser.add_argument(
        '--epoch',
        metavar='total epoches',
        type=int,
        help='the epoch numbers for the whole training',
        default=600)
    parser.add_argument('--ds', metavar='dataset', type=str,
                        help='root path of dataset', default="./data/full")
    parser.add_argument('--cp', metavar='checkpoint', type=str,
                        help='path to checkpoint', default="")
    parser.add_argument(
        '--fcp',
        metavar='checkpoint',
        type=str,
        help='path to finetune checkpoint',
        default="./pretrained")
    parser.add_argument(
        '--relu',
        action='store_true',
        help='enable leaky relu loss')
    parser.add_argument(
        '--dist_mode',
        type=str,
        help="the mode of distance loss, it could be 'l1' or 'l2'",
        default='l2')
    parser.add_argument(
        '--review',
        action='store_true',
        help='penalize false positive prediction inside edge mask')
    parser.add_argument(
        '--review_all',
        action='store_true',
        help='penalize all false positive predictions')
    parser.add_argument(
        '--eval',
        action='store_true',
        help="set the UDF model to eval() when training, this is wired but works... ┑(￣Д ￣)┍")
    parser.add_argument(
        '--debug',
        action='store_true',
        help='enable debug mode')
    '''
    Data loading
    '''
    parser.add_argument('--ps', metavar='patch size', type=int,
                        help='the patch size of training input', default=128)
    parser.add_argument(
        '--workers',
        metavar='workers',
        type=int,
        help='how many process used for loading the training data',
        default=0)
    parser.add_argument('--bs', metavar='batch size', type=int,
                        help='batch size', default=1)
    parser.add_argument(
        '--dist_clip',
        metavar='threshold distance',
        type=float,
        help='the threshold for topology UDF clipping',
        default=4.5)
    parser.add_argument(
        '--dist_pt',
        metavar='threshold distance',
        type=float,
        help='the threshold for point UDF clipping',
        default=4.5)
    parser.add_argument(
        '--no_rotation',
        action='store_true',
        help='skip rotation data augmentation for faster trainning speed')
    parser.add_argument(
        '--jpg',
        action='store_true',
        help='add random jpg compression')
    parser.add_argument(
        '--paper_background',
        action='store_true',
        help='add paper texture as back ground')
    parser.add_argument(
        '--approx_udf',
        action='store_true',
        help="for paper experiment only, NEVER TURN THIS FLAG ON")
    # this option is not really helpful to the final performance, DON'T USE
    parser.add_argument(
        '--insert_skel',
        action='store_true',
        help='insert skeleton into UDF when loading the data')
    '''
    Network
    '''
    parser.add_argument(
        '--up_scale',
        action='store_true',
        help='output UDF with 2x upscale')
    parser.add_argument(
        '--edge_map',
        action='store_true',
        help='enable edge map prediction mode, otherwise will enable keypoint prediction mode')
    parser.add_argument(
        '--ndc_add',
        action='store_true',
        help='add a pretrained NDC network for end-to-end training')
    parser.add_argument(
        '--ndc_scratch',
        action='store_true',
        help='skip parameter loading when adding NDC network')
    parser.add_argument(
        '--udf_scratch',
        action='store_true',
        help='skip parameter loading when adding UDF network')
    parser.add_argument(
        '--udf_fix',
        action='store_true',
        help='fix the parameters in the UDF network')
    parser.add_argument(
        '--ndc_fix',
        action='store_true',
        help='fix the parameters in the ndc network, this option will be overwirten if --ndc_scratch is True')
    parser.add_argument(
        '--hourglass_input_channels',
        type=int,
        help='UDF resnext moduel channels',
        default=64)
    parser.add_argument(
        '--hourglass_channels',
        type=int,
        help='UDF hourglass network channels',
        default=96)
    parser.add_argument(
        '--downsample_depth',
        type=int,
        help='UDF hourglass network deepth',
        default=3)
    parser.add_argument(
        '--cardinality',
        help='double Resnext module cardinality',
        action='store_true')
    parser.add_argument(
        '--backbone',
        type=str,
        help='Backbone seletion of UDF network',
        default="ResNet")
    parser.add_argument(
        '--width',
        type=int,
        help='ResNext module branch width',
        default=8)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    __spec__ = None
    train()
    # data_loader_testing()
