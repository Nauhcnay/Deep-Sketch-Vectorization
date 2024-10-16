import torch
import numpy as np
import argparse
import os
import cv2
import time
import tempfile
import uuid

from types import SimpleNamespace
from os import path
from datetime import datetime
from network.udc import CNN_2d_resnet
from network.keypoint import PyramidalNet
from PIL import Image
from types import SimpleNamespace
from utils.ndc_tools import lines_to_udf_fast, refine_topology, lines_to_svg, map_to_lines, pre_to_map, pre_to_lines, init_base_coord, linemap_to_lines, roll_edge, logical_minus, downsample_ndc
from dataset.preprocess import svg_to_numpy
from torch.nn import functional as F
from network.thin import Thinner
from network.anime2sketch import create_model
from utils.keypt_tools import plot_udf_coord_numpy, draw_cross, udf_to_hm, blend_heatmaps, extract_keypts, vis_pt_single, udf_filter
from torch.nn.functional import normalize, interpolate
from dataset.augmentation import blend_skeletons
from svgpathtools import wsvg
from utils.svg_tools import simplify_graph, refine_topology_2nd, open_svg_flatten, fitBezier, ramerDouglas
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode

# global variables
TEMP_DIR = None
FILE_NAME = None
MODEL_UDF = None
ARGS_UDF = None
MODEL_NDC = None
ARGS_NDC = None
DEVICE = None
CANVAS_SIZE = None
IMG = None
NDC_NPZ = None
UDF_NPZ = None
# global variables

# functions borrowed from https://github.com/Mukosame/Anime2Sketch/


def get_transform(
        load_size=0,
        grayscale=False,
        method=InterpolationMode.BICUBIC,
        convert=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if load_size > 0:
        osize = [load_size, load_size]
        transform_list.append(transforms.Resize(osize, method))
    if convert:
        transform_list += [transforms.ToTensor()]
        if grayscale:
            transform_list += [transforms.Normalize((0.5,), (0.5,))]
        else:
            transform_list += [
                transforms.Normalize(
                    (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def read_img_path(path, load_size):
    """read tensors from a given image path
    Parameters:
        path (str)     -- input image path
        load_size(int) -- the input size. If <= 0, don't resize
    """
    if isinstance(path, str):
        img = Image.open(path).convert('RGB')
    else:
        img = path.convert("RGB")
    aus_resize = None
    if load_size > 0:
        aus_resize = img.size
    transform = get_transform(load_size=load_size)
    image = transform(img)
    return image.unsqueeze(0), aus_resize


def tensor_to_img(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.
    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """

    if not isinstance(input_image, np.ndarray):
        if isinstance(
                input_image,
                torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        # convert it into a numpy array
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / \
            2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)
# functions borrowed from https://github.com/Mukosame/Anime2Sketch/


def load_checkpoint_udf(filepath):
    """
    Loads model from a checkpoint
    we will need the distance threshold to accruately recover the UDF for the next NDC model
    """
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
    try:
        args_udf = SimpleNamespace(**checkpoint['param'])
    except BaseException:
        args_udf = checkpoint['param']
    args_udf = vars(args_udf)
    if "hourglass_input_channels" not in args_udf:
        args_udf["hourglass_input_channels"] = 64
    if "hourglass_channels" not in args_udf:
        args_udf["hourglass_channels"] = 96
    if "downsample_depth" not in args_udf:
        args_udf["downsample_depth"] = 3
    if "cardinality" not in args_udf:
        args_udf["cardinality"] = False
    args_udf = SimpleNamespace(**args_udf)
    loadmodel = PyramidalNet(
        n_classes=6,
        up_scale=args_udf.up_scale,
        hourglass_channels=args_udf.hourglass_channels,
        hourglass_input_channels=args_udf.hourglass_input_channels,
        downsample_depth=args_udf.downsample_depth,
        cardinality=args_udf.cardinality)
    loadmodel.load_state_dict(checkpoint['model_state_dict'])
    return loadmodel, args_udf


def load_checkpoint_ndc(filepath):
    """
    Loads model from a checkpoint
    we will need the distance threshold to accruately recover the UDF for the next NDC model
    """
    # device = 'cuda' if torch.cuda.is_available() else 'cpu'
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=False)
    args_ndc = checkpoint['param']
    if isinstance(args_ndc, dict) == False:
        args_ndc = vars(args_ndc)
    if "noisy" not in args_ndc:
        args_ndc['noisy'] = False
    if "resnext_f" not in args_ndc:
        args_ndc['resnext_f'] = False
    if "resnext_i" not in args_ndc:
        args_ndc['resnext_i'] = False
    args_ndc = SimpleNamespace(**args_ndc)

    loadmodel = CNN_2d_resnet('cpu', channels=args_ndc.dims,
                              layers=args_ndc.layers,
                              multi_scale=args_ndc.msb,
                              batch_norm=args_ndc.bn,
                              drop_out=args_ndc.do,
                              coord_conv=args_ndc.coord,
                              noisy=args_ndc.noisy,
                              resnext_feature=args_ndc.resnext_f,
                              resnext_input=args_ndc.resnext_i)
    loadmodel.load_state_dict(checkpoint['model_state_dict'])
    return loadmodel, args_ndc


def add_keypt(img, keypt_dict, org_size):
    color_dict = {
        'sharp_turn': np.array([0, 255, 0]),
        'end_point': np.array([255, 0, 0]),
        'junc': np.array([0, 0, 255]),
    }
    h, w = org_size
    hh, ww = img.shape[0], img.shape[1]
    ratio = hh / h
    assert abs(ratio - ww / w) < 1
    for key, value in keypt_dict.items():
        cd = (value * ratio).astype(int)
        if len(cd) > 0:
            cd[cd[:, 0] >= ww] = ww - 1
            cd[cd[:, 1] >= hh] = hh - 1
        color = color_dict[key]
        img = draw_cross(img, cd, color)
    return img


def load_img(img_input, device, up_scale,
             thin=None,
             line_extractor=None,
             resize=False,
             path_to_out=None,
             img_name=None,
             resize_to=1024,
             ui_mode=True):
    # todo: make function support path and numpy array input at the same time
    if isinstance(img_input, str):
        _, png = path.split(img_input)
        name, _ = path.splitext(png)
    elif img_name is not None:
        name, _ = path.splitext(img_name)
    elif isinstance(img_input, Image.Image):
        # generate a random file name
        name = str(uuid.uuid4())
    else:
        raise ValueError("Invalid input format %s" % str(type(img_input)))

    if line_extractor is not None and thin is None:
        print("log:\tpre-processing with line extractor")
        line_extractor.eval()
        img_tensor, aus_size = read_img_path(img_input, 512)
        aus_tensor = line_extractor(img_tensor.to(device))
        img_np = tensor_to_img(aus_tensor)
        img_np = cv2.resize(
            img_np,
            (aus_size[0],
             aus_size[1]),
            interpolation=cv2.INTER_AREA)
    elif thin is not None and line_extractor is None:
        print("log:\tpre-processing with line normalizer")
        img_temp = thin(img_input.convert("L"))
        img_np = (img_temp.cpu().squeeze().numpy() * 255).astype(int)
    elif line_extractor is not None and thin is not None:
        print("log:\tpre-processing with line extractor and normalizer")
        line_extractor.eval()
        img_tensor, aus_size = read_img_path(img_input, 512)
        aus_tensor = line_extractor(img_tensor.to(device))
        img_np = tensor_to_img(aus_tensor)
        img_np = cv2.resize(
            img_np,
            (aus_size[0],
             aus_size[1]),
            interpolation=cv2.INTER_AREA)
        img_temp = thin(img_np)
        img_np = (img_temp.cpu().squeeze().numpy() * 255).astype(int)
    else:
        assert line_extractor is None and thin is None
        try:
            img_np = np.array(Image.open(img_input).convert("L"))
        except BaseException:
            img_np = np.array(img_input.convert("L"))
    h, w = img_np.shape[0], img_np.shape[1]
    h_raw = h
    w_raw = w
    if len(img_np.shape) == 3:
        if img_np.shape[2] == 4:
            background = np.ones((h, w, 3)) * 255
            img_np_rgb = img_np[:, :, 0:3]
            img_alpha = np.expand_dims(img_np[:, :, 3], -1).astype(float) / 255
            img_np = (img_np_rgb * img_alpha + background *
                      (1 - img_alpha)).mean(axis=-1)
        else:
            img_np = img_np.mean(axis=-1)
    if resize:
        long_side = h if h > w else w
        ratio = resize_to / long_side
        h = int(h * ratio)
        w = int(w * ratio)
        img_np = cv2.resize(img_np.astype(np.uint8), (w, h),
                            interpolation=cv2.INTER_AREA)
    h = h // 8 * 8
    w = w // 8 * 8
    img_np = img_np[:h, :w]
    img = torch.FloatTensor(img_np / 255).unsqueeze(0).unsqueeze(0)
    # change image color to blue
    if ui_mode:
        img_alpha = 1 - img_np[...,
                               np.newaxis].repeat(3,
                                                  axis=-1).astype(float) / 255
        img_r = np.ones((h, w)) * 99
        img_g = np.ones((h, w)) * 197
        img_b = np.ones((h, w)) * 218
        img_line = np.stack((img_r, img_g, img_b), axis=-1)
        img_bg = np.ones((h, w, 3)) * 255
        img_np = (img_line * img_alpha + img_bg *
                  (1 - img_alpha)).astype(np.uint8)
    else:
        img_np = np.repeat(img_np[..., np.newaxis], 3, axis=-1)
    # save updated input image if possible
    if path_to_out is not None:
        if up_scale:
            img_np = cv2.resize(
                img_np, (w * 2, h * 2), interpolation=cv2.INTER_AREA)
        if os.path.exists(path_to_out) == False:
            os.makedirs(path_to_out)
        Image.fromarray(img_np).save(path.join(path_to_out, name + '.png'))
    return img.to(device), img_np, (h_raw, w_raw), name


def Ext(pts):
    try:
        return np.repeat(pts[:, np.newaxis, :], 2, axis=1).reshape(-1, 2)
    except BaseException:
        return None


def load_npz(path_to_npz, device, dist=-1):
    if dist == -1:
        dist = 255
    gts = np.load(path_to_npz)

    # load keypoints
    end_pt = gts['end_point']
    sharp_pt = gts['sharp_turn']
    t_pt = gts['T']
    x_pt = gts['X']
    star_pt = gts['star']

    # load or create udfs
    udf_topo = gts['udf'][np.newaxis, np.newaxis, ...].clip(0, dist)
    h, w = udf_topo.shape[0] - 1, udf_topo.shape[1] - 1
    # udf_endpt = lines_to_udf_fast(Ext(end_pt), (h, w)).clip(0, dist) / dist
    # udf_sharp = lines_to_udf_fast(Ext(sharp_pt), (h, w)).clip(0, dist) / dist
    # udf_t = lines_to_udf_fast(Ext(t_pt), (h, w)).clip(0, dist) / dist
    # udf_x = lines_to_udf_fast(Ext(x_pt), (h, w)).clip(0, dist) / dist
    # udf_star = lines_to_udf_fast(Ext(star_pt), (h, w)).clip(0, dist) / dist

    udf_topo = torch.FloatTensor(udf_topo).to(device)
    # udf_endpt = torch.FloatTensor(udf_endpt).to(device)
    # udf_sharp = torch.FloatTensor(udf_sharp).to(device)
    # udf_t = torch.FloatTensor(udf_t).to(device)
    # udf_x = torch.FloatTensor(udf_x).to(device)
    # udf_star = torch.FloatTensor(udf_star).to(device)

    # return udf_topo, udf_endpt, udf_sharp, udf_t, udf_x, udf_star, gts
    return udf_topo, gts


def D(t):
    try:
        n = t.detach().cpu().numpy().squeeze()
        return n
    except BaseException:
        return t


def pre_udf(net, img):
    net.eval()
    with torch.no_grad():
        udf_topo_pre, udfs_pre = net.forward_with_downscale_outs(img)
        udf_all_pre, udf_end_pre, udf_sharp_pre, udf_junc_pre, udf_usm_pre = udfs_pre

        # udf_topo_pre = D(udf_topo_pre)
        udf_all_pre = D(udf_all_pre)
        udf_end_pre = D(udf_end_pre)
        udf_sharp_pre = D(udf_sharp_pre)
        udf_junc_pre = D(udf_junc_pre)
        usm_pre = D(udf_usm_pre)

    return udf_topo_pre, udf_all_pre, udf_end_pre, udf_sharp_pre, udf_junc_pre, usm_pre


def pre_svg(net, udf, gsize):
    base_coord = init_base_coord(udf[..., :-1, :-1], udf.device, gsize)
    net.eval()
    with torch.no_grad():
        edge_maps_pre, pt_map_pre = net.forward(udf, gsize)
    return edge_maps_pre, pt_map_pre + base_coord

# API for front end


def load_model(args, device):
    # load pretrained model
    assert (os.path.exists(args.model_udf))
    assert (os.path.exists(args.model_ndc))

    model_udf, args_udf = load_checkpoint_udf(args.model_udf)
    model_udf = model_udf.to(device)
    model_ndc, args_ndc = load_checkpoint_ndc(args.model_ndc)
    model_ndc = model_ndc.to(device)
    return model_udf, args_udf, model_ndc, args_ndc


def predict_UDF(
        img,
        img_np,
        model_udf,
        out_path=None,
        name=None,
        usm_thr=0.5):
    udf_topo_pre, udf_all_pre, udf_end_pre, udf_sharp_pre, udf_junc_pre, usm_pre = \
        pre_udf(model_udf, img)
    usm_pre_ = (
        usm_pre.clip(
            0, 1) < usm_thr) | (
        udf_all_pre.clip(
            0, 1) < usm_thr / 3)
    # extract keypoints from the prediction
    udf_all_pre = udf_filter(udf_all_pre)
    udf_end_pre = udf_filter(udf_end_pre)
    udf_sharp_pre = udf_filter(udf_sharp_pre)
    udf_junc_pre = udf_filter(udf_junc_pre)
    udfs_pre = [udf_end_pre, udf_sharp_pre, udf_junc_pre]
    h, w = udf_all_pre.shape
    img_np = cv2.resize(img_np, (w, h), interpolation=cv2.INTER_AREA)
    keypt_pre_np, keypt_pre_dict = plot_udf_coord_numpy(
        img_np, udfs_pre, udf_all_pre)
    if out_path is not None:
        np.savez(path.join(out_path, name + "_udf.npz"),
                 end_point=[keypt_pre_dict["end_point"]],
                 sharp_turn=[keypt_pre_dict["sharp_turn"]],
                 junc=[keypt_pre_dict["junc"]],
                 usm=[usm_pre],
                 usm_applied=[usm_pre_],
                 usm_uncertain=None)
    for k in keypt_pre_dict:
        keypt_pre_dict[k] = keypt_pre_dict[k] / 2

    return udf_topo_pre, usm_pre_, keypt_pre_dict, keypt_pre_np


def predict_SVG(
        udf_topo_pre,
        model_ndc,
        args_udf,
        args_ndc,
        canvas_size,
        out_path=None,
        refine=True,
        name=None,
        to_npz=True):
    canvas_h, canvas_w = canvas_size
    # predict SVG from UDF
    if args_udf.dist_clip != args_ndc.dist_clip:
        udf_topo_pre = (udf_topo_pre * args_udf.dist_clip).clip(0,
                                                                args_ndc.dist_clip) / args_ndc.dist_clip
    else:
        udf_topo_pre = udf_topo_pre.clip(0, 1)
    gsizes = [0.5] if args_udf.up_scale else [1]
    edge_maps_pre, pt_map_pre = pre_svg(model_ndc, udf_topo_pre, gsizes)
    pt_map_pre = D(pt_map_pre.squeeze().permute(1, 2, 0))

    # reconstruct vector image from prediction
    edge_maps_pre = edge_maps_pre.permute(0, 2, 3, 1)[..., :-1].squeeze()
    edge_maps_pre_x, edge_maps_pre_y, _ = pre_to_map(edge_maps_pre)
    edge_maps_pre_xy = torch.stack((edge_maps_pre_x, edge_maps_pre_y), dim=-1)
    lines_pre, linemap_pre_x, linemap_pre_y, edge_maps_pre_xy = map_to_lines(
        D(edge_maps_pre_xy), pt_map_pre, refine=refine)

    # save to result
    if out_path is not None:
        svg_pre = path.join(out_path, name + "_raw.svg")
        lines_to_svg(lines_pre * 2, canvas_w * 2, canvas_h * 2, svg_pre)
        if to_npz:
            path_npz_ndc = path.join(out_path, name + "_ndc.npz")
            path_npz_udf = path.join(out_path, name + "_udf.npz")
            assert path.exists(path_npz_udf)
            np.savez(
                path_npz_ndc,
                edge_map=D(edge_maps_pre_xy),
                pt_map=pt_map_pre,
                lines_map_x=linemap_pre_x[np.newaxis, ...],
                lines_map_y=linemap_pre_y[np.newaxis, ...],
                lines_refined=None)
    return linemap_pre_x, linemap_pre_y, pt_map_pre, edge_maps_pre_xy


def simplify_SVG(
        path_to_svg,
        keypt_pre_dict,
        bezier=False,
        rdp_simplify=False,
        epsilon=0.4,
        skip_len=4):
    start_time = time.time()
    p, name = path.split(path_to_svg)
    name, _ = path.splitext(name)
    name = "_".join(name.split("_")[:-1])
    paths, (h, w) = open_svg_flatten(path_to_svg)
    strokes, nodes = simplify_graph(
        paths, keypt_pre_dict, mode="hybird", skip_len=skip_len)

    # strokes, _ = refine_topology_2nd(strokes, keypt_pre_dict, nodes[-1])
    if rdp_simplify:
        print("log:\tRDP simplify...")
        strokes = ramerDouglas(strokes, epsilon)
        end_time = time.time()
        # f.write("curve simplify within %f Seconds, "%(end_time - start_time))
    if bezier:
        print("log:\tBezier curve fitting...")
        try:
            strokes = fitBezier(strokes)
            end_time = time.time()
        except Exception as e:
            print(str(e))
        # f.write("curve fitting within %f Seconds, "%(end_time - start_time))
    attributes = [{"fill": 'none',
                   "stroke": "#000000",
                   "stroke-width": '1',
                   "stroke-linecap": "round"}] * len(strokes)
    wsvg(strokes,
         stroke_widths=[0.5] * len(strokes),
         dimensions=(w, h),
         filename=path.join(p, name + "_final.svg"),
         attributes=attributes)
    # also save another version for visulization
    vis_stroke(strokes, (h, w), path.join(p, name + "_vis.svg"))


def init_temp_dir():
    global TEMP_DIR
    if TEMP_DIR is None:
        # TEMP_DIR = path.join(tempfile.gettempdir(), "sketchvg")
        TEMP_DIR = "./web/output/"
        print("log:\tsetting temp folder to %s" % TEMP_DIR)

# step 1


def open_img(
        img_input,
        thin=False,
        line_extractor=False,
        name=None,
        resize_to=512):
    global DEVICE
    global IMG
    global IMG_NP
    global FILE_NAME
    global CANVAS_SIZE
    init_temp_dir()
    # predict udf

    resize_to = int(resize_to)
    # DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    DEVICE = 'cpu'
    ww, hh = img_input.size
    longer = ww if ww > hh else hh
    resize = True if longer > resize_to else False
    if resize:
        ratio = resize_to / longer
        print("log:\timage size (%dx%d) is too large, resize to (%dx%d)" %
              (hh, ww, int(hh * ratio), int(ww * ratio)))
    else:
        ratio = 1

    IMG, IMG_NP, canvas_size, FILE_NAME = load_img(img_input, DEVICE, True,
                                                   thin=Thinner() if thin else None,
                                                   line_extractor=create_model('default').to(
                                                       DEVICE) if line_extractor else None,
                                                   resize=resize,
                                                   path_to_out=TEMP_DIR,
                                                   img_name=name,
                                                   resize_to=resize_to)
    h, w = IMG.shape[2], IMG.shape[3]
    CANVAS_SIZE = (h, w)
    return h * 2, w * 2


def usm_to_regions(usm):
    usm_regions = {}
    usm_indices = []
    for r in np.unique(usm):
        if r == 0:
            continue
        coords = np.array(np.where(usm == r)).T
        usm_regions[r] = coords
        usm_indices.append(coords)
    if len(usm_indices) > 0:
        usm_indices = np.concatenate(usm_indices, axis=0)
    return usm_regions, usm_indices


def vectorize(path_model_ndc, path_model_udf, usm_thr=0.4, refine=True):
    start = time.time()
    global MODEL_UDF
    global ARGS_UDF
    global MODEL_NDC
    global ARGS_NDC
    global DEVICE
    init_temp_dir()
    global TEMP_DIR
    global FILE_NAME
    global CANVAS_SIZE
    global IMG
    global IMG_NP
    global UDF_NPZ
    global NDC_NPZ
    global SVG_REFINED
    args = {}
    args['model_udf'] = path_model_udf
    args['model_ndc'] = path_model_ndc
    args = SimpleNamespace(**args)
    # init model
    if MODEL_UDF is None:
        MODEL_UDF, ARGS_UDF, MODEL_NDC, ARGS_NDC = load_model(args, DEVICE)

    # udf_topo_pre, usm_pre_, keypt_pre_dict, _ = predict_UDF(IMG, IMG_NP, MODEL_UDF, out_path = TEMP_DIR, name = FILE_NAME, usm_thr = usm_thr)
    print("log:\tworking mode is set to %s" % DEVICE)
    udf_topo_pre, usm_pre_, keypt_pre_dict, _ = predict_UDF(
        IMG,
        IMG_NP,
        MODEL_UDF,
        out_path=TEMP_DIR,
        name=FILE_NAME,
        usm_thr=usm_thr)

    # predict svg
    predict_SVG(
        udf_topo_pre,
        MODEL_NDC,
        ARGS_UDF,
        ARGS_NDC,
        CANVAS_SIZE,
        out_path=TEMP_DIR,
        refine=refine,
        name=FILE_NAME)

    # refine topology with automatic mode
    path_npz_udf = path.join(TEMP_DIR, FILE_NAME + "_udf.npz")
    path_npz_ndc = path.join(TEMP_DIR, FILE_NAME + "_ndc.npz")
    UDF_NPZ = dict(np.load(path_npz_udf, allow_pickle=True))
    NDC_NPZ = dict(np.load(path_npz_ndc, allow_pickle=True))

    lines_x = NDC_NPZ['lines_map_x'][0].copy()
    lines_y = NDC_NPZ['lines_map_y'][0].copy()
    lines_pre, refined_dict, usm_applied, usm_uncertain = refine_topology(
        NDC_NPZ['edge_map'], NDC_NPZ['pt_map'], UDF_NPZ['usm_applied'][0], lines_x, lines_y, keypt_pre_dict, down_rate=4, downsample=False, manual=False)

    canvas_h, canvas_w = CANVAS_SIZE
    # this line should be removed since we could read it from memory
    lines_to_svg(
        lines_pre * 2,
        canvas_w * 2,
        canvas_h * 2,
        path.join(
            TEMP_DIR,
            FILE_NAME + "_refine.svg"))
    # for paper figures only, comment this line for regular UI demo!
    # simplify_SVG(path.join(TEMP_DIR, FILE_NAME + "_raw.svg"), keypt_pre_dict, bezier = False, rdp_simplify = True)

    end = time.time()
    print(
        "log:\tinitail vectorization %s finished in %.2f seconds" %
        (FILE_NAME, end - start))

    # convert usm to coordinates
    # usm_applied_r = usm_to_regions(usm_applied)
    # usm_uncertain_r = usm_to_regions(usm_uncertain)
    # save to npz
    # when saving to numpy, keypoint coord need be doubled
    UDF_NPZ["end_point"] = np.array([keypt_pre_dict["end_point"] * 2])
    UDF_NPZ["sharp_turn"] = np.array([keypt_pre_dict["sharp_turn"] * 2])
    UDF_NPZ["junc"] = np.array([keypt_pre_dict["junc"] * 2])
    UDF_NPZ["usm_applied"] = np.array([usm_applied])
    UDF_NPZ["usm_uncertain"] = np.array([usm_uncertain])
    # DC lines outside USM
    NDC_NPZ["lines_map_x"] = np.array([lines_x])
    NDC_NPZ["lines_map_y"] = np.array([lines_y])
    # refined lines by USM surgery
    NDC_NPZ["lines_refined"] = np.array([refined_dict])

    # we also could this line later
    npz_save(path_npz_udf, path_npz_ndc, UDF_NPZ, NDC_NPZ)

    return UDF_NPZ

# step 2, this step could be repeated many times


def usm_surgery(op, usm_updated=None, keypt_updated=None,
                manual=True, name=None, canvas_size=None):
    start = time.time()
    # init temp folder and temp fild name
    init_temp_dir()
    global TEMP_DIR
    global FILE_NAME
    global CANVAS_SIZE
    global UDF_NPZ
    global NDC_NPZ
    if FILE_NAME is None:
        assert name is not None
        name, _ = path.splitext(name)
        FILE_NAME = name

    # load keypoint and USM from saved record
    if CANVAS_SIZE is None:
        assert canvas_size is not None
        CANVAS_SIZE = canvas_size
    canvas_h, canvas_w = CANVAS_SIZE

    if UDF_NPZ is None or NDC_NPZ is None:
        path_to_npz_udf = path.join(TEMP_DIR, FILE_NAME + "_udf.npz")
        path_to_npz_ndc = path.join(TEMP_DIR, FILE_NAME + "_ndc.npz")
        UDF_NPZ = dict(np.load(path_to_npz_udf, allow_pickle=True))
        NDC_NPZ = dict(np.load(path_to_npz_ndc, allow_pickle=True))
    need_push = False
    need_refine = False

    if op == 'read':
        keypts, usm_applied, usm_uncertain, edge_map, pt_map, lines_x, lines_y, lines_refined = usm_read(
            UDF_NPZ, NDC_NPZ)
    elif op == 'undo':
        usm_pop(UDF_NPZ, NDC_NPZ)
        keypts, usm_applied, usm_uncertain, edge_map, pt_map, lines_x, lines_y, lines_refined = usm_read(
            UDF_NPZ, NDC_NPZ)
        need_push = True
    elif op == 'modify':
        # todo: update logic here
        keypts = {}
        # read new keypoint from UI
        for key in keypt_updated:
            keypts[key] = keypt_updated[key] / 2
            keypts[key][..., (0, 1)] = keypts[key][..., (1, 0)]
        # read from last USM
        _, usm_applied, usm_uncertain, edge_map, pt_map, lines_x, lines_y, lines_refined = usm_read(
            UDF_NPZ, NDC_NPZ)
        assert usm_applied.shape == usm_updated.shape
        # now we assume usm_updated is a boolean map
        usm_applied, usm_iter, modify_idx = update_usm(
            usm_applied, usm_updated)
        # print("log:\tgot region %s modified"%str(np.unique(usm_iter)))
        # if there is region modified, restore the line map and remove refined
        # lines in the corresponding regions
        if len(modify_idx) > 0:
            # remove refined lines
            for r in modify_idx:
                lines_refined[r] = []
            # usm_mask = usm_applied != 0
            # a dirty fix
            # usm_mask1 = logical_minus(usm_mask, roll_edge(usm_mask, 1, axis = 'x'))
            # usm_mask2 = logical_minus(usm_mask, roll_edge(usm_mask, 1, axis = 'y'))
            # usm_mask = logical_minus(usm_mask, usm_mask1 | usm_mask2)
            # restore line map
            # usm_mask = np.repeat(usm_mask[..., np.newaxis], 2, axis = -1)
            # edge_map[usm_mask] = False
            # _, lines_x, lines_y, _ = map_to_lines(edge_map, pt_map)

        usm_uncertain, _, _ = update_usm(
            usm_uncertain, usm_updated, remove_only=True)
        need_refine = True
        need_push = True
    else:
        raise ValueError("Unrecognized operation %s" % op)

    # run usm surgery or just reconstruct lines
    if need_refine:
        # only apply usm surgery for user modified regions
        lines_pre, lines_refined_iter, _, _ = refine_topology(
            edge_map, pt_map, usm_iter, lines_x, lines_y, keypts, downsample=False, manual=True)
        lines_refined_list = lines_refined_dict_to_list(lines_refined)
        if len(lines_refined_iter) > 0:
            lines_refined.update(lines_refined_iter)
    else:
        lines_pre = linemap_to_lines(lines_x, lines_y)
        lines_refined_list = lines_refined_dict_to_list(lines_refined)

    if len(lines_refined_list) > 0:
        lines_pre = np.concatenate(
            (lines_pre, np.concatenate(
                lines_refined_list, axis=0)), axis=0)

    # save output result
    if need_push:
        if op == 'modify':
            keypts = keypt_updated
        else:
            for key in keypts:
                keypts[key] = keypts[key] * 2
        UDF_NPZ, NDC_NPZ = usm_push(
            UDF_NPZ, NDC_NPZ, keypts, usm_applied, usm_uncertain, lines_x, lines_y, lines_refined)

    svg_refined = lines_to_svg(
        lines_pre * 2,
        canvas_w * 2,
        canvas_h * 2,
        path.join(
            TEMP_DIR,
            FILE_NAME + "_refine.svg"),
        paths2Drawing=True)
    lines_to_svg(
        lines_pre * 2,
        canvas_w * 2,
        canvas_h * 2,
        path.join(
            TEMP_DIR,
            FILE_NAME + "_refine.svg"))

    end = time.time()
    print(
        "log:\t %s under sample surgery finished in %.2f seconds" %
        (FILE_NAME, end - start))

    return usm_applied, usm_uncertain, keypts, svg_refined


def lines_refined_dict_to_list(lines_refined):
    lines_refined_list = []
    remove_list = []
    for k in lines_refined:
        if len(lines_refined[k]) > 0:
            lines_refined_list.append(lines_refined[k])
        else:
            remove_list.append(k)
    for k in remove_list:
        del lines_refined[k]

    return lines_refined_list


def update_usm(usm, usm_updated, remove_only=False):
    # duplicate usm
    usm = usm.copy()
    usm_iter = np.zeros(usm.shape).astype(int)
    # segment updated usm to regions
    _, regions = cv2.connectedComponents(
        usm_updated.astype(np.uint8), connectivity=4)
    # remove stray usm regions
    ridx, rcount = np.unique(regions, return_counts=True)
    ridx_small = ridx[rcount < 3]
    if len(ridx_small) > 0:
        for r in ridx_small:
            regions[regions == r] = 0
    # find difference between original usm and udpated usm
    dmask = (usm != 0) ^ (regions != 0)
    temp_idx = np.unique(regions[dmask])
    modify_idx = []
    add_idx = []
    next_idx = usm.max() + 1
    for r in temp_idx:
        if r == 0:
            continue
        m_new_region = regions == r
        midx = np.unique(usm[m_new_region])
        midx = midx[midx != 0]
        if len(midx) == 0 and remove_only == False:
            add_idx.append(next_idx)
            usm[m_new_region] = next_idx
            next_idx += 1
            usm_iter[m_new_region] = next_idx
        elif len(midx) == 1:
            modify_idx.append(midx[0])
            usm[usm == midx[0]] = 0
            usm[m_new_region] = midx[0]
            usm_iter[m_new_region] = midx[0]
    return usm, usm_iter, modify_idx


def merge_usm_update(usm, usm_iter):
    pass


def npz_save(path_to_npz_udf, path_to_npz_ndc, udf_npz, ndc_npz):
    np.savez(path_to_npz_udf,
             end_point=udf_npz["end_point"],
             sharp_turn=udf_npz["sharp_turn"],
             junc=udf_npz["junc"],
             usm=udf_npz["usm"],
             usm_applied=udf_npz["usm_applied"],
             usm_uncertain=udf_npz["usm_uncertain"])
    np.savez(path_to_npz_ndc,
             edge_map=ndc_npz["edge_map"],
             pt_map=ndc_npz["pt_map"],
             lines_map_x=ndc_npz["lines_map_x"],
             lines_map_y=ndc_npz["lines_map_y"],
             lines_refined=ndc_npz["lines_refined"])


def usm_read(udf_npz, ndc_npz, idx=-1):
    keypts = {}
    integrity_check(udf_npz, ndc_npz)
    keypts["end_point"] = udf_npz["end_point"][idx].copy() / 2
    keypts["sharp_turn"] = udf_npz["sharp_turn"][idx].copy() / 2
    keypts["junc"] = udf_npz["junc"][idx].copy() / 2
    usm_applied = udf_npz['usm_applied'][idx].copy()
    usm_uncertain = udf_npz['usm_uncertain'][idx].copy()
    edge_map = ndc_npz["edge_map"].copy()
    pt_map = ndc_npz["pt_map"].copy()
    lines_x = ndc_npz["lines_map_x"][idx].copy()
    lines_y = ndc_npz["lines_map_y"][idx].copy()
    if ndc_npz["lines_refined"] is not None:
        lines_refined = ndc_npz["lines_refined"][idx].copy()
    else:
        lines_refined = []
    return keypts, usm_applied, usm_uncertain, edge_map, pt_map, lines_x, lines_y, lines_refined


def usm_pop(udf_npz, ndc_npz):
    integrity_check(udf_npz, ndc_npz)
    keypts = {}
    if len(udf_npz["end_point"]) > 1:
        keypts["end_point"] = udf_npz["end_point"][-1] / 2
        udf_npz["end_point"] = udf_npz["end_point"][:-1]  # pop
        keypts["sharp_turn"] = udf_npz["sharp_turn"][-1] / 2
        udf_npz["sharp_turn"] = udf_npz["sharp_turn"][:-1]  # pop
        keypts["junc"] = udf_npz["junc"][-1] / 2
        udf_npz["junc"] = udf_npz["junc"][:-1]  # pop
        usm_applied = udf_npz['usm_applied'][-1]
        udf_npz["usm_applied"] = udf_npz["usm_applied"][:-1]  # pop
        usm_uncertain = udf_npz['usm_uncertain'][-1]
        udf_npz["usm_uncertain"] = udf_npz["usm_uncertain"][:-1]  # pop
        edge_map = ndc_npz["edge_map"]
        pt_map = ndc_npz["pt_map"]
        lines_x = ndc_npz["lines_map_x"][-1]
        ndc_npz["lines_map_x"] = ndc_npz["lines_map_x"][:-1]  # pop
        lines_y = ndc_npz["lines_map_y"][-1]
        ndc_npz["lines_map_y"] = ndc_npz["lines_map_y"][:-1]  # pop
        assert ndc_npz["lines_refined"] is not None
        lines_refined = ndc_npz["lines_refined"][-1]
        ndc_npz["lines_refined"] = ndc_npz["lines_refined"][:-1]  # pop
    else:
        keypts, usm_applied, usm_uncertain, edge_map, pt_map, lines_x, lines_y, lines_refined = usm_read(
            udf_npz, ndc_npz)
    return keypts, usm_applied, usm_uncertain, edge_map, pt_map, lines_x, lines_y, lines_refined


def append_pts(npz, value, key):
    # pts_list = list(npz[key].squeeze())
    # pts_list.append(list(value))
    # npz[key] = np.array(pts_list)
    pts_list = npz[key]
    pts_list = list(pts_list) 
    if len(pts_list) == 0 and len(value) == 0:
        pts_list = np.array([])
    elif len(pts_list) == 0:
        pts_list = np.array([value])
    elif len(value) == 0:
        # do nothing
        pass
    else:
        pts_list.append(value)
    npz[key] = np.array(pts_list, dtype=object)


def usm_push(
        udf_npz,
        ndc_npz,
        keypts,
        usm_applied,
        usm_uncertain,
        lines_x,
        lines_y,
        lines_refined):
    integrity_check(udf_npz, ndc_npz)
    if len(udf_npz["end_point"]) >= 2:
        usm_pop(udf_npz, ndc_npz)
    udf_npz = dict(udf_npz)
    ndc_npz = dict(ndc_npz)
    append_pts(udf_npz, keypts["end_point"], "end_point")
    append_pts(udf_npz, keypts["sharp_turn"], "sharp_turn")
    append_pts(udf_npz, keypts["junc"], "junc")
    # udf_npz["end_point"] = np.append(udf_npz["end_point"], keypts["end_point"][np.newaxis, ...], axis = 0)
    # udf_npz["sharp_turn"] = np.append(udf_npz["sharp_turn"], keypts["sharp_turn"][np.newaxis, ...], axis = 0)
    # udf_npz["junc"] = np.append(udf_npz["junc"], keypts["junc"][np.newaxis, ...], axis = 0)
    udf_npz["usm_applied"] = np.append(
        udf_npz["usm_applied"], usm_applied[np.newaxis, ...], axis=0)
    udf_npz["usm_uncertain"] = np.append(
        udf_npz["usm_uncertain"], usm_uncertain[np.newaxis, ...], axis=0)
    ndc_npz["lines_map_x"] = np.append(
        ndc_npz["lines_map_x"], lines_x[np.newaxis, ...], axis=0)
    ndc_npz["lines_map_y"] = np.append(
        ndc_npz["lines_map_y"], lines_y[np.newaxis, ...], axis=0)
    append_pts(ndc_npz, lines_refined, "lines_refined")
    # ndc_npz["lines_refined"] = np.append(ndc_npz["lines_refined"], lines_refined[np.newaxis, ...], axis = 0)
    return udf_npz, ndc_npz


def integrity_check(udf_npz, ndc_npz):
    assert len(
        udf_npz["end_point"].shape) == 3 or len(
        udf_npz["end_point"].shape) == 1
    assert len(
        udf_npz["sharp_turn"].shape) == 3 or len(
        udf_npz["sharp_turn"].shape) == 1
    assert len(udf_npz["junc"].shape) == 3 or len(udf_npz["junc"].shape) == 1
    assert len(
        udf_npz["usm_applied"].shape) == 3 or len(
        udf_npz["usm_applied"].shape) == 1
    assert len(
        udf_npz["usm_uncertain"].shape) == 3 or len(
        udf_npz["usm_uncertain"].shape) == 1
    assert len(udf_npz["end_point"]) > 0
    assert len(udf_npz["end_point"]) == len(udf_npz["sharp_turn"])
    assert len(udf_npz["end_point"]) == len(udf_npz["junc"])
    assert len(udf_npz["end_point"]) == len(udf_npz["usm_applied"])
    assert len(udf_npz["end_point"]) == len(udf_npz["usm_uncertain"])

    assert len(ndc_npz["lines_map_x"].shape) == 3
    assert len(ndc_npz["lines_map_y"].shape) == 3
    if ndc_npz["lines_refined"] is not None:
        assert len(ndc_npz["lines_refined"].shape) == 1
    assert len(udf_npz["end_point"]) == len(ndc_npz["lines_map_x"])
    assert len(udf_npz["end_point"]) == len(ndc_npz["lines_map_y"])
    assert len(udf_npz["end_point"]) == len(ndc_npz["lines_refined"])

# step3


def finalize(down_rate=16, bezier=False, rdp_simplify=False):
    global TEMP_DIR
    global FILE_NAME
    global CANVAS_SIZE
    global UDF_NPZ
    global NDC_NPZ
    start = time.time()
    init_temp_dir()
    # assert UDF_NPZ is not None and NDC_NPZ is not None
    # try to read the NPZ file if those two global variables are not initialized
    if UDF_NPZ is None or NDC_NPZ is None:
        path_to_npz_udf = path.join(TEMP_DIR, FILE_NAME+"_udf.npz")
        path_to_npz_ndc = path.join(TEMP_DIR, FILE_NAME+"_ndc.npz")
        UDF_NPZ = dict(np.load(path_to_npz_udf, allow_pickle = True))
        NDC_NPZ = dict(np.load(path_to_npz_ndc, allow_pickle = True))
    keypts, usm_applied, usm_uncertain, edge_map, pt_map, lines_x, lines_y, lines_refined = usm_read(
        UDF_NPZ, NDC_NPZ)
    # usm_applied = np.zeros(usm_applied.shape)
    lines_pre = downsample_ndc(
        edge_map,
        pt_map,
        keypts,
        usm_applied,
        lines_x,
        lines_y,
        down_rate=down_rate)
    lines_refined_list = lines_refined_dict_to_list(lines_refined)
    if len(lines_refined_list) > 0:
        lines_pre = np.concatenate(
            (lines_pre, np.concatenate(
                lines_refined_list, axis=0)), axis=0)
    canvas_h, canvas_w = CANVAS_SIZE
    lines_to_svg(
        lines_pre * 2,
        canvas_w * 2,
        canvas_h * 2,
        path.join(
            TEMP_DIR,
            FILE_NAME + "_final.svg"))
    simplify_SVG(
        path.join(
            TEMP_DIR,
            FILE_NAME +
            "_final.svg"),
        keypts,
        bezier=bezier,
        rdp_simplify=rdp_simplify)
    end = time.time()
    print(
        "log:\t %s post processing finished in %.2f seconds" %
        (FILE_NAME, end - start))

def vis_stroke(strokes, canvas_size, save_path):
    h, w = canvas_size
    # get random colors for each stroke
    colors = np.random.randint(0, 255, size=(len(strokes), 3))
    colors = [tuple(c) for c in colors]
    # save to file
    # attributes = [{"fill":'none', "stroke-width":'1', "stroke-linecap":"round"}] * len(strokes)
    wsvg(strokes,
         colors=colors,
         stroke_widths=[2] * len(strokes),
         dimensions=(w, h),
         filename=save_path)


if __name__ == "__main__":
    out_folder = 'svg_full'
    parser = argparse.ArgumentParser(
        description="End to end testing of DeepSketch")
    parser.add_argument(
        "--input",
        "-i")
    parser.add_argument(
        "--model_udf",
        "-m1",
        default="./pretrained/udf_full.pth")
    parser.add_argument(
        "--model_ndc",
        "-m2",
        default="./pretrained/ndc_full.pth")
    parser.add_argument("--output", "-o")
    parser.add_argument("--device", "-d ", default='cpu')
    '''
    Pre processing
    '''
    parser.add_argument("--line_extractor", action='store_true', help="enable anime2sketch line extractor before vectorize")
    parser.add_argument("--thin", action='store_true', help="enable line thinner before vectorize")
    parser.add_argument(
        "--resize_to", 
        default=512, 
        type=int, 
        help="downsacle image to its long edge equals resize_to, this option will not work if device is set to cpu")
    '''
    Post processing
    '''
    parser.add_argument("--bezier", action="store_true",
                        help='enable bezier curve fitting')
    parser.add_argument("--refine", action="store_true",
                        help='enable dual contouring refinement')
    parser.add_argument("--rdp", action="store_true",
                        help='enable Ramer–Douglas–Peucker similification')
    parser.add_argument("--eps", default=0.4, type=float)
    # too large downsample will affect the final output quality, try to decrease this if needed
    # down rate can only be even numbers
    parser.add_argument("--down_rate", default=8, type=int,
                        help='ratio for dual contouring downsampling')
    parser.add_argument(
        "--skip_len",
        default=4,
        type=int,
        help='drop lines if it contains less than skip_len segements')
    args = parser.parse_args()
    if args.output is None:
        args.output = args.input

    device = args.device
    print("log:\twork mode is set to %s" % device)
    model_udf, args_udf, model_ndc, args_ndc = load_model(args, device)
    # record running time into log
    with open(path.join(args.output, 'sketchvg_log.txt'), 'w') as f:
        for img_name in os.listdir(args.input):
            if ("png" in img_name or "jpg" in img_name or "bmp" in img_name) and "res" not in img_name:
                name, _ = os.path.splitext(img_name)
                if os.path.exists(
                    path.join(
                        args.output,
                        out_folder,
                        name +
                        "_final.svg")):
                    continue
                if 'res' in img_name or 'keypt' in img_name or 'usm' in img_name:
                    continue
                print("log:\topening %s" % img_name)
                f.write("Processing " + img_name + ' ...')
                f.flush()
                # vectorization
                time_begin = time.time()
                time_begin_ = time.time()
                # load input data
                if device == 'cuda':
                    # if True:
                    ww, hh = Image.open(path.join(args.input, img_name)).size
                    longer = ww if ww > hh else hh
                    resize = True if longer > args.resize_to else False
                    if resize:
                        ratio = args.resize_to / longer
                        print("log:\timage size (%dx%d) is too large, resize to (%dx%d)" % (
                            hh, ww, int(hh * ratio), int(ww * ratio)))
                else:
                    resize = False
                    ratio = 1
                img, img_np, canvas_size, _ = load_img(path.join(args.input, img_name), device, args_udf.up_scale,
                                                       thin=Thinner(
                                                           use_gpu=True if device == 'cuda' else False) if args.thin else None,
                                                       line_extractor=create_model('default').to(
                                                           device) if args.line_extractor else None,
                                                       resize=resize,
                                                       path_to_out=path.join(
                                                           args.output, out_folder),
                                                       resize_to=args.resize_to,
                                                       ui_mode=False)
                tensor_h, tensor_w = img.shape[2], img.shape[3]

                # predict UDF from sketch
                udf_topo_pre, usm_pre_, keypt_pre_dict, keypt_pre_np = predict_UDF(
                    img, img_np, model_udf, None, name)

                '''
                For Debug
                '''
                # keypt_pre_end, _ = vis_pt_single(img_np, udfs_pre[0])
                # keypt_pre_sharp, _ = vis_pt_single(img_np, udfs_pre[1])
                # keypt_pre_junc, _ = vis_pt_single(img_np, udfs_pre[2])

                # predict SVG from UDF
                linemap_pre_x, linemap_pre_y, pt_map_pre, edge_maps_pre_xy = predict_SVG(
                    udf_topo_pre, model_ndc, args_udf, args_ndc, (tensor_h, tensor_w), out_path=path.join(
                        args.output, out_folder), refine=args.refine, name=name, to_npz=False)

                time_end = time.time()
                f.write("vectorized in %f Seconds, " % (time_end - time_begin))
                f.flush()

                # USM surgery
                time_begin = time.time()
                lines_pre, _, usm_applied, usm_uncertain = refine_topology(
                    D(edge_maps_pre_xy),
                    pt_map_pre,
                    usm_pre_,
                    linemap_pre_x,
                    linemap_pre_y,
                    keypt_pre_dict,
                    down_rate=args.down_rate,
                    downsample=True,
                    full_auto_mode=True)
                usm_applied = usm_applied != 0
                lines_to_svg(
                    lines_pre * 2,
                    tensor_w * 2,
                    tensor_h * 2,
                    path.join(
                        path.join(
                            args.output,
                            out_folder),
                        name + "_refine.svg"))
                time_end = time.time()
                f.write(
                    "USM surgery in %f Seconds, " %
                    (time_end - time_begin))
                f.flush()
                # stroke traversal
                time_begin = time.time()
                simplify_SVG(
                    path.join(
                        path.join(
                            args.output,
                            out_folder),
                        name + "_refine.svg"),
                    keypt_pre_dict,
                    bezier=args.bezier,
                    rdp_simplify=args.rdp,
                    epsilon=args.eps,
                    skip_len=args.skip_len)
                time_end = time.time()
                f.write(
                    "stroke grouping in %f Seconds, " %
                    (time_end - time_begin))
                f.write("total in %f Seconds\n" % (time_end - time_begin_))
                f.flush()
                ## visualize
                svg_pre = path.join(
                    path.join(
                        args.output,
                        out_folder),
                    name + "_raw.svg")
                res_pre = svg_to_numpy(svg_pre)
                if res_pre is None:
                    res_pre = np.ones((tensor_h, tensor_w, 3)) * 255
                else:
                    res_pre = res_pre[..., np.newaxis].repeat(3, axis=-1)
                h, w = res_pre.shape[0], res_pre.shape[1]
                img_np = cv2.resize(
                    img_np, (w, h), interpolation=cv2.INTER_AREA)
                keypt_pre_np = cv2.resize(
                    keypt_pre_np, (w, h), interpolation=cv2.INTER_AREA)

                res_pre_keypt = blend_skeletons(res_pre, (usm_applied.astype(
                    int), (usm_uncertain != 0).astype(int)), usm_mode=True)
                res_pre_keypt = add_keypt(
                    res_pre_keypt, keypt_pre_dict, (tensor_h, tensor_w))
                keypt_pre_list = []
                keypt_to_color = {
                    "end_point": "green",
                    "sharp_turn": "red",
                    "junc": "blue"}
                color_list = []
                for key in keypt_pre_dict:
                    for i in range(len(keypt_pre_dict[key])):
                        keypt_pre_list.append(
                            complex(*keypt_pre_dict[key][i]))
                        color_list.append(keypt_to_color[key])
                pt_num = len(keypt_pre_list)
                assert len(keypt_pre_list) == len(color_list)
                Image.fromarray(
                    res_pre_keypt.astype(
                        np.uint8)).save(
                    path.join(
                        path.join(
                            args.output,
                            out_folder),
                        name + "_usm.png"))
