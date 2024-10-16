from pathlib import Path as P
import os, sys
directory = os.path.realpath(os.path.dirname(__file__))
directory = str(P(directory).parent)
if directory not in sys.path:
    sys.path.append(directory)

import cv2
import pandas as pd
import numpy as np
import lxml.etree as ET
from PIL import Image
from os.path import join, exists
from dataset.preprocess import rasterize
from PIL import Image
from utils.metric_multiple import run
import matplotlib.pyplot as plt
import svgpathtools
from svgpathtools import CubicBezier, Line, Path, wsvg, svg2paths
from utils.svg_tools import open_svg_flatten, get_line_approximation, flatten_paths_to_pts_array, build_edge, build_connectivity
from functools import reduce
from losses import chamfer_distance_vec
import multiprocessing as mp
import random
import seaborn as sns

def compute_one_res_metric(png, path_to_input, path_to_res, method_name):
    gt_path = join(path_to_input,'svg', png.replace('.png', '.svg'))
    if 'sketchvg' in method_name or 'wo' in method_name:
        res_path = join(path_to_res, 'svg', png.replace('.png', '_raw.svg'))
    else:
        res_path = join(path_to_res, 'svg', png.replace('.png', '.svg'))
    keypt_path = join(path_to_input, 'keypt', png.replace('.png', '.npz'))
    
    print("log:\topening %s"%res_path)
    if exists(gt_path) and exists(res_path):
        gt_svg, canvas_gt = open_svg_flatten(gt_path)
        try:
            res_svg, canvas_res = open_svg_flatten(res_path)
        except:
            return png, [None, None, None, None, None], 1
        if canvas_gt != canvas_res:
            print("log:\tfound unmatched size of %s"%png)
            raise ValueError()
        gt_len = 0
        need_divid = False
        ## compute stroke length
        for p in gt_svg:
            gt_len += p.length()
        res_len = 0
        # get stroke number from sketch
        res_path_num = 1 if (len(res_svg)  == 0) else len(svg2paths(res_path)[0])
        for p in res_svg:
            if isinstance(p, svgpathtools.QuadraticBezier): need_divid = True
            res_len += p.length()
        # record to result dict
        diff_len = abs(gt_len - res_len)
        ## compute vector chamfer distance
        res_pt = get_line_approximation(res_svg, sample_dist = 0.1, mode = 'pt')
        # svgpathtools.wsvg(nodes = res_pt, node_radii= [0.05]*len(res_pt), dimensions = (canvas_gt[1], canvas_gt[0]), filename = "test.svg")
        gt_pt = get_line_approximation(gt_svg, sample_dist = 0.1, mode = 'pt')
        chamfer = chamfer_distance_vec(res_pt, gt_pt)
        ## compute output's vertex valence
        # get ground truth keypoint valences
        compute_valence = False
        # try:
        #     keypt = np.load(keypt_path)
        #     compute_valence = True
        # except:
        #     pass
        if compute_valence:
            # get vertex valence from output vector image
            if method_name == 'virtual':
                # detect_svg(path_to_svg, 0.0001, 0.00095, 0.007, path_to_result=res_path.replace('.svg', '.npz'), to_svg=False, multiproc = False)
                keypt_res = np.load(res_path.replace('.svg', '.npz').replace('svg', 'keypt'))
                diff_v1 = abs(len(keypt_res['end_point'] - len(keypt['end_point'])))
                diff_v3 = abs(len(keypt_res['t_junction'] - len(keypt['t_junction'])))  + abs(len(keypt_res['x_junction'] - len(keypt['x_junction'])))
            else:
                pts_array = flatten_paths_to_pts_array(res_svg)
                edges, pts_array = build_edge(pts_array)
                _, valence = build_connectivity(pts_array, edges)
                diff_v1 = abs((valence == 1).sum() - len(keypt['end_point']))
                diff_v3 = abs((valence == 3).sum() - len(keypt['t_junction'])) + abs((valence >= 4).sum() - len(keypt['x_junction']))
            return png, [diff_len, chamfer, diff_v1, diff_v3], res_path_num
        else:
            return png, [diff_len, chamfer, None, None], res_path_num
    else:
        return png, [None, None, None, None], 1

def N(udf):
    return (udf / udf.max() * 255).clip(0, 255).astype(np.uint8)

def to_udf(pts, size):
    h, w =  size
    res = np.ones(size)
    res[tuple(pts.T.astype(int))] = 0
    return N(cv2.distanceTransform(res.astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE))

def normalize(root, threshold = 0.002, color = "#000000", cap = "round", join = "round", canvas_size = None, ratio = 1):
    '''
    Given:
        root, the root xml element of svg
        threshold, stroke width which equals thershold * length of artboard longer side
        color, stroke color, default is black "#000000"
        cap, stroke cap type
        join, stroke joint type
    Return:
        root of normalized svg
    '''

    # get namespace
    tag_root = root.tag
    namespace = tag_root[tag_root.find('{'):tag_root.rfind('}')+1]
    # drop image layers
    try:
        # get all strokes in root
        # only XPath could work, I don't know why
        strokes = root.findall(".//" + namespace + "path")
        polygons = root.findall(".//" + namespace + "polygon")
        polylines = root.findall(".//" + namespace + "polyline")
        styles = root.findall(".//" + namespace + "style")
        lines = root.findall(".//" + namespace + "line")
        ellipses = root.findall(".//" + namespace + "ellipse")
        rects = root.findall(".//" + namespace + "rect")
        circles = root.findall(".//" + namespace + "circle")
        # this is for any special case that all elements above are not in the svg
        g = root.findall(".//" + namespace + "g")

        # get the size of artboard
        if canvas_size is not None:
            height, width = canvas_size
            # if ratio == 1:
            #     assert height == int(root.attrib['height'].strip('px'))
            #     assert width == int(root.attrib['width'].strip('px'))
            root.attrib['width'] = str(width)
            root.attrib['height'] = str(height)
            root.attrib['viewBox'] = "0 0 %d %d"%(width, height)
        else:
            width = int(root.attrib['width'].strip('px'))
            height = int(root.attrib['height'].strip('px'))

        root.attrib["style"] = "background-color:white"
        # stroke_width = width if width > height else height
        # stroke_width = stroke_width * threshold
        stroke_width = 0.5
        for style in styles:
            # split styles
            style_text = style.text.strip("\n")
            style_lines = []
            end = style_text.find("}")
            while end != -1:
                style_lines.append(style_text[:end+1])
                style_text = style_text[end+1:]
                end = style_text.find("}")        
            for i in range(len(style_lines)):
                st = style_lines[i].strip("\t")
                if "stroke-width" in style_lines[i]:
                    style_lines[i] = st[: st.find("stroke-width:") + len("stroke-width:")] + '%f'%stroke_width + st[st.find(';', st.find("stroke-width")) :]
                else:
                    style_lines[i] = st[: st.find("}")] + 'stroke-width:%f;'%stroke_width + "}"
                st = style_lines[i].strip("\t")
                if "stroke-linecap" in style_lines[i]:
                    style_lines[i] = st[: st.find("stroke-linecap:") + len("stroke-linecap:")] + cap + st[st.find(';', st.find("stroke-linecap")) :]
                else:
                    style_lines[i] = st[: st.find("}")] + 'stroke-linecap:%s;'%cap + "}"
                st = style_lines[i].strip("\t")
                if "stroke-linejoin" in style_lines[i]:
                    style_lines[i] = st[: st.find("stroke-linejoin:") + len("stroke-linejoin:")] + join + st[st.find(';', st.find("stroke-linejoin")) :]
                else:
                    style_lines[i] = st[: st.find("}")] + 'stroke-linejoin:%s;'%join + "}"
                st = style_lines[i].strip("\t")
                if "stroke:" in style_lines[i]:
                    style_lines[i] = st[: st.find("stroke:") + len("stroke:")] + color + st[st.find(';', st.find("stroke:")) :]
                else:
                    style_lines[i] = st[: st.find("}")] + 'stroke:%s;'%color + "}"
            style.text = '\n'.join(style_lines)
        
        # then change every stroke style individually
        for elements in [strokes, polylines, polygons, lines, ellipses, rects, circles, g]:
            for element in elements:
                element.attrib["stroke"] = color
                element.attrib["stroke-width"] = str("%f"%stroke_width)
                element.attrib["stroke-linecap"] = cap
                element.attrib["stroke-linejoin"] = join
                element.attrib["fill"] = "none"
                element.attrib.pop("style", None)
                if ratio != 1:
                    # resize each stroke
                    d = element.attrib['d'].split(' ')
                    for i in range(len(d)):
                        if ',' in d[i]:
                            value = d[i].split(',')
                            assert len(value) == 2
                            for j in range(len(value)):
                                value[j] = str(float(value[j]) * ratio)
                            d[i] = ','.join(value)
                    element.attrib['d'] = ' '.join(d)
                    for key in element.attrib:
                        if key == "points":
                            points = element.attrib[key].split(" ")
                            for i in range(len(points)-1, -1, -1):
                                if "nan" in points[i]:
                                    points.pop(i)
                            element.attrib[key] = " ".join(points)
        return True
    except Exception as e:
        print(str(e))
        return False

def read_log(fpath, fname, png_dict):
    with open(join(fpath, fname), 'r') as f:
        lines = f.readlines()
    for l in lines:
        if "Processing" not in l: continue
        # get file name 
        fname = l[:l.find('.png')+4].lstrip("Processing ")
        assert fname in png_dict
        if 'timeout' in l: continue
        # get running time
        l_ = l.split(" ")
        run_time = float(l_[-2])
        if 'sketchvg' in fpath:
            if 'curve' in l:
                curve_time = float(l_[-11])
                png_dict[fname] = [(run_time, curve_time)]
            else:
                png_dict[fname] = [(run_time, None)]
        else:
            png_dict[fname] = [run_time]

if __name__ == "__main__":
    __spec__ = None
    # re-normalize svg files to thicker stroke
    if False:
        process_benchmark = False
        if process_benchmark:
            path_in = '../data/benchmark/256_long/svg'
        else:
            path_in = '../experiments/08.exp_compairison/03.poly/02.cluster/uncleaned/svg'
            path_in_png = '../data/benchmark/uncleaned'
        for svg in os.listdir(path_in):
            # and "raw" not in svg and "refine" not in svg and "vis" not in svg
            if '.svg' in svg and 'keypt' not in svg and "vis" not in svg:
                if process_benchmark: 
                    _, canvas_size = open_svg_flatten(join(path_in, svg))
                    h, w = canvas_size
                    edge = h if h > w else w
                    ratio = 256 / edge
                    canvas_size = (int(h * ratio + 0.5), int(w * ratio + 0.5))
                else:
                    img = np.array(Image.open(join(path_in_png, svg.replace('.png.svg', '.png').replace('.svg', '.png').replace("_final", '').replace("_refine", "").replace("_raw", ""))))
                    canvas_size = (img.shape[0], img.shape[1])
                    ratio = 0.5 if "sketchvg" in path_in else 1
                print('log:\topening %s'%svg)
                with open(join(path_in, svg), 'r') as f:
                    lines = f.readlines()
                    # data = read_in_chunks(f)

                for i in range(len(lines) - 1, -1 , -1):
                    if '\n' == lines[i]:
                        lines.pop(i)

                if len(lines) == 0: continue
                
                for i in range(len(lines) - 1, -1 , -1):
                    if '<image' in lines[i] and '<g>' in lines[i] and '</g>' not in lines[i]:
                        lines.pop(i+1)
                        lines.pop(i)
                    elif '<image' in lines[i]:
                        lines.pop(i)

                if '</svg>' not in lines[-1]:
                    print('log:\tincompelet file, drop the last line')
                    lines.pop(-1)
                    lines.append('</svg>')

                with open(join(path_in, svg), 'w') as f:
                    f.write("\n".join(lines))

                tree = ET.parse( join(path_in, svg) )
                if normalize(tree.getroot(), canvas_size = canvas_size, ratio = ratio) is False:
                    print("Error:\tstroke normlize error")
                tree.write(join(path_in, svg))
            # import pdb
            # pdb.set_trace()

    # try to connect line in benchmark svg as much as possible
    # this is necessary when rasterizing them into png images
    if False:
        path_in = '../data/benchmark/512_short/svg'
        path_out = '../data/benchmark/512_short/svg_connected'
        res = []
        def has_pt(pt, end_pts_s, end_pts_e):
            s = np.abs(end_pts_s - pt) < 0.01
            e = np.abs(end_pts_e - pt) < 0.01
            return s.any() or e.any()

        def find_pt(pt, pts):            
            return np.abs(pts - pt) < 0.01
        
        def pop_pt(idx, end_pts_s, end_pts_e):
            start = end_pts_s[idx]
            end = end_pts_e[idx]
            end_pts_s[idx] = complex(-1, -1)
            end_pts_e[idx] = complex(-1, -1)
            return start, end

        for svg in os.listdir(path_in):
            if '.svg' not in svg: continue
            print("log:\topening %s"%svg)
            paths, canvas_size = open_svg_flatten(join(path_in, svg))
            h, w = canvas_size
            # extract all end points
            end_pts_s = []
            end_pts_e = []
            paths_ = []
            paths__ = []
            for p in paths:
                # we only conncect bezier curves and lines
                if isinstance(p, CubicBezier) == False and isinstance(p, Line) == False:
                    paths__.append(p)
                else:
                    end_pts_s.append(p.start)
                    end_pts_e.append(p.end)
                    paths_.append(p)

            assert len(end_pts_s) == len(end_pts_e)
            res = []
            end_pts_s = np.array(end_pts_s)
            end_pts_e = np.array(end_pts_e)
            
            while (end_pts_s != complex(-1, -1)).any():
                stroke = Path()
                # random select one path
                idx = np.random.choice(np.where(end_pts_s != complex(-1, -1))[0])
                stroke.append(paths_[idx].reversed())
                start, end = pop_pt(idx, end_pts_s, end_pts_e)
                # search from start point
                while has_pt(start, end_pts_s, end_pts_e):
                    hit_start = find_pt(start, end_pts_s)
                    hit_end = find_pt(start, end_pts_e) 
                    if hit_start.any():
                        # choose the first hit
                        idx_ = np.where(hit_start)[0][0]
                        stroke.append(paths_[idx_])
                        _, start = pop_pt(idx_, end_pts_s, end_pts_e)
                    else:
                        assert hit_end.any()
                        idx_ = np.where(hit_end)[0][0]
                        stroke.append(paths_[idx_].reversed())
                        start, _ = pop_pt(idx_, end_pts_s, end_pts_e)
                # search from end point
                stroke = stroke.reversed()
                while has_pt(end, end_pts_s, end_pts_e):
                    hit_start = find_pt(end, end_pts_s)
                    hit_end = find_pt(end, end_pts_e)
                    if hit_start.any():
                        # choose the first hit
                        idx_ = np.where(hit_start)[0][0]
                        stroke.append(paths_[idx_])
                        _, end = pop_pt(idx_, end_pts_s, end_pts_e)
                    else:
                        assert hit_end.any()
                        idx_ = np.where(hit_end)[0][0]
                        stroke.append(paths_[idx_].reversed())
                        end, _ = pop_pt(idx_, end_pts_s, end_pts_e)
                res.append(stroke)
            res = res + paths__
            wsvg(res, stroke_widths = [0.5]*len(res), dimensions = (w, h), filename = join(path_out, svg))

    # add paper texutre to generated benchmark png
    if False:
        path_in = '../data/benchmark/256_long/png_hard'
        path_out = '../data/benchmark/256_long/png_hard_texture'
        # path_in = 'F:/2.Projects/03.Sketch/00.paper/sketchvg/source_png_for_figs'
        # path_out = 'F:/2.Projects/03.Sketch/00.paper/sketchvg'
        BG_COLOR = 209
        bg_color = np.random.randint(150, BG_COLOR)
        BG_SIGMA = 5
        MONOCHROME = 1
        def blank_image(width=1024, height=1024, background=bg_color):
            """
            It creates a blank image of the given background color
            """
            img = np.full((height, width, MONOCHROME), background, np.uint8)
            return img

        def add_noise(img, sigma=BG_SIGMA):
            """
            Adds noise to the existing image
            """
            width, height, ch = img.shape
            n = noise(width, height, sigma=sigma)
            img = img + n
            return img.clip(0, 255)

        def noise(width, height, ratio=1, sigma=BG_SIGMA):
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

        def texture(image, sigma=BG_SIGMA, turbulence=10):
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
                result += noise(cols, rows, ratio, sigma=sigma)
                i += 1
            cut = np.clip(result, 0, 255)
            return cut.astype(np.uint8)
        for png in os.listdir(path_in):
            if '.png' not in png: continue
            print("log:\topening %s"%png)
            img = np.array(Image.open(join(path_in, png)))
            # add a random back ground to the image
            # make sure the returned image only have RGB channels
            h, w, _ = img.shape
            img_rgb = img[:, :, 0:3]
            alpha = img[:, :, 3]
            # alpha[alpha == 255] = 0
            alpha = np.expand_dims(alpha, -1).astype(float) / 255
            # generate background and mix it back into image
            dice = np.random.uniform()
            edge = h if h > w else w
            blank = blank_image(edge, edge, background=np.random.randint(BG_COLOR, 245))
            bg = texture(blank, sigma=np.random.randint(1, BG_SIGMA), 
                turbulence=np.random.randint(5, 10))[:h, :w, ...]
            Image.fromarray((img_rgb * alpha + bg * (1 - alpha)).mean(axis = -1).astype(np.uint8)).save(join(path_out, png))

    # convert benchmark svg to pngs
    if False:
        path_in = '../experiments/08.exp_compairison/01.sketchvg_ver1/512_short/svg'
        path_out = '../experiments/08.exp_compairison/01.sketchvg_ver1/512_short/png'
        path_out2x = '../experiments/08.exp_compairison/01.sketchvg_ver1/512_short/png2x'
        path_out4x = '../experiments/08.exp_compairison/01.sketchvg_ver1/512_short/png4x'
        # path_in = '../data/benchmark/512_short/svg'
        # path_out = '../data/benchmark/512_short/png'
        # path_out2x = '../data/benchmark/512_short/png2x'
        # path_out4x = '../data/benchmark/512_short/png4x'

        for svg in os.listdir(path_in):
            if '.svg' in svg and 'keypt' not in svg:
                print('log:\topening %s'%svg)
                rasterize(join(path_in, svg),join(path_out, svg.replace('.svg', '.png')), dpi = 96)
                rasterize(join(path_in, svg),join(path_out2x, svg.replace('.svg', '.png')), dpi = 192)
                rasterize(join(path_in, svg),join(path_out4x, svg.replace('.svg', '.png')), dpi = 384)

    # visualize one ground turth
    if False:
        path = '../experiments/07.exp_for_fig/0000963.npz'
        p = '../experiments/07.exp_for_fig'
        npz = np.load(path)
        udf = npz['udf']
        Image.fromarray(N(udf)).save(join(p, 'udf.png'))

        usm = npz['under_sampled']
        usm = N(cv2.distanceTransform((~usm).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE))
        Image.fromarray(cv2.applyColorMap(255 - usm, cv2.COLORMAP_INFERNO )).save(join(p, 'usm.png'))

        end = npz['end_point']
        Image.fromarray(to_udf(end, udf.shape)).save(join(p, 'end.png'))

        st = npz['sharp_turn']
        Image.fromarray(to_udf(st, udf.shape)).save(join(p, 'st.png'))

        t = npz['T']
        Image.fromarray(to_udf(t, udf.shape)).save(join(p, 't.png'))

        x = npz['X']
        Image.fromarray(to_udf(x, udf.shape)).save(join(p, 'x.png'))

    # read comparison results and compute metrics, read running time log, save all results into one cvs file
    if True:
        # gt path
        path_in1 = '../data/benchmark/256_long/'
        path_in2 = '../data/benchmark/384_long/'
        path_in3 = '../data/benchmark/512_long/'
        path_in4 = '../data/benchmark/768_long/'
        path_in5 = '../data/benchmark/1024_long/'
        path_in = [path_in1, path_in2, path_in3, path_in4, path_in5]

        ## full benchmark
        if False:
            path_vec_sketchvg2 = '../experiments/08.exp_compairison/01.sketchvg_ver1/512_long_4.5w_w3'
            path_vec_sketchvg3 = '../experiments/08.exp_compairison/01.sketchvg_ver1/512_long_4.5_w3_cluster'
            path_vec_mac_w3 = '../experiments/08.exp_compairison/01.sketchvg_ver1/512_long_mac_w3'
            path_vec_mac_w8 = '../experiments/08.exp_compairison/01.sketchvg_ver1/512_long_mac_w8'
        path_vec_sketchvg1 = '../experiments/08.exp_compairison/01.sketchvg_ver1/02.cluster/256_long'
        path_vec_sketchvg2 = '../experiments/08.exp_compairison/01.sketchvg_ver1/02.cluster/384_long'
        path_vec_sketchvg3 = '../experiments/08.exp_compairison/01.sketchvg_ver1/02.cluster/512_long'
        path_vec_sketchvg4 = '../experiments/08.exp_compairison/01.sketchvg_ver1/02.cluster/768_long'
        path_vec_sketchvg5 = '../experiments/08.exp_compairison/01.sketchvg_ver1/02.cluster/1024_long'
        path_vec_keypt1 = '../experiments/08.exp_compairison/02.keypt/02.cluster/256_long'
        path_vec_keypt2 = '../experiments/08.exp_compairison/02.keypt/02.cluster/384_long'
        path_vec_keypt3 = '../experiments/08.exp_compairison/02.keypt/02.cluster/512_long'
        path_vec_keypt4 = '../experiments/08.exp_compairison/02.keypt/02.cluster/768_long'
        path_vec_keypt5 = '../experiments/08.exp_compairison/02.keypt/02.cluster/1024_long'
        path_vec_poly1 = '../experiments/08.exp_compairison/03.poly/02.cluster/256_long'
        path_vec_poly2 = '../experiments/08.exp_compairison/03.poly/02.cluster/384_long'
        path_vec_poly3 = '../experiments/08.exp_compairison/03.poly/02.cluster/512_long'
        path_vec_poly4 = '../experiments/08.exp_compairison/03.poly/02.cluster/768_long'
        path_vec_poly5 = '../experiments/08.exp_compairison/03.poly/02.cluster/1024_long'
        path_vec_virtual1 = '../experiments/08.exp_compairison/04.virtual/02.cluster/256_long'
        path_vec_virtual2 = '../experiments/08.exp_compairison/04.virtual/02.cluster/384_long'
        path_vec_virtual3 = '../experiments/08.exp_compairison/04.virtual/02.cluster/512_long'
        path_vec_virtual4 = '../experiments/08.exp_compairison/04.virtual/02.cluster/768_long'
        path_vec_virtual5 = '../experiments/08.exp_compairison/04.virtual/02.cluster/1024_long'
        path_to_res = [
            path_vec_sketchvg1, path_vec_sketchvg2, path_vec_sketchvg3, path_vec_sketchvg4, path_vec_sketchvg5,
            path_vec_keypt1, path_vec_keypt2, path_vec_keypt3, path_vec_keypt4, path_vec_keypt5,
            path_vec_poly1, path_vec_poly2, path_vec_poly3, path_vec_poly4, path_vec_poly5,
            path_vec_virtual1, path_vec_virtual2, path_vec_virtual3, path_vec_virtual4, path_vec_virtual5,
            ]
        method_name = [
            'sketchvg256','sketchvg384','sketchvg512','sketchvg768','sketchvg1024', 
            'keypt256','keypt384','keypt512','keypt768','keypt1024', 
            'poly256','poly384','poly512','poly768','poly1024', 
            'virtual256','virtual384','virtual512','virtual768','virtual1024'
            ]
        log_name = [
            'sketchvg_log.txt','sketchvg_log.txt','sketchvg_log.txt','sketchvg_log.txt','sketchvg_log.txt', 
            'keypt_vector_log.txt','keypt_vector_log.txt','keypt_vector_log.txt','keypt_vector_log.txt','keypt_vector_log.txt',
            'poly_vector_log.txt','poly_vector_log.txt','poly_vector_log.txt','poly_vector_log.txt','poly_vector_log.txt',
            'virtual_sketching_log.txt','virtual_sketching_log.txt','virtual_sketching_log.txt','virtual_sketching_log.txt','virtual_sketching_log.txt'
            ]
        method_name_dict = {"sketchvg_ver1": "Ours", "keypt":"[Puhachov 2021]", "poly":"[Bessmeltsev 2019]", "virtual":"[Mo 2021]"}


        ## basic benchmark
        # path_vec_sketchvg1 = '../experiments/08.exp_compairison/01.sketchvg_ver1/512_long_basic_w8'
        # path_vec_sketchvg2 = '../experiments/08.exp_compairison/01.sketchvg_ver1/512_long_basic'
        # path_vec_keypt = '../experiments/08.exp_compairison/02.keypt/512_long_basic'
        # path_vec_poly = '../experiments/08.exp_compairison/03.poly/512_long_basic'
        # path_vec_virtual = '../experiments/08.exp_compairison/04.virtual/512_long_basic'
        # path_to_res = [path_vec_sketchvg1, path_vec_sketchvg2, path_vec_keypt, path_vec_poly, path_vec_virtual]
        # method_name = ['sketchvg', 'sketchvg', 'keypt', 'poly', 'virtual']
        # log_name = ['sketchvg_log.txt', 'sketchvg_log.txt', 'keypt_vector_log.txt', 'poly_vector_log.txt', 'virtual_sketching_log.txt']

        ## ablation
        # path_vec_womsb = '../experiments/08.exp_compairison/01.sketchvg_ver1/Abl_woMSB'
        # path_vec_woskel = '../experiments/08.exp_compairison/01.sketchvg_ver1/Abl_woSKEL'
        # path_vec_woup = '../experiments/08.exp_compairison/01.sketchvg_ver1/Abl_woUP'
        # path_vec_refine = '../experiments/08.exp_compairison/01.sketchvg_ver1/Abl_woREFINE'
        # path_to_res = [path_vec_mac_w3, path_vec_mac_w8, path_vec_womsb, path_vec_woskel, path_vec_woup, path_vec_refine]
        # method_name = ['Mac_w3', 'Mac_w8', 'woMSB', 'woSKEL', 'woUP', "woREFINE"]
        # log_name = ['sketchvg_log.txt', 'sketchvg_log.txt', 'sketchvg_log.txt', 'sketchvg_log.txt', 'sketchvg_log.txt', 'sketchvg_log.txt']
        def eval_and_record(path_to_res, path_to_input, method_name, log_name, overwrite = False):
            path_to_csv = join(path_to_res, "%s_metrics_vec.csv"%method_name)
            if exists(path_to_csv) and overwrite == False: 
                df_res = pd.read_csv(path_to_csv)
                return df_res, df_res["diff_len"].count() / df_res["diff_len"].size
            if overwrite:
                print("warning:\toverwrite mode is on, will regenerate %s"%join(path_to_res, "%s_metrics_vec.csv"%method_name))
            print("log:\tcomputing metircs for %s"%method_name)
            # initial empty dataframe
            df_res = pd.DataFrame()
            png_list = os.listdir(join(path_to_input, 'png_hard_texture'))
            png_list.sort()
            df_res['input'] = png_list
            png_dict = {}
            for png in png_list:
                png_dict[png] = None

            # read log as dictionary
            read_log(path_to_res, log_name, png_dict)

            # compute vector metrics
            png_list = df_res['input']
            all_combos = []
            results = []
            for png in png_list:
                all_combos.append([png, path_to_input, path_to_res, method_name])
                results.append(compute_one_res_metric(png, path_to_input, path_to_res, method_name))
            # with mp.Pool(8) as pool:
            #     results = pool.starmap(compute_one_res_metric, all_combos)
                
            diff_len = []
            diff_v1 = []
            diff_v3 = []
            chamfer = []
            run_time = []
            run_time_ = []
            curve_time = []
            for res in results:
                png = res[0]
                if png_dict[png] is not None:
                    # running time, diff_len, chamfer, diff_v1, diff_v3, diff_v4, diff_star
                    if "sketchvg" not in path_to_res:
                        run_time.append(png_dict[png][0] / res[-1])
                        run_time_.append(png_dict[png][0])
                        curve_time.append(None)
                    else:
                        run_time.append(png_dict[png][0][0] / res[-1])
                        run_time_.append(png_dict[png][0][0])
                        curve_time.append(png_dict[png][0][1])

                    diff_len.append(res[1][0])
                    chamfer.append(res[1][1])
                    diff_v1.append(res[1][2])
                    diff_v3.append(res[1][3])
                    
                else:
                    run_time_.append(None)
                    run_time.append(None)
                    diff_len.append(None)
                    diff_v1.append(None)
                    diff_v3.append(None)
                    chamfer.append(None)
                    curve_time.append(None)
            
            df_res["diff_len"] = diff_len
            df_res["diff_v1"] = diff_v1
            df_res["diff_v3"] = diff_v3
            df_res["chamfer"] = chamfer
            df_res['time per file'] = run_time_
            df_res['time per stroke'] = run_time
            df_res['curve simplify/fitting time'] = curve_time
            df_res.to_csv(path_to_csv, index = False)
            return df_res, df_res["diff_len"].count() / df_res["diff_len"].size
        
        vis = True
        df_diff_len = pd.DataFrame() # stroke length different
        df_chamfer = pd.DataFrame() # chamfer distance
        df_time = pd.DataFrame() # running time
        df_comp_rate = pd.DataFrame() # complete rate

        dict_diff_len = {} 
        dict_chamfer = {}
        dict_time = {} 
        dict_comp_rate = {}
        
        list_resolution = ([256]*369+[384]*369+[512]*369+[768]*369+[1024]*369)*4
        list_resolution_ = [256,384,512,768,1024]*4
        def get_res_name(path_to_res):
            res = int(path_to_res.split("/")[-1].split("_")[0])
            name = str(path_to_res.split("/")[-3].split(".")[1])
            return res, name
        def update_dict(res_dict, res_list, method_name, metric_name):
            if metric_name in res_dict:
                res_dict[metric_name] = res_dict[metric_name] + res_list
                res_dict['Methods'] = res_dict['Methods'] + [method_name] * len(res_list)
            else:
                res_dict[metric_name] = res_list
                res_dict['Methods'] = [method_name] * len(res_list)
        def dict_to_df(res_dict, res_df, list_res):
            for name in res_dict:
                res_df[name] = res_dict[name]
            res_df["Resolution"] = list_res
            return res_df
        metric_name = ["Stroke Length Difference", "Chamfer Distance", "Running Time", "Success Rate"]
        for i in range(len(method_name)):
            if path_to_res[i] == None:continue
            df_res, success_rate = eval_and_record(path_to_res[i], path_in[i%5], method_name[i], log_name[i], overwrite = False)
            # visualize all experiment data as box plot
            if vis:
                res, name = get_res_name(path_to_res[i])
                update_dict(dict_diff_len, df_res["diff_len"].to_list(),  method_name_dict[name], metric_name[0])
                update_dict(dict_chamfer, df_res["chamfer"].to_list(), method_name_dict[name], metric_name[1])
                update_dict(dict_time, df_res["time per file"].to_list(), method_name_dict[name], metric_name[2])
                update_dict(dict_comp_rate, [success_rate], method_name_dict[name], metric_name[3])
        
        df_diff_len = dict_to_df(dict_diff_len, df_diff_len, list_resolution)
        df_chamfer = dict_to_df(dict_chamfer, df_chamfer, list_resolution)
        df_time = dict_to_df(dict_time, df_time, list_resolution)
        df_comp_rate = dict_to_df(dict_comp_rate, df_comp_rate, list_resolution_)
        # save re-grouped data into csv file
        df_diff_len.to_csv("metric_len_diff.csv", index = False)
        df_chamfer.to_csv("metric_chamfer.csv", index = False)
        df_time.to_csv("metric_running_time.csv", index = False)
        df_comp_rate.to_csv("metric_complete_rate.csv", index = False)
        # draw box plot
        # boxplot = df_diff_len.boxplot(by=['resolution'], showfliers = False, layout = (1, 4), figsize=(40, 10), rotation=45)
        plt.figure(figsize=(7, 4))
        boxplot = sns.boxplot(data=df_diff_len, x="Resolution", y=metric_name[0], hue="Methods", whis=(0, 100), showfliers=True)
        plt.yscale('log')
        plt.savefig("../../00.paper/sketchvg/figs/fig_comp_stroke_len_diff.pdf", format = 'pdf')
        plt.clf()

        boxplot = sns.boxplot(data=df_chamfer, x="Resolution", y=metric_name[1], hue="Methods", whis=(0, 100) , showfliers=True)
        plt.yscale('log')
        plt.savefig("../../00.paper/sketchvg/figs/fig_comp_chamfer.pdf", format = 'pdf')
        plt.clf()

        boxplot = sns.boxplot(data=df_time, x="Resolution", y=metric_name[2], hue="Methods", whis=(0, 100), showfliers=True)
        plt.yscale('log')
        plt.savefig("../../00.paper/sketchvg/figs/fig_comp_running_time.pdf", format = 'pdf')
        plt.clf()

        # plt.figure(figsize=(6, 4))
        # boxplot = sns.barplot(data=df_comp_rate, x="Resolution", y=metric_name[3], hue="Methods")
        # boxplot.set_ylim(0.969,1.01)
        # plt.savefig("../../00.paper/sketchvg/figs/fig_success_rate.pdf", format = 'pdf')
        # plt.clf()

        
        

    # compute raster metrics
    if False:
        # python3 tools/metric_multiple.py -gt "example/simple-single-dot.png" -i "example/simple-single-dot-horizontal1.png" -d 0 --f-measure --chamfer --hausdorff
        which = set()
        which.add( 'f_score' )
        which.add( 'chamfer' )
        which.add( 'hausdorff' )

        res_chamfer = []
        res_hausdorff = []
        res_f = []
        runtime = []
        refinetime = []

        size_list = ['2x', '4x']
        png_list = df_res['input']
        for s in size_list:
            for png in png_list:
                gt_path = join(path_in,'png'+s, png)
                vec_path = join(path_vec, 'png'+s, png)

                # compute the distance if exists result
                if os.path.exists(vec_path):
                    res = run(vec_path, gt_path, which, distances = [0], visualize = False, force = False)
                    res_chamfer.append(res['chamfer'])
                    res_f.append(res['f_score'][0])
                    res_hausdorff.append(res['hausdorff'])
                else:
                    res_chamfer.append('skipped')
                    res_f.append('skipped')
                    res_hausdorff.append('skipped')

            df_res["chamfer"+ "_" + s] = res_chamfer
            df_res["f"+ "_" + s] = res_f
            df_res["hausdorff"+ "_" + s] = res_hausdorff
            res_chamfer.clear()
            res_f.clear()
            res_hausdorff.clear()

        
        df_res.to_csv(join(path_vec, "%s_metrics_vec.csv"%method_name), index = False)

    # visualize data
    if False:
        # table path
        table_path = '../experiments/08.exp_compairison/00.tables'
        path_sketchvg = '../experiments/08.exp_compairison/01.sketchvg_ver0/512_short'
        path_keypt = '../experiments/08.exp_compairison/02.keypt/512_short'
        path_poly = '../experiments/08.exp_compairison/03.poly/512_short'
        path_virtual = '../experiments/08.exp_compairison/04.virtual/512_short'
        
        ## seperate the experiment result to different files
        df_sketchvg = pd.read_csv(join(path_sketchvg, "sketchvg_metrics.csv"))
        df_keypt = pd.read_csv(join(path_keypt, "keypt_metrics.csv"))
        df_poly = pd.read_csv(join(path_poly, "poly_metrics.csv"))
        df_virtual = pd.read_csv(join(path_virtual, "virtual_metrics.csv"))
        
        df_sketchvg_chamfer = df_sketchvg[["input", "chamfer", "chamfer_2x", "chamfer_4x"]]
        df_sketchvg_chamfer = df_sketchvg_chamfer.rename(columns={'chamfer':'sketchvg_chamfer','chamfer_2x':'sketchvg_chamfer_2x','chamfer_4x':'sketchvg_chamfer_4x'})
        df_keypt_chamfer = df_keypt[["input", "chamfer", "chamfer_2x", "chamfer_4x"]]
        df_keypt_chamfer = df_keypt_chamfer.rename(columns={'chamfer':'keypt_chamfer','chamfer_2x':'keypt_chamfer_2x','chamfer_4x':'keypt_chamfer_4x'})
        df_poly_chamfer = df_poly[["input", "chamfer", "chamfer_2x", "chamfer_4x"]]
        df_poly_chamfer = df_poly_chamfer.rename(columns={'chamfer':'poly_chamfer','chamfer_2x':'poly_chamfer_2x','chamfer_4x':'poly_chamfer_4x'})
        df_virtual_chamfer = df_virtual[["input", "chamfer", "chamfer_2x", "chamfer_4x"]]
        df_virtual_chamfer = df_virtual_chamfer.rename(columns={'chamfer':'virtual_chamfer','chamfer_2x':'virtual_chamfer_2x','chamfer_4x':'virtual_chamfer_4x'})

        df_sketchvg_f = df_sketchvg[["input", "f", "f_2x", "f_4x"]]
        df_keypt_f = df_keypt[["input", "f", "f_2x", "f_4x"]]
        df_poly_f = df_poly[["input", "f", "f_2x", "f_4x"]]
        df_virtual_f = df_virtual[["input", "f", "f_2x", "f_4x"]]

        df_sketchvg_hausdorff = df_sketchvg[["input", "hausdorff", "hausdorff_2x", "hausdorff_4x"]]
        df_keypt_hausdorff = df_keypt[["input", "hausdorff", "hausdorff_2x", "hausdorff_4x"]]
        df_poly_hausdorff = df_poly[["input", "hausdorff", "hausdorff_2x", "hausdorff_4x"]]
        df_virtual_hausdorff = df_virtual[["input", "hausdorff", "hausdorff_2x", "hausdorff_4x"]]

        # visualize metrics
        chamfer_list = [df_sketchvg_chamfer, df_keypt_chamfer, df_poly_chamfer, df_virtual_chamfer]
        df_metrics_chamfer = reduce(lambda left, right: pd.merge(left, right, on=['input']), chamfer_list)
        df_metrics_chamfer = df_metrics_chamfer.drop(columns = ['input'])

        f_list = [df_sketchvg_f, df_keypt_f, df_poly_f, df_virtual_f]
        df_metrics_f = reduce(lambda left, right: pd.merge(left, right, on=['input']), f_list)
        df_metrics_f = df_metrics_f.drop(columns = ['input'])

        hausdorff_list = [df_sketchvg_hausdorff, df_keypt_hausdorff, df_poly_hausdorff, df_virtual_hausdorff]
        df_metrics_hausdorff = reduce(lambda left, right: pd.merge(left, right, on=['input']), hausdorff_list)
        df_metrics_hausdorff = df_metrics_hausdorff.drop(columns = ['input'])

        # df_metrics_runtime = df[[ "sketchvg_runtime", "keypt_runtime", "poly_runtime", "virtual_runtime"]]
        
        df_metrics_chamfer = df_metrics_chamfer.apply(pd.to_numeric, errors = 'coerce')
        df_metrics_chamfer_avg = df_metrics_chamfer.mean(axis = 0)
        ax = df_metrics_chamfer_avg.plot(x = "Methods", y = "Chamfer Distance", kind = "bar")
        custom_labels = ["Our","Our2x","Our4x","Puha","Puha2x","Puha4x","Bess","Bess2x","Bess4x","Mo","Mo2x","Mo4x"]
        ax.set_xticklabels(custom_labels)
        ax.tick_params(axis='x', rotation=45)
        # ax.legend(['Methods'])
        plt.savefig(join(table_path, 'res_chamfer.pdf'), format = 'pdf')
        plt.close()

        # df_metrics_f = df_metrics_f.apply(pd.to_numeric, errors = 'coerce')
        # df_metrics_f_avg = df_metrics_f.mean(axis = 0)
        # ax = df_metrics_f_avg.plot(x = "Methods", y = "F Score", kind = "bar")
        # custom_labels = ["Our", "Puhachov", "Bessmeltsev", "Mo"] 
        # ax.set_xticklabels(custom_labels)
        # ax.tick_params(axis='x', rotation=0)
        # # ax.legend(['Methods'])
        # plt.savefig('res_f.pdf', format = 'pdf')
        # plt.close()

        # df_metrics_hausdorff = df_metrics_hausdorff.apply(pd.to_numeric, errors = 'coerce')
        # df_metrics_hausdorff_avg = df_metrics_hausdorff.mean(axis = 0)
        # ax = df_metrics_hausdorff_avg.plot(x = "Methods", y = "Hausdorff Distance", kind = "bar")
        # custom_labels = ["Our", "Puhachov", "Bessmeltsev", "Mo"] 
        # ax.set_xticklabels(custom_labels)
        # ax.tick_params(axis='x', rotation=0)
        # # ax.legend(['Methods'])
        # plt.savefig('res_hausdorff.pdf', format = 'pdf')
        # plt.close()

        # df_metrics_runtime = df_metrics_runtime.apply(pd.to_numeric, errors = 'coerce')
        # df_metrics_runtime_avg = df_metrics_runtime.mean(axis = 0)
        # ax = df_metrics_runtime_avg.plot(x = "Methods", y = "Running Time", kind = "bar")
        # custom_labels = ["Our", "Puhachov", "Bessmeltsev", "Mo"] 
        # ax.set_xticklabels(custom_labels)
        # ax.tick_params(axis='x', rotation=0)
        # # ax.legend(['Methods'])
        # plt.savefig('res_runtime.pdf', format = 'pdf')
        # plt.close()

    # visualize data of NDC network comparison
    if False:
        path_in = '../data/benchmark/gt'
        path_ndc = '../experiments/08.exp_compairison/05.sketchvg_ndc'
        log_ndc = 'ndc_exp_log_noisy.txt'
        res_ndc_exp = []

        # initial empty dataframe
        df = pd.DataFrame()
        npz_list = os.listdir(path_in)
        df['input image'] = npz_list

        # log running information
        with open(join(path_ndc, log_ndc), 'r') as f:
            lines_sketchvg = f.readlines()
       
        x_acc_2d_list = []
        x_recall_2d_list = []
        y_acc_2d_list = []
        y_recall_2d_list = []
        a_acc_2d_list = []
        a_recall_2d_list = []
        valence_2d_list = []
        
        x_acc_3d_list = []
        x_recall_3d_list = []
        y_acc_3d_list = []
        y_recall_3d_list = []
        a_acc_3d_list = []
        a_recall_3d_list = []
        valence_3d_list = []

        # acc_x_pos, acc_x_neg, acc_y_pos, acc_y_neg, acc_a_pos, acc_a_neg
        for l in lines_sketchvg:
            l_ = l.strip('\n').split(" ")
            fname = l_[0]
            valence_3d = float(l_[-1])
            x_pos_3d = float(l_[-7])
            x_neg_3d = float(l_[-6])
            y_pos_3d = float(l_[-5])
            y_neg_3d = float(l_[-4])
            a_pos_3d = float(l_[-3])
            a_neg_3d = float(l_[-2])
            x_acc_3d = (x_pos_3d + x_neg_3d) / 2
            x_recall_3d = x_pos_3d
            y_acc_3d = (y_pos_3d + y_neg_3d) / 2
            y_recall_3d = y_pos_3d
            a_acc_3d = (a_pos_3d + a_neg_3d) / 2
            a_recall_3d = a_pos_3d
            x_acc_3d_list.append(x_acc_3d)
            x_recall_3d_list.append(x_recall_3d)
            y_acc_3d_list.append(y_acc_3d)
            y_recall_3d_list.append(y_recall_3d)
            a_acc_3d_list.append(a_acc_3d)
            a_recall_3d_list.append(a_recall_3d)
            valence_3d_list.append(valence_3d)

            

            valence_2d = float(l_[8])
            x_pos_2d = float(l_[2])
            x_neg_2d = float(l_[3])
            y_pos_2d = float(l_[4])
            y_neg_2d = float(l_[5])
            a_pos_2d = float(l_[6])
            a_neg_2d = float(l_[7])
            x_acc_2d = (x_pos_2d + x_neg_2d) / 2
            x_recall_2d = x_pos_2d
            y_acc_2d = (y_pos_2d + y_neg_2d) / 2
            y_recall_2d = y_pos_2d
            a_acc_2d = (a_pos_2d + a_neg_2d) / 2
            a_recall_2d = a_pos_2d
            
            x_acc_2d_list.append(x_acc_2d)
            x_recall_2d_list.append(x_recall_2d)
            y_acc_2d_list.append(y_acc_2d)
            y_recall_2d_list.append(y_recall_2d)
            a_acc_2d_list.append(a_acc_2d)
            a_recall_2d_list.append(a_recall_2d)
            valence_2d_list.append(valence_2d)

        df["X_acc_2d"] = x_acc_2d_list
        df["X_recall_2d"] = x_recall_2d_list
        df["Y_acc_2d"] = y_acc_2d_list
        df["Y_recall_2d"] = y_recall_2d_list
        df["A_acc_2d"] = a_acc_2d_list
        df["A_recall_2d"] = a_recall_2d_list
        df["X_acc_3d"] = x_acc_3d_list
        df["X_recall_3d"] = x_recall_3d_list
        df["Y_acc_3d"] = y_acc_3d_list
        df["Y_recall_3d"] = y_recall_3d_list
        df["A_acc_3d"] = a_acc_3d_list
        df["A_recall_3d"] = a_recall_3d_list

        df["valence_2d"] = valence_2d_list
        df["valence_3d"] = valence_3d_list

        df.to_csv(join('../experiments/08.exp_compairison', "exp_ndc_comparison.csv"), index = False)


