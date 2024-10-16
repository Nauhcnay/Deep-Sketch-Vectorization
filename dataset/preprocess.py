'''

'''
import json
import os
import numpy as np
import subprocess
import random
import shutil
import cv2
import matplotlib.pyplot as plt
import threading
import time
import ndjson

from PIL import Image
from svgpathtools import *
from tqdm import tqdm
from skimage.feature import peak_local_max
from skimage.morphology import skeletonize
from skimage.util import invert
from scipy.signal import convolve2d
from random import sample, shuffle
from shutil import which
# set up a timeout for each subprocess
# https://stackoverflow.com/questions/4158502/kill-or-terminate-subprocess-when-timeout
# but why?
# and this seems better
# https://alexandra-zaharia.github.io/posts/kill-subprocess-and-its-children-on-timeout-python/
class RunCmd(threading.Thread):
    def __init__(self, cmd, timeout = 10):
        threading.Thread.__init__(self)
        self.cmd = cmd
        self.timeout = timeout

    def run(self):
        self.p = subprocess.Popen(self.cmd)
        self.p.wait()

    def Run(self):
        self.start()
        self.join(self.timeout)

        if self.is_alive():
            self.p.terminate()      #use self.p.kill() if process needs a kill -9
            self.join()

def svg_to_numpy(svg, dpi = 600, retry = 3):
    if retry < 0:
        print("error:\tcan't rasterize %s, \n pls check if the svg file is correct"%svg)
        return None
    png = svg.replace(".svg", "_temp.png")
    rasterize(svg, png, dpi)
    try:
        vec_png = cv2.imread(png, cv2.IMREAD_GRAYSCALE)
        os.remove(png)
        return vec_png
    except:
        svg_to_numpy(svg, dpi = 600, retry = retry - 1)
    

def rasterize(svg, png, dpi = 400):
    '''
    export svg to png by inkscape, the inkscape should be added into environment path before calling this function
    Given:
        svg: path to the svg file
        png: path to the export png file
        dpi: export png resolution, larger dpi will output larger resolution
    Action:
        export svg to png
    '''
    if which("inkscape") is not None:
        RunCmd(["inkscape", svg, 
            "--export-filename=" + png,  
            "--export-background=white", "--export-dpi=%d"%dpi, "--export-area-snap"]).Run()
    else:
        if which("magick") is None:
            raise ValueError("Can't find rasterize app on your system!")
        RunCmd(["magick", "convert", "-density"," %d"%dpi, svg, png]).Run()


def dict_to_nplist(pts, nplist):
    if len(pts) == 0: return None # skip if no strokes are found
    for stroke in pts:
        assert len(stroke) >= 2
        np_path = []
        for i in range(1, len(stroke)):
            start = np.array(stroke[i-1])
            end = np.array(stroke[i])
            np_path.append(Line(start, end))
        nplist.append(np.array(np_path))

def dict_to_paths(pts, paths):
    if len(pts) == 0: return None # skip if no strokes are found
    for stroke in pts:
        assert len(stroke) >= 2
        svg_path = Path()
        for i in range(1, len(stroke)):
            start = complex(*stroke[i-1])
            end = complex(*stroke[i])
            svg_path.append(Line(start, end))
        paths.append(svg_path)

def nplist_to_svg(nplist):
    paths = []
    # tiny_line_con = 0
    zigzag_lines_con = 0
    p_counter = 0
    for p in nplist:
        svg_path = Path()
        last_tan = None
        for l in p:
            tiny_line_con = 0
            zigzag_lines_con = 0
            start = complex(*(l[0].tolist()))
            end = complex(*(l[1].tolist()))
            line = Line(start, end)
            # test if this polyline is tiny line
            llen = line.length()
            if llen < 0.5:
                continue
            svg_path.append(line)
            p_counter += 1
            # elif llen <= 3:
            #     tiny_line_con += 1
            # test if current line has sharp turn 
            curr_tan = line.unit_tangent()
            if last_tan is not None and (last_tan.real * curr_tan.real + last_tan.imag * curr_tan.imag) < -0.5:
                zigzag_lines_con += 1
            last_tan = curr_tan
        # if tiny_line_con > 5 or zigzag_lines_con > 5:
        if zigzag_lines_con >= 5:
            continue
        else:
            paths.append(svg_path)
    
    if len(paths) == 0:
        print("log:\tfind unusable sketch, skipping")
        p_counter = float('int')

    return paths, p_counter

def find_bounding_box(nplist):
    x_min, y_min = float('inf'), float('inf')
    x_max, y_max = 0, 0
    for p in nplist:
        x = p[:, :, 0] # get all x
        y = p[:, :, 1] # get all y
        if x_min > x.min(): x_min = x.min() # update the minimu coord
        if y_min > y.min(): y_min = y.min()
        if x_max < x.max(): x_max = x.max()
        if y_max < y.max(): y_max = y.max()
    return x_min, y_min, x_max, y_max

def trans_nplist(nplist, x, y):
    for p in nplist:
        x_np = p[:, :, 0] # get all x
        y_np = p[:, :, 1] # get all y
        if x < 0: assert x_np.min() > abs(x)
        if y < 0: assert y_np.min() > abs(y)
        p[:, :, 0] = x_np.astype(float) + x
        p[:, :, 1] = y_np.astype(float) + y

def resize_nplist(nplist, ratio):
    assert ratio > 0
    for i in range(len(nplist)):
        nplist[i] = nplist[i].astype(float) * ratio

def creative_sketch_to_svg(json_input, dir_output, svg_counter, svg_margin = 20):
    '''
    convert json file in creative sketch dataset to svg file, 
    each file will be translated and resized so that its
    shorter size will be 288 and its width and height ratio will not greater than
    1.3. And it will have 20 pix width margin (value set by svg_margin) at boundary
    '''
    # skip existing files
    if os.path.exists(os.path.join(dir_output, "%07d.svg"%svg_counter)):
        svg_counter += 1
        return svg_counter
    with open(json_input, "r") as f:
        data = json.load(f)
    
    # parse the json file to numpy list
    svg_paths = []
    nplist = []
    for key1 in data:
        if type(data[key1]) == dict:
            for key2 in data[key1]:
                dict_to_nplist(data[key1][key2], nplist)
        else:
            dict_to_nplist(data[key1], nplist)
    # adjust strokes in numpy list and write them into svg
    return adjust_and_to_svg(nplist, dir_output, svg_margin, svg_counter, edge_size = 300, 
        path_thres_low = 400, path_thres_high = 1000, wh_ratio_thres = 1.3)

def adjust_and_to_svg(nplist, dir_output, svg_margin, svg_counter, path_thres_low = 400, path_thres_high = 1000, grid_size = 8, edge_size = 288, wh_ratio_thres = 1.3, padding = False):
    # find out top left corner
    x_min, y_min, x_max, y_max = find_bounding_box(nplist)
    w, h = x_max - x_min, y_max - y_min
    if w == 0 or h == 0: return svg_counter # we probably get a empty svg
    # off set the path coord, make the shorter size of the image always equal to 288
    x_offset, y_offset = svg_margin - x_min, svg_margin - y_min
    x_max += x_offset
    y_max += y_offset
    trans_nplist(nplist, x_offset, y_offset)
    # filter out the current svg if its width/height ratio is too large    
    wh_ratio = w / h if w > h else h / w
    if wh_ratio > wh_ratio_thres: return svg_counter # skip if the shape of sketch is too wide or too long    

    if padding is False:    
        resize_ratio = (edge_size - svg_margin) / x_max if x_max < y_max else (edge_size - svg_margin) / y_max
        resize_nplist(nplist, resize_ratio)
        # adjust width and height to dual contouring grid size
        x_max = int(x_max * resize_ratio + svg_margin)
        y_max = int(y_max * resize_ratio + svg_margin)
        dc_offset_x = grid_size - x_max % grid_size
        dc_offset_y = grid_size - y_max % grid_size
        x_max += dc_offset_x
        y_max += dc_offset_y
        assert x_max % grid_size == 0 and y_max % grid_size == 0
    else:
        resize_ratio = (edge_size - svg_margin) / x_max if x_max > y_max else (edge_size - svg_margin) / y_max
        resize_nplist(nplist, resize_ratio)
        x_max = int(x_max * resize_ratio) + svg_margin
        y_max = int(y_max * resize_ratio) + svg_margin
        x_offset = (edge_size - x_max) / 2
        y_offset = (edge_size - y_max) / 2
        trans_nplist(nplist, x_offset, y_offset)
        x_max, y_max = edge_size, edge_size
    # save to svg files    
    # thanks to https://github.com/ivanpuhachov/line-drawing-vectorization-polyvector-flow-dataset/blob/main/utils/process_ndjson_quickdraw.py
    # magick_strokewidth = 0.2537178821563721
    magick_strokewidth = 1
    svg_paths, path_nums = nplist_to_svg(nplist)
    if path_nums < path_thres_low: return svg_counter # skip current svg if it has too less strokes
    if path_nums > path_thres_high: return svg_counter # skip current svg if it has too much strokes (probably messy sketch)
    out_path = os.path.join(dir_output, "%07d.svg"%svg_counter)
    wsvg(svg_paths, filename=out_path,
        stroke_widths=[magick_strokewidth for stroke in svg_paths],
      dimensions=("%dpx"%x_max, "%dpx"%y_max),
      svg_attributes={'xmlns': 'http://www.w3.org/2000/svg', 'xmlns:ev': 'http://www.w3.org/2001/xml-events', 'xmlns:xlink': 'http://www.w3.org/1999/xlink', 'baseProfile': 'full', 'width': '%dpx'%x_max, 'height': '%dpx'%y_max, 'version': '1.1', 'viewBox': '0 0 %d %d'%(x_max, y_max)}
    )
    # for debug
    # rasterize(out_path, out_path.replace(".svg", ".png"))
    svg_counter += 1
    return svg_counter

def prepare_creative_sketch(dir_input, dir_output, svg_counter):
    '''
    convert all JSON files of createive sketch dataset:
    https://songweige.github.io/projects/creative_sketech_generation/gallery_creatures.html
    into svg files, also rename each file with a 7 digit number
    '''
    skip_file_list = ["id_to_class.json"]
    for root, dirs, files in os.walk(dir_input):
        for folder in tqdm(dirs):
            for sub_root, sub_dirs, sketches_json in os.walk(os.path.join(root, folder)):
                for sketch_json in sketches_json:
                    # print("opening %s"%os.path.join(sub_root, sketch_json))
                    if ".json" not in sketch_json: continue
                    svg_counter = creative_sketch_to_svg(os.path.join(sub_root, sketch_json), dir_output, svg_counter)
    return svg_counter

def prepare_quick_draw(dir_input, dir_output, svg_counter, export_ratio):
    for json in tqdm(os.listdir(dir_input)):
        # convert each drawing to numpy list
        _, ext = os.path.splitext(os.path.join(dir_input, json))
        if ext != ".ndjson": continue # we only read ndjson file
        if "tornado" in json: continue # skip tornado class since it really not fit for vectorization
        with open(os.path.join(dir_input, json), "r") as f:
            data = ndjson.load(f)
        counter = 0
        target_num = int(len(data) * export_ratio)
        target_num = target_num if target_num > 20 else 20
        # shuffle(data) # make sure we have chance to get the svg from any place
        for sketch in data:
            if counter > target_num: break
            if os.path.exists(os.path.join(dir_output, "%07d.svg"%svg_counter)):
                svg_counter += 1
                continue
            nplist = []
            skip = False
            for i in range(len(sketch["drawing"])):
                stroke = sketch["drawing"][i]
                try:
                    # in case we get none drawing data
                    pts = np.array(stroke).squeeze().T[:,0:2]
                except:
                    continue
                segs = []
                for i in range(len(pts) - 1):
                    segs.append([pts[i], pts[i + 1]])
                nplist.append(np.array(segs))
            # adjust and write to svg
            svg_counter_new = adjust_and_to_svg(nplist, dir_output, 20, svg_counter, edge_size = 300,
                path_thres_low = 15, path_thres_high = 100, padding = False)
            if svg_counter != svg_counter_new:
                counter += 1
            svg_counter = svg_counter_new

def svg_augmentation(svg_input):
    # generate several svg sketches with random per stroke width
    pass

# thanks to https://stackoverflow.com/questions/29731726/how-to-calculate-a-gaussian-kernel-matrix-efficiently-in-numpy
def gkern(l=None, sig=2.):
    """
    creates gaussian kernel with side length `l` and a sigma of `sig`
    """
    if l is None:
        l = 2 * sig * 6 + 1
    ax = np.linspace(-(l - 1) // 2., (l - 1) // 2., l)
    gauss = np.exp(-0.5 * np.square(ax) / np.square(sig))
    kernel = np.outer(gauss, gauss)
    return kernel

def skeleton_to_udf(img, name, png_output, size, wh = None):
    '''
        wh means the width and height of the image. Yes, we can get w and h from img but 
        it turns out that resize the img directly sometime will have 1 pixel difference
        so we have to get the width and height from somewhere else
    '''
    if wh is None:
        img_temp = img_resize(img, False, name, shorter_size = size) # if we want to ignore this difference, uncomment this line and comment the next line
    else:
        img_temp = cv2.resize(img, wh, interpolation = cv2.INTER_AREA)
    img_temp = to_gray_threshold_reverse(img_temp, True)
    img_temp = skeletonize(img_temp, method='lee').astype(np.uint8)
    if img_temp.max() == 1:
        img_temp *= 255 
    img_temp = 255 - cv2.distanceTransform(255 - img_temp, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    cv2.imwrite(png_output.replace("_h5.png", "_h5_%d.png"%size), img_temp)
    return img_temp

def create_training_set(svg_path, trainset_folder, samlpe_num = 3000, regenerate_gt_only = False, distance_field = False, hm_type = "guassian"):
    '''
    Given:
        svg_path, path to svg files, 
        sample_num, how many sketches will be sampled, 0 means don't sample
        Todo:
        1. add data augmentation, generate rough sketches from the cleaned version
    Action:
        sample svg to "dataset_train" folder
    '''
    sketch_list = os.listdir(os.path.join(svg_path, "sketch"))
    pt_path = os.path.join(svg_path, "keypoints")

    # this folder saves sampled svg images, the png training image need to be generated by another script
    sample_sketch_path = os.path.join(trainset_folder, "sketch")
    # this folder saves png gt images 
    trainset_folder, _ = os.path.split(trainset_folder)
    sample_gt_path = os.path.join(trainset_folder, "gt")
    sample_train00_path = os.path.join(trainset_folder, "train", "00")
    
    # start sample if the sample number is greater than 0
    if sample_num > 0:
        sketch_list = random.sample(sketch_list, sample_num)

    # this targets to regenerate all gt png images
    if regenerate_gt_only:
        sketch_list = os.listdir(sample_sketch_path)

    # 1st pass, copy all sampled svgs, and generate rasterized sketch with basic brush type
    if os.path.exists(sample_train00_path) == False:
        os.makedirs(sample_train00_path)
    for sketch in tqdm(sketch_list):
        name, ext = os.path.splitext(sketch)
        assert "svg" in ext
        sketch_path = os.path.join(svg_path, "sketch", sketch)
        # skip the svg if it has been copied already
        if os.path.exists(os.path.join(sample_sketch_path, sketch)) == False:
            shutil.copy(sketch_path, sample_sketch_path)
        png_output = os.path.join(sample_train00_path, sketch.replace(".svg", ".png"))
        if os.path.exists(png_output) == False:
            # the dpi in inkscape and illustrator seems different, I have to use dpi 400 in inkscape to get the same
            # output resolution as the dpi 150 (options.resolution) in illustrator
            RunCmd(["inkscape", sketch_path, 
            "--export-filename=" + png_output,  
            "--export-background=white", "--export-dpi=400"]).Run()

    # 2nd pass, generate GT of each sketch, the best way is to generate the heat map from point
    for sketch in tqdm(sketch_list):
        name, ext = os.path.splitext(sketch)
        sketch_path = os.path.join(svg_path, "sketch", sketch)
        
        # generate the gt from svg if it exists
        pt_endpoint = name + "_1" + ".svg" # endpoint is h2
        test_path = os.path.join(sample_gt_path, pt_endpoint.replace("_1.svg", "_h1_288.png"))
        
        if os.path.exists(test_path): continue
        # print("Log:\t convert %s\t%.2f%%"%(sketch_path, i/len(sketch_list)*100))
        coords_h2, gt_size = svg_to_png(os.path.join(pt_path, pt_endpoint), 
            os.path.join(sample_gt_path, pt_endpoint.replace("_1.svg", "_h2.png")), True, sketch_path, hm_type=hm_type)        
        pt_sharpturn = name + "_2" + ".svg" # sharp turn is h4
        coords_h4, _ = svg_to_png(os.path.join(pt_path, pt_sharpturn), 
            os.path.join(sample_gt_path, pt_sharpturn.replace("_2.svg", "_h4.png")), True, sketch_path, hm_type=hm_type)
        assert gt_size == _
        pt_junction = name + "_3" + ".svg" # junction is h3
        coords_h3, _ = svg_to_png(os.path.join(pt_path, pt_junction), 
            os.path.join(sample_gt_path, pt_junction.replace("_3.svg", "_h3.png")), True, sketch_path, hm_type=hm_type)
        assert gt_size == _

        # let's add h5!
        # rasterize the sketch image
        # generate the target size directly seems better
        # if gt_size[0] > gt_size[1]: # if height longer than width
        #     RunCmd(["inkscape", sketch_path, 
        #         "--export-filename=" + png_output,  
        #         "--export-background=white", "--export-width=288"]).Run()
        # else:
        #     RunCmd(["inkscape", sketch_path, 
        #         "--export-filename=" + png_output,  
        #         "--export-background=white", "--export-height=288"]).Run()
        if "udf" in hm_type:
            png_output = os.path.join(sample_gt_path, name + "_h5.png")
            RunCmd(["inkscape", sketch_path, 
                "--export-filename=" + png_output,  
                "--export-background=white", "--export-dpi=400"]).Run()
            gt_h5_large = cv2.imread(png_output)
            gt_h5_288 = skeleton_to_udf(gt_h5_large, name, png_output, 288)
            if hm_type == "udf_v2":
                gt_h5_128 = img_resize(gt_h5_288, True, shorter_size = 128)
                cv2.imwrite(png_output.replace("_h5.png", "_h5_128.png"), gt_h5_128)
                gt_h5_96 = img_resize(gt_h5_288, True, shorter_size = 96)
                cv2.imwrite(png_output.replace("_h5.png", "_h5_96.png"), gt_h5_96)
            elif hm_type == "udf_v1":
                skeleton_to_udf(gt_h5_288, name, png_output.replace("_h5.png", "_h5_128.png"), 128)
                skeleton_to_udf(gt_h5_288, name, png_output.replace("_h5.png", "_h5_96.png"), 96)
            os.remove(png_output)

        # create h1
        # h1 288
        coord_list = [coords_h2, coords_h3, coords_h4]
        for i in range(len(coord_list) - 1, -1, -1):
            if len(coord_list[i]) == 0: coord_list.pop(i)
        if len(coord_list) == 0:
            print("Waning:\t%s doesn't have any key points, please double check if this is correct!")
            coords_h1 = np.array([])
        else:
            coords_h1 = np.concatenate(coord_list)
        img_h1_288, _ = img_resize_coord(gt_size, coords_h1, 288, hm_type = hm_type)
        
        cv2.imwrite(os.path.join(sample_gt_path, pt_endpoint.replace("_1.svg", "_h1_288.png")), img_h1_288)
        # h1 128
        if hm_type == "udf_v2":
            img_temp = img_resize(img_h1_288, True, name, shorter_size = 128)
        else:
            img_temp, _ = img_resize_coord(gt_size, coords_h1, 128, hm_type = hm_type, ksize = 7)
        cv2.imwrite(os.path.join(sample_gt_path, pt_endpoint.replace("_1.svg", "_h1_128.png")), img_temp)
        if hm_type == "udf_v2":
            img_temp = img_resize(img_h1_288, True, name, shorter_size = 96)
        else:
            # will small size work?
            img_temp, _ = img_resize_coord(gt_size, coords_h1, 96, hm_type = hm_type, ksize = 5)
        cv2.imwrite(os.path.join(sample_gt_path, pt_endpoint.replace("_1.svg", "_h1_96.png")), img_temp)
        check_sketch_size(sample_gt_path, name)

def check_sketch_size(gt_path, name):
    h1_96 = os.path.join(gt_path, name + "_h1_96.png")
    h1_128 = os.path.join(gt_path, name + "_h1_128.png")
    h1_288 = os.path.join(gt_path, name + "_h1_288.png")

    h2_96 = os.path.join(gt_path, name + "_h2_96.png")
    h2_128 = os.path.join(gt_path, name + "_h2_128.png")
    h2_288 = os.path.join(gt_path, name + "_h2_288.png")

    h3_96 = os.path.join(gt_path, name + "_h3_96.png")
    h3_128 = os.path.join(gt_path, name + "_h3_128.png")
    h3_288 = os.path.join(gt_path, name + "_h3_288.png")

    h4_96 = os.path.join(gt_path, name + "_h4_96.png")
    h4_128 = os.path.join(gt_path, name + "_h4_128.png")
    h4_288 = os.path.join(gt_path, name + "_h4_288.png")

    h5_96 = os.path.join(gt_path, name + "_h5_96.png")
    h5_128 = os.path.join(gt_path, name + "_h5_128.png")
    h5_288 = os.path.join(gt_path, name + "_h5_288.png")

    if os.path.exists(h5_96):
        img_96 = [h1_96, h2_96, h3_96, h4_96, h5_96]
        img_128 = [h1_128, h2_128, h3_128, h4_128, h5_128]
        img_288 = [h1_288, h2_288, h3_288, h4_288, h5_288]
    else:
        img_96 = [h1_96, h2_96, h3_96, h4_96]
        img_128 = [h1_128, h2_128, h3_128, h4_128]
        img_288 = [h1_288, h2_288, h3_288, h4_288]

    h1_96_size = cv2.imread(h1_96).shape
    h1_128_size = cv2.imread(h1_128).shape
    h1_288_size = cv2.imread(h1_288).shape
    
    for i in range(1, len(img_96)):
        other_96_size = cv2.imread(img_96[i]).shape
        other_128_size = cv2.imread(img_128[i]).shape
        other_288_size = cv2.imread(img_288[i]).shape
        assert h1_96_size == other_96_size
        assert h1_128_size == other_128_size
        assert h1_288_size == other_288_size

def cleanup_coords(coords):
    # let's cleanup the coords
    # print("Log:\tstart cleanup coords... ", end = "")
    if len(coords) == 0:
        # print("Done.")
        return coords
    coords_h = np.expand_dims(coords, 0) # horizontal 
    coords_v = np.expand_dims(coords, 1) # vertical
    dis_matrix = np.sqrt(np.square(coords_v - coords_h).sum(axis = -1)) # get coord distance matrix
    idx_too_close = np.array(np.where(np.logical_and(0 < dis_matrix, dis_matrix < 1.5))).T # find coords that are too close to each other
    groups = []
    skip_id = np.unique(idx_too_close.flatten())
    # generate merge group
    for c in idx_too_close:
        if len(groups) == 0: 
            groups.append({c[0]:None, c[1]:None})
            continue
        new_group = True
        for g in groups:
            if c[0] in g or c[1] in g:
                g[c[0]] = None
                g[c[1]] = None
                new_group = False
        if new_group:
            groups.append({c[0]:None, c[1]:None})
    # merge to new coords
    coords_merged = []
    for i, c in enumerate(coords):
        if i in skip_id: continue
        coords_merged.append(coords[i].tolist())
    for g in groups:
        new_coord = coords[list(g.keys())].mean(axis = 0).astype(int)
        coords_merged.append(new_coord.tolist())
    # print("Done.")
    return np.unique(np.array(coords_merged), axis = 0)

def to_gray_threshold_reverse(img, skeleton = False):
    img = img[:,:,0:3].mean(axis = -1).astype(np.uint8)
    _, img = cv2.threshold(img,240,255,cv2.THRESH_BINARY)
    # img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
    img = (255 - img).astype(np.uint8) # RGB to gray, reverse, the point will have 255 value
    if skeleton:
        return img.astype(bool)
    else:
        return img

def svg_to_png(svg_input, png_output, as_gt, sketch_input = None, dpi = 400, hm_type = "guassian"):
    '''
    hm_type, "guassian" as guassian blur or 
            "udf_v1" as unsinged distance field and regenerate a new distance field for each resolution (288, 128, 96) or
            "udf_v2" as unsinged distance field but only downscale the image for each resolution
    '''
    # 1. convert to png with higher resolution
    if os.path.exists(svg_input):
        RunCmd(["inkscape", svg_input, 
            "--export-filename=" + png_output,  
            "--export-background=white", "--export-dpi=%d"%dpi]).Run()
        img = cv2.imread(png_output)
    else:
        assert sketch_input is not None
        assert os.path.exists(sketch_input)
        # if the key point svg does not exist, convert the sketch svg to get the image size
        RunCmd(["inkscape", sketch_input, 
            "--export-filename=" + png_output,  
            "--export-background=white", "--export-dpi=%d"%dpi]).Run()
        img = cv2.imread(png_output)
        img = np.ones(img.shape) * 255

    # time.sleep(3) # hope this could solve the freeze problem
    # print("Log:\tclean up keypoints")
    # 2. RGB to gray, reverse, resize and blur
    if as_gt:
        img = to_gray_threshold_reverse(img)        
    # 3. extract point coordinations
    _, name = os.path.split(png_output)
    coords = np.array(np.where(img == 255)).T
    coords_merged = cleanup_coords(coords) # shrink points that are adjacent to one point
    coords_merged[:,[0, 1]] = coords_merged[:, [1, 0]] # convert h,w coordination to
    # then we should save this information to the file
    h, w = img.shape[0], img.shape[1]
    with open(png_output.replace(".png", ".json"), "w") as f:
        coord_gt = {"height":h, "width":w, "coords":coords_merged.tolist()}
        json.dump(coord_gt, f)
    # and we probably don't need to genrate the heatmap right now...
    img_temp, coords_288 = img_resize_coord((h, w), coords_merged, 288, hm_type = hm_type)
    h_288, w_288 = img_temp.shape[0], img_temp.shape[1]
    cv2.imwrite(png_output.replace(".png", "_288.png"), img_temp)
    if as_gt and hm_type == "udf_v2":
        img_128 = img_resize(img_temp, as_gt, name, shorter_size = 128)
        cv2.imwrite(png_output.replace(".png", "_128.png"), img_128)
    else:
        img_temp, _ = img_resize_coord((h, w), coords_merged, 128, hm_type = hm_type, ksize = 7)
        cv2.imwrite(png_output.replace(".png", "_128.png"), img_temp)
    if as_gt and hm_type == "udf_v2":
        img_96 = img_resize(img_temp, as_gt, name, shorter_size = 96)
        cv2.imwrite(png_output.replace(".png", "_96.png"), img_96)
    else:
        img_temp, _ = img_resize_coord((h, w), coords_merged, 96, hm_type = hm_type, ksize = 5)
        cv2.imwrite(png_output.replace(".png", "_96.png"), img_temp)

    # 4. remove the high resolution png, we won't use that during the training
    os.remove(png_output)
    return coords_merged, (h, w)

def get_wh(hw, shorter_size, name):
    if type(hw) == tuple:
        h, w = hw
    else:
        h, w = hw.shape[0], hw.shape[1]
    if h < 288 or w < 288:
        print("Warning:\tinput image %s seems too small to resize, please double check if your input is correct."%name)
    if h > w:
        ratio = w / shorter_size
        h = int(h / ratio + 0.5)
        w = shorter_size
    else:
        ratio = h / shorter_size
        w = int(w / ratio + 0.5)
        h = shorter_size
    return w, h

def img_resize(img, as_gt, name = "undefined", shorter_size = 288):
    w, h = get_wh(img, shorter_size, name)
    if as_gt:
        return cv2.resize(img, (w, h), interpolation = cv2.INTER_NEAREST)
    else:
        return cv2.resize(img, (w, h), interpolation = cv2.INTER_AREA)

def img_resize_coord(input_size, coords, shorter_size = 288, sigma = 2, hm_type = "guassian", ksize = None):
    # I need to write a different resize function
    h, w = input_size
    ratio = 1
    if h > w:
        ratio = w / shorter_size
        h = int(h / ratio + 0.5)
        w = shorter_size
    else:
        ratio = h / shorter_size
        w = int(w / ratio  + 0.5)
        h = shorter_size

    if len(coords) == 0:
        coords_resized = coords
    else:
        coords_resized = (coords / ratio + 0.5).astype(int)
    if hm_type == "guassian":
        kernel = gkern(l = None, sig = sigma)
        img = coord_to_hm((h, w), coords_resized, kernel)
    elif "udf" in hm_type:
        img = coord_to_udf((h, w), coords_resized)
    return img.astype(np.uint8), coords_resized

def coord_to_img(size, coords, reverse = False):
    '''
    if reverse is True, keypiont is 0 and background is 255
    '''
    img = np.zeros(size, dtype = np.uint8)
    if reverse:
        img = img + 255
    if len(coords) == 0: return img
    c = (coords[:, 1], coords[:, 0]) # heigt (y), width (x)
    if reverse:
        img[c] = 0
    else:
        img[c] = 255
    return img

def coord_to_udf(size, coords):
    img = coord_to_img(size, coords, True)
    img = cv2.distanceTransform(img, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    return 255 - img

def coord_to_hm(size, coords, kernel):
    # convolution seems not correct
    '''
    img = coord_to_img(size, coords)
    img = convolve2d(img, kernel, mode='same')
    '''
    h, w = size
    img = np.zeros(size)
    r, _ = kernel.shape  # we assume the kernel is always square
    r = r // 2
    for pt in coords:
        y, x = pt[1], pt[0]
        # we put one kernel on the image at pt each time
        # check if the kernel at pt has part that outside the image
        imgh = max(0, y - r), min(h, y + r) # the top and bottom coordination in image
        imgw = max(0, x - r), min(w, x + r)
        kh = r - min(r, y), r + min(h - y, r)
        kw = r - min(r, x), r + min(w - x, r)
        img[imgh[0]:imgh[1], imgw[0]:imgw[1]] = np.maximum(
        img[imgh[0]:imgh[1], imgw[0]:imgw[1]],
        kernel[kh[0]:kh[1], kw[0]:kw[1]])
    return img * 255

def get_coords_from_heatmap(heatmap, dist = 2, pt_num = 50, thr_rel = 0.5, debug = False):
    '''
    Given,
        heatmap, 2D numpy array that could be a heatmap or a unsigned distance filed
            the rest parameters are not suggested to change, at least they work pretty well under current
            dectection task...
    Return,
        the keypoint extracted from the given input
    '''
    if debug:
        print("Log:\tgetting coords from UDF")

    # peak the local maximum from the heatmap, this is another candidate set of real keypoints
    coords = peak_local_max(heatmap, min_distance=dist, exclude_border = False, threshold_rel = thr_rel)
    if len(coords) > pt_num:
        coords = coords[np.argsort(heatmap[tuple(coords.T)])[-pt_num:]]
    if len(coords) > 0:
        coords[:, [0, 1]] = coords[:, [1, 0]]
    
    if debug:
        print("Log:\tDone")

    return coords

def plot_points(hm, coords, img = None):
    if img is not None:
        img = os.path.join("../data/sample/train/00/", img.replace(".svg", ".png"))
        assert os.path.exists(img)
        img = Image.open(img)
        plt.imshow(img)
        plt.imshow(hm, cmap='hot_r', alpha = 0.5)
    else:
        plt.imshow(hm, cmap='hot_r')
    plt.scatter(coords[:, 0], coords[:, 1], marker='+', c='lime', label='end', linewidths=0.3)
    plt.show()

if __name__ == "__main__":
    '''
    dataset creation, 
    step 1:
        generate svg file from quick draw and creative dataset
    '''
    '''export creative sketch dataset'''
    CREATIVE_SKETCH = "../data/creative" # this actually point the the "processed_data" folder
    CREATIVE_SKETCH_OUT = "../data/creative_processed/svg"
    prepare_creative_sketch(CREATIVE_SKETCH, CREATIVE_SKETCH_OUT, 0)

    '''export quick draw sketch dataset'''
    QUICK_DRAW_SKETCH = "../data/quick_draw/" # this actually point the the "processed_data" folder
    QUICK_DRAW_OUT = "../data/quick_draw_processed/svg"
    export_ratio = 2e4 / 5e7
    # # DON'T try to export the full dataset, there are 50M sketches!
    # # TODO: add filter to rule out categories that is not good for training
    prepare_quick_draw(QUICK_DRAW_SKETCH, QUICK_DRAW_OUT, 0, export_ratio)
    
    ## sample from the full dataset
    FULL_SVG = "../data/full/svg"
    cs_svgs = os.listdir(CREATIVE_SKETCH_OUT)
    cs_svgs = [os.path.join(CREATIVE_SKETCH_OUT, p) for p in cs_svgs]
    qd_svgs = os.listdir(QUICK_DRAW_OUT)
    qd_svgs = [os.path.join(QUICK_DRAW_OUT, p) for p in qd_svgs]
    # copy the sampled svg to a new folder
    counter = 0
    full_svgs = cs_svgs + qd_svgs
    for svg in full_svgs:
        shutil.copy(svg, os.path.join(FULL_SVG, "%07d.svg"%counter))
        counter += 1
    '''
    After finish this step, goto junction_detection.py for step2 to detect all keypoints of given svg
    '''