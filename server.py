import eel
import numpy as np
import re
import base64
import io
from pathlib import Path
from PIL import Image
from predict_s1 import open_img, vectorize, usm_surgery, finalize, usm_to_regions
# from utils.save_string_to_path_in_background import save_string_to_path_in_background

usm_width = 0
usm_height = 0

DEBUG_FILE = None


def debuginit(outname='server-debug-commands.py'):
    global DEBUG_FILE
    DEBUG_FILE = open(outname, 'w')
    DEBUG_FILE.write(f"""
## Remove randomness for deterministic testing
import random
random.seed(0)
import numpy as np
np.random.seed(0)

from server import *
eel.init('web')
""")
    DEBUG_FILE.flush()
    print(f"Run `python {outname}` to debug the last command.")


def debugme(functionname, *args, **kwargs):
    # Silently skip
    if DEBUG_FILE is None:
        return

    outname = DEBUG_FILE.name
    DEBUG_FILE.write(f"""
{functionname}( *{repr(args)}, **{repr(kwargs)} )
""")
    DEBUG_FILE.flush()
    print(f"{outname}` updated with {functionname}()")


def debugwrap(function):
    print("in debugwrap")

    def function_wrap(*args, **kwargs):
        print("in functionwrap")
        debugme(function.__name__, *args, **kwargs)
        return function(*args, **kwargs)
    function_wrap.__name__ = function.__name__
    return function_wrap


@eel.expose
# @debugwrap
def preprocess_file(name, data, thin=False, line_extractor=False, resize_to=512):
    prefix = name.split('.')[0]
    img_dict = re.match(
        "data:(?P<type>.*?);(?P<encoding>.*?),(?P<data>.*)", data).groupdict()
    binary = base64.b64decode(img_dict['data'])
    image = Image.open(io.BytesIO(binary))
    # generate image
    size = open_img(image, thin=thin, line_extractor=line_extractor,
                    name=prefix, resize_to=int(resize_to))
    # call back
    eel.updateRaster(size[1], size[0])


@eel.expose
# @debugwrap
def process_file(name):
    prefix = name.split('.')[0]
    if not Path("./web/output/{}_raw.svg".format(prefix)).exists():
        # first step
        keypt_pre_dict = vectorize(
            path_model_ndc="./pretrained/ndc_full.pth",
            path_model_udf="./pretrained/udf_full.pth")
    else:
        # read from file
        keypt_pre_dict = np.load(
            "./web/output/{}_udf.npz".format(prefix), allow_pickle=True)
    un_usm_regions, un_usm_indices = usm_to_regions(
        keypt_pre_dict['usm_uncertain'][-1])  # read the latest version
    new_regions = {}  # region index -> region info
    region_dict = {}  # position -> region index
    k_list = []
    v_list = []
    for k, v in un_usm_regions.items():
        k_list.append(int(k))
        v_list.append(v.shape[0])
        for r_index in range(v.shape[0]):
            region_dict["r{}c{}".format(v[r_index, 1], v[r_index, 0])] = int(k)
        new_regions[int(k)] = {"pos": [v[:, 1].tolist(), v[:, 0].tolist()],
                               "center": [np.mean(v[:, 1]), np.mean(v[:, 0])],
                               "bound": [int(min(v[:, 1])), int(min(v[:, 0])), int(max(v[:, 1])-min(v[:, 1])+1), int(max(v[:, 0])-min(v[:, 0])+1)]
                               }
    nums = 10  # number of displayed regions
    top_i = (-np.array(v_list)).argsort()[:nums]  # top indices
    top_k = np.array(k_list)[top_i].tolist()      # top nums regions
    usm_applied = keypt_pre_dict['usm_applied'][-1]
    '''
    now usm are stored as segmented maps, each region has its own label, 
    so we can convert it to dictionary which contains indices per region
    usm_regions, also get the binary map as usm_indices
    '''
    # usm_indices = np.transpose(np.nonzero(usm_applied == True))
    usm_regions, usm_indices = usm_to_regions(usm_applied)
    global usm_width, usm_height
    usm_width = usm_applied.shape[1]
    usm_height = usm_applied.shape[0]
    eel.updateVectorizedFiles(prefix,
                              keypt_pre_dict['end_point'][-1][:, 0].tolist(
                              ), keypt_pre_dict['end_point'][-1][:, 1].tolist(),
                              keypt_pre_dict['sharp_turn'][-1][:, 0].tolist(
                              ), keypt_pre_dict['sharp_turn'][-1][:, 1].tolist(),
                              keypt_pre_dict['junc'][-1][:, 0].tolist(
                              ), keypt_pre_dict['junc'][-1][:, 1].tolist(),
                              usm_indices[:, 1].tolist(
                              ), usm_indices[:, 0].tolist(),
                              top_k, new_regions, region_dict, un_usm_indices[:, 1].tolist(), un_usm_indices[:, 0].tolist())


@eel.expose
# @debugwrap
def update(name, end_x, end_y, sharp_x, sharp_y, junc_x, junc_y, usm_x, usm_y, bezier=False, rdp=False):
    prefix = name.split('.')[0]
    end_point = np.column_stack((end_x, end_y))
    sharp_turn = np.column_stack((sharp_x, sharp_y))
    junc = np.column_stack((junc_x, junc_y))
    global usm_width, usm_height
    usm = np.full((usm_height, usm_width), False)  # order matters
    usm[(usm_y, usm_x)] = True
    keypt_dict = {'end_point': end_point,
                  'sharp_turn': sharp_turn,
                  'junc': junc}
    usm_applied, usm_uncertain, keypts, svg_string = usm_surgery(
        # second step
        'modify', usm, keypt_dict, name=name, canvas_size=(usm_height/2, usm_width/2))
    eel.updateRefinedFiles(prefix, svg_string)
    # save_string_to_path_in_background(svg_string, "./web/output/{}_update.svg".format(prefix))
    # final_process(name, bezier=bezier, rdp=rdp, auto=True)


@eel.expose
# @debugwrap
def final_process(name, bezier=False, rdp=False, auto=False):
    prefix = name.split('.')[0]
    # final step
    finalize(bezier=bezier, rdp_simplify=rdp)
    eel.updateFinalFiles(prefix, auto)


if __name__ == "__main__":
    # debuginit()
    eel.init('web')
    print("Open a web browser to: http://localhost:8888/main.html")
    # eel.start('main.html', mode=False)
    eel.start('main.html', mode=False, all_interfaces=True, port=8888)
