import os
import numpy as np
from os.path import join
from utils.svg_tools import open_svg_flatten
from utils import extrude_linestrips as el
from tqdm import tqdm

def write_obj(dire, vertices, triangles):
    fout = open(dire, 'w')
    for ii in range(len(vertices)):
        fout.write("v "+str(vertices[ii,0])+" "+str(vertices[ii,1])+" "+str(vertices[ii,2])+"\n")
    for ii in range(len(triangles)):
        fout.write("f "+str(triangles[ii,0]+1)+" "+str(triangles[ii,1]+1)+" "+str(triangles[ii,2]+1)+"\n")
    fout.close()

path_to_svg = './data/sample/svg'
path_to_obj = '../NDC/data_preprocessing/objs'
for svg in tqdm(os.listdir(path_to_svg)):
    paths, _ = open_svg_flatten(join(path_to_svg, svg))
    linestrips = []
    for i in range(len(paths)):
        sx = paths[i].start.real
        sy = paths[i].start.imag
        ex = paths[i].end.real
        ey = paths[i].end.imag
        linestrips.append((sx, sy))
        linestrips.append((ex, ey))
    zs = np.linspace(0, 50, 50)
    V, F = el.extrude_linestrip(zs, linestrips)
    V = np.asarray(V)
    F = np.asarray(F)
    write_obj(join(path_to_obj, svg.replace(".svg", ".obj")), V, F)

