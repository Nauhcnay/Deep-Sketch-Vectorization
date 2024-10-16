import pydiffvg
import argparse
import ttools.modules
import torch
import skimage.io
import os

from pydiffvg.shape import Path

gamma = 1.0

def dist_loss(points, dist = 10):
    '''compute the distance from each point's closest one, require the point distance to be zero if two are close enough to each other '''
    # create distance matrix of all points
    pt_dist = []
    stroke_lens = []
    pts = torch.cat(points).to(pydiffvg.get_device())
    pts_col = pts.unsqueeze(1)
    pts_row = pts.unsqueeze(0)
    # dis_matrix = torch.sqrt(torch.sum(torch.square(pts_col - pts_row), axis = -1)).squeeze()
    # this should be equivalent to the uppder line, however, will not cause gradient explosion
    dis_matrix = torch.sum(torch.abs(pts_col - pts_row), axis = -1).squeeze()
    # get the all point pair index that their distance is smaller than the threshold
    for i in range(dis_matrix.shape[0]):
        # find closest point to point i
        if i % 2 == 0: stroke_lens.append(dis_matrix[i][i + 1].unsqueeze(0))
        selected_pts_idx = torch.argsort(dis_matrix[i, :])
        for j in selected_pts_idx:
            if i == j: continue
            if i % 2 and j == i + 1: continue
            if dis_matrix[i][j] > dist: break
            pt_dist.append(dis_matrix[i][j].unsqueeze(0))
            break
    if len(pt_dist) > 0:
        pt_dist = torch.cat(pt_dist)
        pt_dist = pt_dist.mean()
    else:
        pt_dist = False
    if len(stroke_lens) > 0:
        stroke_lens = torch.cat(stroke_lens)
        stroke_lens = stroke_lens[stroke_lens < 2]
        if len(stroke_lens) > 0:
            stroke_lens = -stroke_lens.mean()    
        else:
            stroke_lens = False    
    else:
        stroke_lens = False
    return pt_dist, stroke_lens


def main(args):
    perception_loss = ttools.modules.LPIPS().to(pydiffvg.get_device())

    _, fn = os.path.split(args.target)
    name, _ = os.path.splitext(fn)
    target = torch.from_numpy(skimage.io.imread(args.target)).to(torch.float32) / 255.0
    target = target[:, :, :3]
    target = target.pow(gamma) # why we need to consider gamma?
    target = target.to(pydiffvg.get_device())
    target = target.unsqueeze(0)
    target = target.permute(0, 3, 1, 2) # NHWC -> NCHW

    # read the svg
    canvas_width, canvas_height, shapes, shape_groups = \
        pydiffvg.svg_to_scene(args.svg)
    
    # what this function does?
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)

    # create render
    render = pydiffvg.RenderFunction.apply
    
    # render the output
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None, # bg
                 *scene_args)
    # The output image is in linear RGB space. Do Gamma correction before saving the image.
    pydiffvg.imwrite(img.cpu(), './refinement/%s/init.png'%name, gamma=gamma)
    point_vars = []
    for path in shapes:
        path.points.requires_grad = True
        path.stroke_width.requires_grad = True
        # todo: add more control variables for optimization
        point_vars.append(path.points)
    color_vars = {}
    for group in shape_groups:
        if group.fill_color is not None:
            group.fill_color.requires_grad = True
            color_vars[group.fill_color.data_ptr()] = group.fill_color
        else:
            group.stroke_color.requires_grad = True
            color_vars[group.stroke_color.data_ptr()] = group.stroke_color
    color_vars = list(color_vars.values())

    # Optimize
    points_optim = torch.optim.Adam(point_vars, lr=1.0)
    color_optim = torch.optim.Adam(color_vars, lr=0.01)

    # Adam iterations.
    for t in range(args.num_iter):
        print('iteration:', t)
        points_optim.zero_grad()
        color_optim.zero_grad()
        # Forward pass: render the image.
        scene_args = pydiffvg.RenderFunction.serialize_scene(\
            canvas_width, canvas_height, shapes, shape_groups)
        img = render(canvas_width, # width
                     canvas_height, # height
                     2,   # num_samples_x
                     2,   # num_samples_y
                     0,   # seed
                     None, # bg
                     *scene_args)
        # Compose img with white background
        img = img[:, :, 3:4] * img[:, :, :3] + torch.ones(img.shape[0], img.shape[1], 3, device = pydiffvg.get_device()) * (1 - img[:, :, 3:4])
        # Save the intermediate render.
        pydiffvg.imwrite(img.cpu(), './refinement/%s/iter_%d.png'%(name, t), gamma=gamma)
        img = img[:, :, :3]
        # Convert img from HWC to NCHW
        img = img.unsqueeze(0)
        img = img.permute(0, 3, 1, 2) # NHWC -> NCHW
        if args.use_lpips_loss:
            loss = perception_loss(img, target)
        else:
            loss = (img - target).pow(2).mean()
        # add other loss for better topology reconstruction
        loss_distance, loss_length = dist_loss(point_vars)
        
        if loss_distance:
            loss += 0.001 * loss_distance
        print('render loss:', loss.item())
    
        # Backpropagate the gradients.
        loss.backward()
        # Take a gradient descent step.
        points_optim.step()
        color_optim.step()
        for group in shape_groups:
            group.stroke_color.data.clamp_(0.0, 1.0)

        if t % 10 == 0 or t == args.num_iter - 1:
            pydiffvg.save_svg('./refinement/%s/iter_%d.svg'%(name, t),
                              canvas_width, canvas_height, shapes, shape_groups)

    # Render the final result.
    scene_args = pydiffvg.RenderFunction.serialize_scene(\
        canvas_width, canvas_height, shapes, shape_groups)
    img = render(canvas_width, # width
                 canvas_height, # height
                 2,   # num_samples_x
                 2,   # num_samples_y
                 0,   # seed
                 None, # bg
                 *scene_args)
    # Save the intermediate render.
    pydiffvg.imwrite(img.cpu(), './refinement/%s/final.png'%name, gamma=gamma)
    # Convert the intermediate renderings to a video.
    from subprocess import call
    call(["ffmpeg", "-framerate", "24", "-i",
        "./refinement/" + name + "/iter_%d.png", "-vb", "20M",
        "./refinement/%s/out.mp4"%name])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("svg", help="source SVG path")
    parser.add_argument("target", help="target image path")
    parser.add_argument("--use_lpips_loss", dest='use_lpips_loss', action='store_true')
    parser.add_argument("--num_iter", type=int, default=250)
    args = parser.parse_args()
    main(args)
