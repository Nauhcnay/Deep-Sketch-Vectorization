import torch
import torch.nn as nn
import torch.nn.functional as F

from network.keypoint import Siren, ResidualModule, ResNextModule


class NDC_resnet_base(nn.Module):
    def __init__(self, device):
        super(NDC_resnet_base, self).__init__()
        self.keypt_init = None
        self.ic = None
        self.jc = None
        self.device = device

    # here defines functions that could be used by both netwokrs
    def init_keypt(self, x, gsizes):
        b, _, h, w = x.shape
        base_coord = []
        for gsize in gsizes:
            gsize = float(gsize)
            hg = int(h * gsize)
            wg = int(w * gsize)
            xs = torch.linspace(start = gsize / 2, end = wg - gsize / 2, steps = w) 
            ys = torch.linspace(start = gsize / 2, end = hg - gsize / 2, steps = h)
            xx, yy = torch.meshgrid(xs, ys, indexing = 'xy')
            base_coord.append(torch.stack((xx, yy), dim = -1).unsqueeze(0).permute(0, 3, 1, 2))
        base_coord = torch.cat(base_coord, dim = 0)
        return base_coord.to(self.device)
    
    def add_coord_channels(self, x):
        # get hard coded i, j channel of x
        b, c, h, w = x.shape
        if self.jc is None:
            self.gen_coord_j(b, h , w)
        else:
            _, _, h1, w1 = self.jc.shape
            if h1 != h or w1 != w: self.gen_coord_j(b, h, w)
                
        if self.ic is None:
            self.gen_coord_i(b, h, w)
        else:
            _, _, h1, w1 = self.ic.shape
            if h1 != h or w1 != w: self.gen_coord_i(b, h, w)
        assert self.ic is not None
        assert self.jc is not None
        return torch.cat((x, self.ic[0:b, ...], self.jc[0:b, ...]), dim = 1)
    
    def gen_coord_i(self, b, h, w):
        self.ic = torch.arange(0, w, 1).reshape(1, w).unsqueeze(0).unsqueeze(0).expand(b, 1, h, w).to(self.device).float()
        self.ic = self.standardize(self.ic)
        self.ic.requires_grad = False
    
    def gen_coord_j(self, b, h, w):
        self.jc = torch.arange(0, h, 1).reshape(h, 1).unsqueeze(0).unsqueeze(0).expand(b, 1, h, w).to(self.device).float()       
        self.jc = self.standardize(self.jc)
        self.jc.requires_grad = False

    def standardize(self, x):
        x = (x - x.mean()) / x.std()
        x = x / x.abs().max()
        return x

'''
Chuan's version of Nerual Dual Contouring network
'''
class conv2d_ndc(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, batch_norm = False,
            activation = torch.nn.LeakyReLU(negative_slope = 0.01, inplace = True), 
            stride = 1, padding = 'same', bias = True, dilation = 1):
        super(conv2d_ndc, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, dilation = dilation, stride = stride, padding = padding, bias = bias)
        self.bn = torch.nn.BatchNorm2d(out_channels) if batch_norm else None
        self.act = activation
    def forward(self, input):
        output = self.conv(input)
        if self.bn is not None:
            output = self.bn(output)
        if self.act is not None:
            output = self.act(output)
        return output

class resnet_block_ndc(nn.Module):
    def __init__(self, in_channels, out_channels,
            # inplace could reduce the memory usage, but nothing different at performance
            activation = nn.LeakyReLU(negative_slope = 0.01, inplace = True),
            batch_norm = False,
            kernel_size = 1,
            dilation = 1):
        super(resnet_block_ndc, self).__init__()
        self.conv_1 = conv2d_ndc(in_channels, out_channels, kernel_size, dilation = dilation, stride=1, padding='same', bias=True)
        self.conv_2 = conv2d_ndc(in_channels, out_channels, kernel_size, dilation = dilation, stride=1, padding='same', bias=True, activation = None)
        self.act = activation
        self.bn = torch.nn.BatchNorm2d(channels) if batch_norm else None

    def forward(self, input):
        output = self.conv_1(input)
        output = self.conv_2(output) + input
        if self.bn is not None:
            output = self.bn(output)
        if self.act is not None:
            output = self.act(output)
        return output

class CNN_2d_resnet(NDC_resnet_base):

    def __init__(self, 
            device, 
            channels = 64, 
            drop_out = False, 
            out_bool = True, 
            out_float = True, 
            multi_scale = False, 
            batch_norm = True, 
            layers = [6, 6, 2],
            coord_conv = False, 
            noisy = False,
            resnext_feature = False,
            resnext_input = False):
        super(CNN_2d_resnet, self).__init__(device)
        self.ef_dim = channels
        self.out_bool = out_bool
        self.out_float = out_float
        self.layers = layers
        self.multi_scale = multi_scale
        self.noisy = noisy
        self.coord_conv = coord_conv
        self.resnext_feature = resnext_feature
        self.resnext_input = resnext_input
        self.dim_plus = [0] * layers[0]
        if self.coord_conv:
            self.dim_plus[0] = 2
        act = nn.LeakyReLU(negative_slope = 0.01, inplace = True)
        if drop_out:
            self.dropout = nn.Dropout(p = 0.2)
        else:
            self.dropout = None
        '''
        Domain coversion
        '''
        if self.resnext_input:
            self.conv_input = nn.Sequential(
                ResNextModule( 
                    inp_channels=1,
                    out_channels=self.ef_dim,
                    intermediate_channels=16,
                    final_batchnorm = batch_norm,
                    cardinality = True),
                conv2d_ndc(
                    self.ef_dim, self.ef_dim, 2, 
                    batch_norm = batch_norm, 
                    activation = act, 
                    padding = 0))
        else:
            self.conv_input = conv2d_ndc(1, self.ef_dim, 2, 
                batch_norm = batch_norm, 
                activation = act, 
                padding = 0)
        '''
        Multi-Grid branches
        '''
        if self.multi_scale:
            self.feature_extraction_3x3 = nn.ModuleList([
                nn.Sequential(
                    ResidualModule(self.ef_dim + self.dim_plus[i], self.ef_dim, final_batchnorm = batch_norm, 
                        activation_function = act, kernel_size = 3),
                    conv2d_ndc(self.ef_dim, self.ef_dim, 3, dilation = 1,
                        batch_norm = batch_norm, activation = act))
                for i in range(self.layers[0])
            ])
            
            self.feature_extraction_7x7 = nn.ModuleList([
                nn.Sequential(
                    ResidualModule(self.ef_dim + self.dim_plus[i], self.ef_dim, final_batchnorm = batch_norm, 
                        activation_function = act, kernel_size = 3),
                    conv2d_ndc(self.ef_dim, self.ef_dim, 3, dilation = 3,
                        batch_norm = batch_norm, activation = act))
                for i in range(self.layers[0])
            ])
            ## use larger grid size if the input UDF is noisy
            if self.noisy:
                self.feature_extraction_11x11 = nn.ModuleList([
                    nn.Sequential(
                        ResidualModule(self.ef_dim + self.dim_plus[i], self.ef_dim, final_batchnorm = batch_norm, 
                            activation_function = act, kernel_size = 3),
                        conv2d_ndc(self.ef_dim, self.ef_dim, 3, dilation = 5,
                            batch_norm = batch_norm, activation = act))
                    for i in range(self.layers[0])
                ])
            else:
                self.feature_extraction_5x5 = nn.ModuleList([
                nn.Sequential(
                    ResidualModule(self.ef_dim + self.dim_plus[i], self.ef_dim, final_batchnorm = batch_norm, 
                        activation_function = act, kernel_size = 3),
                    conv2d_ndc(self.ef_dim, self.ef_dim, 3, dilation = 2,
                        batch_norm = batch_norm, activation = act))
                for i in range(self.layers[0])
                ])
            '''
            Bottle Neck (sort like)
            '''
            self.bottle_neck = nn.ModuleList([
                ResidualModule(
                    self.ef_dim * 3 if i == 0 else self.ef_dim, self.ef_dim, 
                    activation_function = act, 
                    final_batchnorm = batch_norm, 
                    kernel_size = 3)
                for i in range(self.layers[1])
            ])
        else:
            if self.resnext_feature:
                self.feature_extraction = nn.ModuleList([
                    ResNextModule( 
                    inp_channels=self.ef_dim,
                    out_channels=self.ef_dim,
                    intermediate_channels=int(self.ef_dim / 8),
                    final_batchnorm = batch_norm,
                    cardinality = True) 
                    for i in range(self.layers[0])
                    ])
            else:
                self.feature_extraction = nn.ModuleList([
                    nn.Sequential(
                        ResidualModule(
                            self.ef_dim + self.dim_plus[i], 
                            self.ef_dim, 
                            final_batchnorm = batch_norm, 
                            activation_function = act, 
                            kernel_size = 3),
                        conv2d_ndc(
                            self.ef_dim, 
                            self.ef_dim, 3, 
                            batch_norm = batch_norm, 
                            activation = act))
                    for i in range(self.layers[0])
                ])
            '''
            Bottle Neck (sort like)
            '''
            self.bottle_neck = nn.ModuleList([
                ResidualModule(
                    self.ef_dim, 
                    self.ef_dim, 
                    kernel_size = 3, 
                    activation_function = act, 
                    final_batchnorm = batch_norm)
                for i in range(self.layers[1])
            ])
        '''
        MLP (sort like, but I think this structure is not that reasonable... why?)
        '''
        self.mlp = nn.ModuleList([
            conv2d_ndc(
                self.ef_dim, 
                self.ef_dim, 1, 
                stride=1, 
                batch_norm = batch_norm, 
                activation = act)
            for i in range(self.layers[2])
        ])

        if self.out_bool:
            self.conv_out_bool = nn.Conv2d(self.ef_dim, 5, 1, stride=1, padding=0, bias=True)
        
        if self.out_float:
            self.conv_out_float = nn.Conv2d(self.ef_dim, 2, 1, stride=1, padding=0, bias=True)
    
    def forward(self, x, gsizes = None):
        # convert image to UDF domain
        out = self.conv_input(x)
        if self.coord_conv:
            assert gsizes is not None
            base_coord = self.init_keypt(out, gsizes)
            out = torch.cat((out, base_coord), dim = 1)
        # extract features
        if self.multi_scale and self.resnext_feature == False:
            out_3x3 = out
            out_7x7 = out
            out_5x5 = out
            out_11x11 = out
            for i in range(self.layers[0]):
                out_3x3 = self.feature_extraction_3x3[i](out_3x3)
            if self.noisy == False:
                for i in range(self.layers[0]):
                    out_5x5 = self.feature_extraction_5x5[i](out_5x5)
            for i in range(self.layers[0]):
                out_7x7 = self.feature_extraction_7x7[i](out_7x7)
            if self.noisy:
                for i in range(self.layers[0]):
                    out_11x11 = self.feature_extraction_11x11[i](out_11x11)
                out = torch.cat((out_3x3, out_7x7, out_11x11), dim = 1)
            else:
                out = torch.cat((out_3x3, out_5x5, out_7x7), dim = 1)
        else:
            for i in range(self.layers[0]):
                out = self.feature_extraction[i](out)

        for i in range(self.layers[1]):
            if self.dropout is not None:
                out = self.dropout(out)
            out = self.bottle_neck[i](out)
    
        for i in range(self.layers[2]):
            out = self.mlp[i](out)
    
        b, _, _, _ = out.shape

        '''
        Output to different predictions
        '''

        if self.out_bool and self.out_float:
            out_bool = self.conv_out_bool(out)
            out_float = self.conv_out_float(out)

            b, _, _, _ = out_float.shape
            return out_bool, out_float
            
        elif self.out_bool:
            out_bool = self.conv_out_bool(out)
            out_z = self.conv_out_z(out)
            return out_bool, out_z

        elif self.out_float:
            out_float = self.conv_out_float(out)
            return out_float

'''
For Debug
'''
class resnet_block(nn.Module):
    def __init__(self, ef_dim, kernel_size = 1):
        super(resnet_block, self).__init__()
        self.ef_dim = ef_dim
        self.conv_1 = nn.Conv2d(self.ef_dim, self.ef_dim, kernel_size, stride=1, padding='same', bias=True)
        self.conv_2 = nn.Conv2d(self.ef_dim, self.ef_dim, kernel_size, stride=1, padding='same', bias=True)

    def forward(self, input):
        output = self.conv_1(input)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        output = self.conv_2(output)
        output = output+input
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        return output

class CNN_2d_rec7_resnet(NDC_resnet_base):

    def __init__(self, device, out_bool = True, multi_gt_channel = False,
            out_float = True, large_kernel = False, coord_conv = False,
            layers = [2, 6, 2], channels = None):
        super(CNN_2d_rec7_resnet, self).__init__(device)
        # self.device = device
        self.ef_dim = 64
        self.out_bool = out_bool
        self.out_float = out_float
        self.coord_conv = coord_conv
        
        if large_kernel:
            if coord_conv:
                self.conv_0 = nn.Conv2d(3, self.ef_dim, 7, stride=1, padding='same', bias=True)
            else:
                self.conv_0 = nn.Conv2d(1, self.ef_dim, 7, stride=1, padding='same', bias=True)
            self.res_1 = resnet_block(self.ef_dim, kernel_size = 1) # I guess the kernel size of resnet should still be 1? but why?
            self.conv_1 = nn.Conv2d(self.ef_dim, self.ef_dim, 5, stride=1, padding='same', bias=True)
            self.res_2 = resnet_block(self.ef_dim, kernel_size = 1)
            self.conv_2 = nn.Conv2d(self.ef_dim, self.ef_dim, 3, stride=1, padding='same', bias=True)
        else:
            if coord_conv:
                self.conv_0 = nn.Conv2d(3, self.ef_dim, 3, stride=1, padding='same', bias=True)
            else:
                self.conv_0 = nn.Conv2d(1, self.ef_dim, 3, stride=1, padding='same', bias=True)
            self.res_1 = resnet_block(self.ef_dim)
            self.conv_1 = nn.Conv2d(self.ef_dim, self.ef_dim, 3, stride=1, padding='same', bias=True)
            self.res_2 = resnet_block(self.ef_dim)
            self.conv_2 = nn.Conv2d(self.ef_dim, self.ef_dim, 3, stride=1, padding='same', bias=True)

        self.res_3 = resnet_block(self.ef_dim)
        self.res_4 = resnet_block(self.ef_dim)
        self.res_5 = resnet_block(self.ef_dim)
        self.res_6 = resnet_block(self.ef_dim)
        self.res_7 = resnet_block(self.ef_dim)
        self.res_8 = resnet_block(self.ef_dim)

        self.conv_3 = nn.Conv2d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv2d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        
        self.conv_out_skel = nn.Conv2d(self.ef_dim, 1, 1, stride=1, padding=0, bias=True)

        if self.out_bool:
            # we need to output 4 channels for the prediction, 2 for edge of one direction, respectively
            self.conv_out_bool = nn.Conv2d(self.ef_dim, 4, 1, stride=1, padding=0, bias=True)
        if self.out_float:
            self.conv_out_float = nn.Conv2d(self.ef_dim, 2, 1, stride=1, padding=0, bias=True)
    
    def forward(self, x):
        # we need to init keypoint coordinates
        self.init_keypt(x)
        if self.coord_conv:
            out = self.add_coord_channels(x)
        else:
            out = x
        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.res_1(out)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.res_2(out)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.res_3(out)
        out = self.res_4(out)
        out = self.res_5(out)
        out = self.res_6(out)
        out = self.res_7(out)
        out = self.res_8(out)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=False)
        out_skel = self.conv_out_skel(out)
        b, _, _, _ = out.shape

        if self.out_bool and self.out_float:
            out_bool = self.conv_out_bool(out)
            out_float = self.conv_out_float(out)    
            b, _, _, _ = out_float.shape
            return out_bool, self.keypt_init[:b, :, :, :] + out_float.clamp(-0.5, 0.5), None, out_skel
            
        
        elif self.out_bool:
            out_bool = self.conv_out_bool(out)
            return out_bool

        elif self.out_float:
            out_float = self.conv_out_float(out)
            return self.keypt_init + out_float.clamp(-0.5, 0.5)


class resnet_block3d(nn.Module):
    def __init__(self, ef_dim):
        super(resnet_block3d, self).__init__()
        self.ef_dim = ef_dim
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)

    def forward(self, input):
        output = self.conv_1(input)
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        output = self.conv_2(output)
        output = output+input
        output = F.leaky_relu(output, negative_slope=0.01, inplace=True)
        return output

class CNN_3d_rec7_resnet(nn.Module):

    def __init__(self, out_bool, out_float, is_undc=False):
        super(CNN_3d_rec7_resnet, self).__init__()
        self.ef_dim = 64
        self.out_bool = out_bool
        self.out_float = out_float
        
        self.conv_0 = nn.Conv3d(1, self.ef_dim, 3, stride=1, padding=0, bias=True)

        self.res_1 = resnet_block3d(self.ef_dim)
        self.conv_1 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)

        self.res_2 = resnet_block3d(self.ef_dim)
        self.conv_2 = nn.Conv3d(self.ef_dim, self.ef_dim, 3, stride=1, padding=0, bias=True)

        self.res_3 = resnet_block3d(self.ef_dim)
        self.res_4 = resnet_block3d(self.ef_dim)
        self.res_5 = resnet_block3d(self.ef_dim)
        self.res_6 = resnet_block3d(self.ef_dim)
        self.res_7 = resnet_block3d(self.ef_dim)
        self.res_8 = resnet_block3d(self.ef_dim)

        self.conv_3 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)
        self.conv_4 = nn.Conv3d(self.ef_dim, self.ef_dim, 1, stride=1, padding=0, bias=True)

        if self.out_bool:
            self.conv_out_bool = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)
            
        if self.out_float:
            self.conv_out_float = nn.Conv3d(self.ef_dim, 3, 1, stride=1, padding=0, bias=True)

    def forward(self, x):
        out = x

        out = self.conv_0(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.res_1(out)
        out = self.conv_1(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.res_2(out)
        out = self.conv_2(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        out = self.res_3(out)
        out = self.res_4(out)
        out = self.res_5(out)
        out = self.res_6(out)
        out = self.res_7(out)
        out = self.res_8(out)

        out = self.conv_3(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)
        out = self.conv_4(out)
        out = F.leaky_relu(out, negative_slope=0.01, inplace=True)

        if self.out_bool and self.out_float:
            out_bool = self.conv_out_bool(out)
            out_float = self.conv_out_float(out)
            return torch.sigmoid(out_bool), out_float
        elif self.out_bool:
            out_bool = self.conv_out_bool(out)
            return out_bool
            # return torch.sigmoid(out_bool)
        elif self.out_float:
            out_float = self.conv_out_float(out)
            return out_float
