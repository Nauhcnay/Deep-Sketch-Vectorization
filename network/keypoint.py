import torch
import torch.nn as nn


class MyConv(nn.Module):
    def __init__(self, inp_channels, out_channels, kernel_size=3, padding=1, stride=1, batch_norm=True):
        # a nice introduce to super()
        # https://realpython.com/python-super/
        super(MyConv, self).__init__()
        self.inpc = inp_channels
        self.outc = out_channels
        self.conv = nn.Conv2d(inp_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding)
        if batch_norm:
            self.bn = nn.BatchNorm2d(out_channels)
        else:
            self.bn = None
        if padding != (kernel_size - 1) // 2 and padding != 'same':
            print("Warning: convolution is not 'same'")

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        return x


class ResNextModule(nn.Module):
    # we also need to map the pixle map to UDF domain
    def __init__(self, inp_channels, out_channels, intermediate_channels,
        cardinality = False ,
        activation_function=nn.LeakyReLU(),
        final_batchnorm = False,
        as_back_bone = False,
        width = 8):
        super(ResNextModule, self).__init__()
        self.dropout_rate = 0.1
        self.cardinality = cardinality
        self.width = width # we need to fix this number cause we hard code the branches
        if as_back_bone:
            assert inp_channels % (self.width*4) == 0
            intermediate_channels = int(inp_channels /  (self.width*4))
            self.intermediate_channels = intermediate_channels
        else:
            self.intermediate_channels = intermediate_channels
        self.final_batchnorm = final_batchnorm
        self.finalbathcnorm = nn.BatchNorm2d(out_channels)
        self.activation = activation_function
        # for back compatibelity
        if cardinality == False:
            self.width = 3
            self.path1 = nn.Sequential(
                torch.nn.Conv2d(in_channels=inp_channels, out_channels=intermediate_channels, kernel_size=1),
                nn.BatchNorm2d(intermediate_channels),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels,
                                kernel_size=3, padding=1),
                nn.BatchNorm2d(intermediate_channels),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout_rate),
                torch.nn.Conv2d(in_channels=intermediate_channels, out_channels=out_channels, kernel_size=1),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(),
                nn.Dropout(self.dropout_rate)
            )
            self.path2 = nn.Sequential(
                    torch.nn.Conv2d(in_channels=inp_channels, out_channels=intermediate_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(intermediate_channels),
                    nn.LeakyReLU(),
                    nn.Dropout(self.dropout_rate),
                    torch.nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels,
                                    kernel_size=3, padding=2, dilation=2),
                    nn.BatchNorm2d(intermediate_channels),
                    nn.LeakyReLU(),
                    nn.Dropout(self.dropout_rate),
                    torch.nn.Conv2d(in_channels=intermediate_channels, out_channels=out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(),
                    nn.Dropout(self.dropout_rate)
                )
            self.path3 = nn.Sequential(
                    torch.nn.Conv2d(in_channels=inp_channels, out_channels=intermediate_channels, kernel_size=3, padding=2, dilation=2),
                    nn.BatchNorm2d(intermediate_channels),
                    nn.LeakyReLU(),
                    nn.Dropout(self.dropout_rate),
                    torch.nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels,
                                    kernel_size=3, padding=2, dilation=2),
                    nn.BatchNorm2d(intermediate_channels),
                    nn.LeakyReLU(),
                    nn.Dropout(self.dropout_rate),
                    torch.nn.Conv2d(in_channels=intermediate_channels, out_channels=out_channels, kernel_size=1),
                    nn.BatchNorm2d(out_channels),
                    nn.LeakyReLU(),
                    nn.Dropout(self.dropout_rate)
                )
        else:
            if self.width == 8:
                self.pathes = nn.ModuleList([
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 1, k2 = 3, k3 = 1, d1 = 1, d2 = 1, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 1, d2 = 1, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 1, d2 = 2, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 2, d2 = 2, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 2, d2 = 3, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 3, d2 = 3, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 3, d2 = 4, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 4, d2 = 4, d3 = 1),
                    ])
            elif self.width == 16:
                self.pathes = nn.ModuleList([
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 1, k2 = 3, k3 = 1, d1 = 1, d2 = 1, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 1, d2 = 1, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 1, d2 = 2, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 2, d2 = 2, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 2, d2 = 3, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 3, d2 = 3, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 3, d2 = 4, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 4, d2 = 4, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 4, d2 = 5, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 5, d2 = 5, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 5, d2 = 6, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 6, d2 = 6, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 6, d2 = 7, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 7, d2 = 7, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 7, d2 = 8, d3 = 1),
                    self.build_block_(inp_channels, intermediate_channels, out_channels, k1 = 3, k2 = 3, k3 = 1, d1 = 8, d2 = 8, d3 = 1)
                    ])
            else:
                raise ValueError("unsupport ResNext branch width %d"%self.width)
        self.plusconv = MyConv(
            inp_channels = self.width * out_channels, 
            out_channels=out_channels, 
            kernel_size=1, 
            padding=0, 
            batch_norm=True)
        self.skipconv = torch.nn.Conv2d(
            in_channels=inp_channels, 
            out_channels=out_channels, 
            kernel_size=1)

    def build_block_(self, inp_channels, intermediate_channels, out_channels,
        k1 = 1, k2 = 3, k3 = 1, d1 = 1, d2 = 1, d3 = 1):
        return nn.Sequential(
        torch.nn.Conv2d(in_channels=inp_channels, out_channels=intermediate_channels, 
            kernel_size=k1, dilation=d1, padding='same'),
        nn.BatchNorm2d(intermediate_channels),
        self.activation,
        nn.Dropout(self.dropout_rate),
        torch.nn.Conv2d(in_channels=intermediate_channels, out_channels=intermediate_channels,
            kernel_size=k2, dilation=d2, padding='same'),
        nn.BatchNorm2d(intermediate_channels),
        self.activation,
        nn.Dropout(self.dropout_rate),
        torch.nn.Conv2d(in_channels=intermediate_channels, out_channels=out_channels, 
            kernel_size=k3, dilation=d3, padding='same'),
        nn.BatchNorm2d(out_channels),
        self.activation,
        nn.Dropout(self.dropout_rate))

    def forward(self, x):
        x_skip = self.skipconv(x)
        # for back compatibelity
        if self.cardinality == False:
            x1 = self.path1(x)
            x2 = self.path2(x)
            x3 = self.path3(x)
            x = self.plusconv(torch.cat((x1, x2, x3), 1))
        else:
            xs = [self.pathes[i](x) for i in range(self.width)]
            x = self.plusconv(torch.cat(xs, 1))
        if self.final_batchnorm:
            out = self.activation(self.finalbathcnorm(x + x_skip))
        else:
            out = self.activation(x + x_skip)
        return out


class ResidualModule(nn.Module):
    def __init__(self, 
        inp_channels: int, 
        out_channels: int, 
        intermediate_channels=64, 
        final_batchnorm=False,
        activation_function=nn.LeakyReLU(), 
        kernel_size = 3):
        """
        inp_channels: input channels
        out_channels: output channels
        intermediate_channels: hidden channels
        """
        super(ResidualModule, self).__init__()
        self.conv1 = MyConv(inp_channels, intermediate_channels, 
            kernel_size=1, 
            padding=0, 
            batch_norm=True)
        self.conv2 = MyConv(intermediate_channels, intermediate_channels, 
            kernel_size = kernel_size, 
            padding='same', 
            batch_norm=True)
        self.conv3 = MyConv(intermediate_channels, out_channels, 
            kernel_size=1, 
            padding=0, 
            batch_norm=True)
        self.skip_conv = MyConv(inp_channels, out_channels, 
            kernel_size=1, 
            padding=0, 
            batch_norm=True)
        if final_batchnorm:
            self.fbn = nn.BatchNorm2d(out_channels)
        else:
            self.fbn = None
        self.act = activation_function

    def forward(self, x):
        x_skip = self.skip_conv(x)
        x = self.act(self.conv1(x))
        x = self.act(self.conv2(x))
        x = self.act(self.conv3(x))
        if self.fbn is not None:
            return self.act(self.fbn(x + x_skip))
        else:
            return self.act(x + x_skip)


class Hourglass(nn.Module):
    def __init__(self, inp_channels, 
        out_channels, 
        inside_channels=96, 
        intermediate_channels=64, 
        downsample_depth=3, up_scale = False,
        back_bone = 'ResNet'):
        """
        inp_channels:
        out_channels:
        inside_channels: channels to use inside hourglass
        intermediate_channels: channels to use in residual blocks
        """
        super(Hourglass, self).__init__()
        self.input_channels = inp_channels
        self.output_channels = out_channels
        self.depth = downsample_depth
        self.maxpool = nn.MaxPool2d(2, 2)
        # self.upsample = nn.Upsample(scale_factor=2, mode='nearest')
        # self.upsample = nn.Upsample(scale_factor=2, mode='bicubic')
        self.upsample = nn.PixelShuffle(upscale_factor=2)
        self.dropout = nn.Dropout(p=0.2)
        self.activation_siren = Siren()
        self.up_scale = up_scale
        self.back_bone = back_bone
        self.shortcutconvs = nn.ModuleList([
            MyConv(inp_channels=inside_channels if i > 0 else inp_channels,
                   out_channels=inside_channels, 
                   kernel_size=1, 
                   padding=0)
            for i in range(self.depth)
        ])

        if self.back_bone == "ResNet":
            self.downwardconvs = nn.ModuleList([
                ResidualModule(
                    inp_channels=inside_channels if i > 0 else inp_channels,
                    out_channels=inside_channels,
                    intermediate_channels=intermediate_channels,
                    final_batchnorm = False)
                for i in range(self.depth)
            ])
        elif self.back_bone == "ResNext":
            self.downwardconvs = nn.ModuleList([
                ResNextModule(
                    inp_channels=inside_channels if i > 0 else inp_channels,
                    out_channels=inside_channels,
                    intermediate_channels=intermediate_channels,
                    final_batchnorm = False,
                    cardinality = True,
                    as_back_bone = True)
                for i in range(self.depth)
            ])
        else:
            raise ValueError("invalid back bone mode %s"%self.back_bone)

        self.downsizeconvs = nn.ModuleList([
            nn.Conv2d(
                in_channels=inside_channels, 
                out_channels=inside_channels, 
                kernel_size=2, 
                stride=2)
            for i in range(self.depth)
        ])

        self.lowestconvs = nn.Sequential(
            ResidualModule(inside_channels, 2*inside_channels, intermediate_channels, 
                final_batchnorm=True,
                activation_function=self.activation_siren),
            self.dropout,
            ResidualModule(2*inside_channels, 2*inside_channels, intermediate_channels, 
                final_batchnorm=True,
                activation_function=self.activation_siren),
            self.dropout,
            ResidualModule(2*inside_channels, inside_channels, intermediate_channels, 
                final_batchnorm=True,
                activation_function=self.activation_siren),
            self.dropout
        )

        self.upsizeconvs = nn.ModuleList([
            nn.Sequential(
                self.upsample,
                MyConv(inp_channels=inside_channels // 4,
                       out_channels=inside_channels, kernel_size=3, padding=1, 
                       )
            )
            for i in range(self.depth)
        ])

        if self.up_scale:
            if self.back_bone == "ResNet":
                self.upwardconvs = nn.ModuleList([
                    ResidualModule(
                        inp_channels=inside_channels, 
                        out_channels=inside_channels,
                        intermediate_channels=intermediate_channels, 
                        final_batchnorm=True)
                    for i in range(self.depth)
                ])
            elif self.back_bone == "ResNext":
                self.upwardconvs = nn.ModuleList([
                    ResNextModule(
                        inp_channels=inside_channels, 
                        out_channels=inside_channels,
                        intermediate_channels=intermediate_channels, 
                        final_batchnorm=True,
                        cardinality = True,
                        as_back_bone = True)
                    for i in range(self.depth)
                ])
            else:
                raise ValueError("invalid back bone mode %s"%self.back_bone)
        else:
            if self.back_bone == "ResNet":
                self.upwardconvs = nn.ModuleList([
                    ResidualModule(
                        inp_channels=inside_channels,
                        out_channels=inside_channels if i < (self.depth - 1) else out_channels,
                        intermediate_channels=intermediate_channels, 
                        final_batchnorm=True)
                    for i in range(self.depth)
                ])
            elif self.back_bone == "ResNext":
                self.upwardconvs = nn.ModuleList([
                    ResNextModule(
                        inp_channels=inside_channels,
                        out_channels=inside_channels if i < (self.depth - 1) else out_channels,
                        intermediate_channels=intermediate_channels, 
                        final_batchnorm=True,
                        cardinality = True,
                        as_back_bone = True)
                    for i in range(self.depth)
                ])
            else:
                raise ValueError("invalid back bone mode %s"%self.back_bone)

        self.domain_conversion = nn.Sequential(
            torch.nn.Conv2d(
                in_channels=out_channels, 
                out_channels=out_channels, 
                kernel_size=2, 
                padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU()
            )

        self.outsideheadconvs = nn.Sequential(
                MyConv(
                    inp_channels=out_channels,
                    out_channels=1, 
                    kernel_size=3, 
                    padding=1, 
                    batch_norm=False),
                nn.Tanh())

        self.plusconv = nn.ModuleList([
            nn.Sequential(
                MyConv(
                    inp_channels=2*inside_channels, 
                    out_channels=inside_channels, 
                    kernel_size=3, 
                    padding=1),
                nn.LeakyReLU()
            )
            for i in range(self.depth)
        ])

        if self.up_scale:
            if self.back_bone == "ResNet":
                self.convs2x = nn.Sequential( 
                    self.upsample,
                    MyConv(
                        inp_channels=inside_channels // 4, 
                        out_channels=inside_channels, 
                        kernel_size=3, 
                        padding=1),
                    ResidualModule(
                        inp_channels=inside_channels, 
                        out_channels= out_channels, 
                        intermediate_channels=intermediate_channels, 
                        final_batchnorm=True)
                )
            elif self.back_bone == "ResNext":
               self.convs2x = nn.Sequential( 
                    self.upsample,
                    MyConv(
                        inp_channels=inside_channels // 4, 
                        out_channels=inside_channels, 
                        kernel_size=3, 
                        padding=1),
                    ResNextModule(
                        inp_channels=inside_channels, 
                        out_channels= out_channels, 
                        intermediate_channels=intermediate_channels, 
                        final_batchnorm=True,
                        cardinality = True,
                        as_back_bone = True)
                )

    def forward_with_downscale_outs(self, x):
        """
        Return fat x (batch_size, self.out_channels, original_H, original_W), and (x4, x2, x1 downscaled single channel heatmaps)
        """
        assert (x.shape[1] == self.input_channels)
        x_shortcuts = []
        outsideheads = []
        # encoder
        for i in range(self.depth):
            # get skip connection features 
            x_shortcuts.append(self.shortcutconvs[i](x))
            x = self.downwardconvs[i](x)
            x = self.downsizeconvs[i](x)
        # bottom neck
        x = self.lowestconvs(x)
        # decoder
        for i in range(self.depth):
            x = self.upsizeconvs[i](x)
            x = self.plusconv[i](torch.cat((x, x_shortcuts[-(i + 1)]), 1))
            x = self.upwardconvs[i](x)
        
        if self.up_scale:
            # upscale
            x_up = self.convs2x(x)
            # convert to UDF domain
            x_udf = self.domain_conversion(x_up)
            return x_up, x_udf
        else:
            x_udf = self.domain_conversion(x)
            # x_udf = self.outsideheadconvs.forward(x)
            return x, x_udf


    def forward(self, x):
        return self.forward_with_downscale_outs(x)[0]


class BaseHeatmapModel(nn.Module):
    def __init__(self, can_predict=False, can_classify=False):
        super(BaseHeatmapModel, self).__init__()
        self.can_predict = can_predict
        self.can_classify = can_classify

    def forward_point_heatmap(self, x):
        """
        Return a heatmap of predicted points
        """
        raise NotImplementedError

    def forward_class_heatmaps(self, x):
        """Return heatmap of predicted points AND (3 heatmaps of classes)"""
        raise NotImplementedError

    def forward_class_downscale_heatmaps(self, x):
        """Return heatmap of predicted points AND (heatmaps when downscaled) AND (3 heatmaps of classes)"""
        raise NotImplementedError

    def forward(self, x):
        return self.forward_point_heatmap(x)

    def freeze_batchnorm(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()
                m.weight.requires_grad = False
                m.bias.requires_grad = False
            if isinstance(m, nn.Dropout):
                m.eval()


class CenterNet(BaseHeatmapModel):
    def __init__(self, n_blocks=1):
        super(CenterNet, self).__init__(can_predict=True, can_classify=False)
        self.n_blocks = n_blocks
        self.hourglass_output_channels = 128
        self.hourglasses = nn.ModuleList([
            Hourglass(
                inp_channels=self.hourglass_output_channels if i > 0 else 1,
                out_channels=self.hourglass_output_channels,
                inside_channels=32,
                intermediate_channels=32
            )
            for i in range(n_blocks)
        ])
        self.head_outs = nn.ModuleList([
            MyConv(inp_channels=self.hourglass_output_channels,
                   out_channels=1,
                   kernel_size=1, padding=0)
            for i in range(n_blocks)
        ])
        self.head_backs = nn.ModuleList([
            MyConv(inp_channels=1,
                   out_channels=self.hourglass_output_channels,
                   kernel_size=1, padding=0)
            for i in range(n_blocks - 1)
        ])
        self.interblocks = nn.ModuleList([
            ResidualModule(inp_channels=self.hourglass_output_channels,
                           out_channels=self.hourglass_output_channels,
                           intermediate_channels=64,
                           activation_function=nn.LeakyReLU())
            for i in range(n_blocks - 1)
        ])

    def forward_point_heatmap(self, x):
        heads = []
        for i in range(self.n_blocks):
            x = self.hourglasses[i](x)
            head = self.head_outs[i](x)
            heads.append(head)
            if i < (self.n_blocks - 1):
                # if it is not the last head
                x = x + self.head_backs[i](head)
                x = self.interblocks[i](x)
        return heads[-1]


class CenterNet_classification(CenterNet):
    def __init__(self, n_blocks=1):
        super(CenterNet_classification, self).__init__(n_blocks=n_blocks)
        print("\n\n THIS IS AN EXPERIMENTAL MODULE WITH CLASSIFICATION HEADS< CHANGE IT LATER \n\n")

        self.classification_heads = nn.ModuleList([
            # MyConv(inp_channels=self.hourglass_output_channels,
            #        out_channels=1,
            #        kernel_size=1, padding=0)
            nn.Sequential(
                ResidualModule(inp_channels=self.hourglass_output_channels,
                               out_channels=1,
                               intermediate_channels=32,
                               activation_function=nn.ReLU()),
                nn.Tanh()
            )
            for i in range(3)
        ])

    def forward_class_heatmaps(self, x):
        heads = []
        for i in range(self.n_blocks):
            x = self.hourglasses[i](x)
            head = self.head_outs[i](x)
            heads.append(head)
            if i < (self.n_blocks - 1):
                # if it is not the last head
                x = x + self.head_backs[i](head)
                x = self.interblocks[i](x)
        classes = [head(x) for head in self.classification_heads]
        return heads[-1], classes


class ForwardNet(nn.Module):
    def __init__(self, n_blocks=1):
        super(ForwardNet, self).__init__()
        self.n_blocks = n_blocks
        self.block = nn.Sequential(
            ResNextModule(inp_channels=1, out_channels=128, intermediate_channels=64),
            ResNextModule(inp_channels=128, out_channels=256, intermediate_channels=128),
            MyConv(inp_channels=256, out_channels=3,kernel_size=1,padding=0,batch_norm=True),
            nn.ReLU()
        )

    def forward(self, x, return_all=False):
        out = self.block(x)
        if return_all:
            return out[:, 0, :, :].unsqueeze(1), out[:, 1, :, :].unsqueeze(1), out[:, 2, :, :].unsqueeze(1)
        else:
            return torch.sum(out, dim=1).unsqueeze(1)


def siren(input):
    """
    return torch.sin(x) as in https://vsitzmann.github.io/siren/
    """
    return torch.sin(input)


class Siren(nn.Module):
    """
    applies siren activation. Wrapper for pytorch
    """

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return siren(x)


class PyramidalNet(BaseHeatmapModel):
    def __init__(self, n_classes, 
        up_scale = False,
        hourglass_input_channels = 64, 
        hourglass_channels = 96, 
        downsample_depth = 3,
        cardinality = False,
        back_bone = "ResNet",
        width = 8):
        super(PyramidalNet, self).__init__(can_predict=True, can_classify=True)
        self.hourglass_input_channels = hourglass_input_channels
        self.up_scale = up_scale
        self.n_classes = n_classes
        self.hourglass_channels = hourglass_channels
        self.cardinality = cardinality
        self.width = width 
        self.inputconv = ResNextModule(inp_channels = 1, 
            out_channels=self.hourglass_input_channels,
            intermediate_channels = 16,
            cardinality = cardinality,
            width = self.width
            )
        
        # backbone network
        self.hourglass = Hourglass(
            inp_channels=self.hourglass_input_channels,
            out_channels=hourglass_channels,
            inside_channels=hourglass_channels,
            intermediate_channels=hourglass_channels,
            up_scale = up_scale,
            downsample_depth = downsample_depth,
            back_bone = back_bone
        )
        
        # we use leaky relu for better prediction topology result
        self.classheads = nn.ModuleList([
            ResNextModule(
                inp_channels=hourglass_channels, 
                out_channels=1, 
                intermediate_channels = 16,
                cardinality = cardinality if i == 0 else False,
                activation_function = nn.LeakyReLU(),
                width = self.width
                )
            for i in range(self.n_classes)
        ])

    def forward_with_downscale_outs(self, x):
        # input
        x = nn.LeakyReLU()(self.inputconv(x))
        
        # backbone
        x_ras, x_udf = self.hourglass.forward_with_downscale_outs(x)

        # predict udf of keypoint and topology
        x_udf = self.classheads[0](x_udf)
        classes = [self.classheads[i](x_ras) for i in range(1, self.n_classes)]
        return x_udf, classes