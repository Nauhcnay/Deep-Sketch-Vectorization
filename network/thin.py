import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from torchvision.utils import save_image
from PIL import Image


class Conv( nn.Module ):
    def __init__(self, in_planes, out_planes, stride=1, kernel_size=3, padding=1 ):
        super(Conv, self).__init__()
        self.conv    = nn.Conv2d( in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding )
    def forward(self, x):
        return F.relu( self.conv( x ), inplace=True )

class ThinningNet( nn.Module ):
    def __init__(self):
        super(ThinningNet, self).__init__()
        self.layers = nn.Sequential(
            nn.ReplicationPad2d(4),
            Conv(  1, 64, 1, 9, 0 ),
            Conv( 64, 64 ),
            Conv( 64, 64 ),
            Conv( 64, 64 ),
            Conv( 64, 64 ),
            Conv( 64, 64 ),
            Conv( 64, 64 ),
            Conv( 64, 64 ),
            nn.Conv2d( 64, 1, kernel_size=3, stride=1, padding=1 )
        )
    def forward(self, x):
        return torch.sigmoid( self.layers( x-0.7 ) )

class Thinner:
    def __init__(self, weightfile='./pretrained/line_thin.pth.tar', use_gpu=False):
        self.use_gpu = use_gpu
        self.model = ThinningNet()
        map_location = None if self.use_gpu else lambda storage, loc: storage
        self.model.load_state_dict( torch.load(weightfile, map_location=map_location) )
        if self.use_gpu:
            self.model.cuda()
        self.model.eval()

    def __call__(self, data ):
        if isinstance(data, str): # string
            data = Image.open(data).convert('L')
        if isinstance(data, Image.Image): # Image
            data = transforms.ToTensor()(data).unsqueeze(0)
        if isinstance(data, np.ndarray): # numpy array
            data = transforms.ToTensor()(Image.fromarray(data).convert('L')).unsqueeze(0)
        if data.dim() < 4:
            data = data.unsqueeze(0)

        if self.use_gpu:
            data = data.cuda()
        with torch.no_grad():
            return self.model.forward( data )[0].data.float()

if __name__ == "__main__":
    infile = sys.argv[1] if  len(sys.argv)>1 else 'in.png'
    outfile = sys.argv[2] if  len(sys.argv)>2 else 'out.png'
    thin = Thinner()
    save_image( thin( infile ), outfile )

