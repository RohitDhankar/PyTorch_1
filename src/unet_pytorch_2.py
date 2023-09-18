# conda activate env2_det2
import torch
import torch.nn as nn

from PIL import Image
from torchvision import transforms
from einops import rearrange



def double_conv(in_c,out_c):
    """
    https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
    """
    conv = nn.Sequential(
        nn.Conv2d(in_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_c,out_c,kernel_size=3),
        nn.ReLU(inplace=True),
    )
    return conv


class unet(nn.Module):
    """
    https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md


    As a consequence,
    the expansive path is more or less symmetric to the contracting path, and yields
    a u-shaped architecture.

    It consists of a contracting path (left side) -- CONTRACTION -- 
    and an expansive path (right side). -- EXPANSION --

    The contracting path follows the typical architecture of a convolutional network. 
    THE CONTRACTION OF THE CONTRACTING PATH IS BEING USED FOR -- DOWNSAMPLING --

    It consists of the repeated application of two 3x3 convolutions (unpadded convolutions)
    , each followed by a rectified linear unit (ReLU) 
    and 
    a 2x2 max pooling operation with stride 2 for downsampling.

    The network does not have any fully connected layers
    and 
    only uses the valid part of each convolution, i.e., the segmentation map only

    contains the pixels, for which the full context is available in the input image.

    """
    def __init__(self) -> None:
        super(unet,self).__init__()
        
        #self.max_pool_2x2 = nn.MaxPool1d(kernel_size=2,stride=2)
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2,stride=2)
        self.down_conv_1 = double_conv(1,64)       #in_c =1  out_c =64
        self.down_conv_2 = double_conv(64,128)     #in_c =64  out_c =128
        self.down_conv_3 = double_conv(128,256)     #in_c =128  out_c =256
        self.down_conv_4 = double_conv(256,512)
        self.down_conv_5 = double_conv(512,1024)

    def forward(self,image):
        """
        """
        x1 = self.down_conv_1(image)
        print("-x1.size()--",x1.size()) #-x1.size()-- torch.Size([1, 64, 568, 1126])
        print("-x1.shape--",x1.shape) #y1=x1.shape ## <class 'torch.Size'> #print(type(y1)) ## <class 'torch.Size'>
        x2 = self.max_pool_2x2(x1)
        print("--type(x2)--",type(x2))
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        print("--type(x4)--",type(x4)) #print("-x4.size()--",x4.size())
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        print("-x9.size()--",x9.size())

        # transforms.functional.to_pil_image(x1_t) # Needs as INPUT a -- Tensor of shape C x H x W

     

obj_unet = unet()
print("--type(obj_unet)--",type(obj_unet))
image_in = torch.rand((1,1,572,572))
#(num_samples, channels, height, width)
##(ns, channels, height, width)

# image_in = Image.open("./img_inputs/dog.jpg")
# preprocess = transforms.Compose([transforms.Resize(572),
#                                 transforms.Grayscale(num_output_channels=1),
#                                 transforms.ToTensor()])
# image_in = preprocess(image_in)
# batch_images_gray = torch.unsqueeze(image_in, 1) # Batch of NUM_SAMPLES = 1 , which is a BATCH of 1 GRAY SCALE IMAGE Only 
#batch_images_rgb = torch.unsqueeze(image_in, 0) # Batch of NUM_SAMPLES = 1 , which is a BATCH of 1 RGB - 3 Channel IMAGE Only 


obj_unet.forward(image_in) # OK 
"""
--type(obj_unet)-- <class '__main__.unet'>
-x1.size()-- torch.Size([1, 64, 568, 568])
-x1.shape-- torch.Size([1, 64, 568, 568])
--type(x2)-- <class 'torch.Tensor'>
--type(x4)-- <class 'torch.Tensor'>
-x9.size()-- torch.Size([1, 1024, 28, 28])
"""

