# Pre Trained Networks 
import torch 
from torchvision import models
#print(dir(models))
alexnet = models.AlexNet()
#print(type(alexnet)) #<class 'torchvision.models.alexnet.AlexNet'>
resnet = models.resnet101(pretrained=True)
#print(type(resnet)) #<class 'torchvision.models.resnet.ResNet'>
#print(resnet)
## FOOBAR_TBD - Give all - transforms - in ARGPARSE - 

from torchvision import transforms
preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )])
#print(type(preprocess))    #<class 'torchvision.transforms.transforms.Compose'>    
from PIL import Image
#img = Image.open("/img_inputs/box.png")
#print(type(img)) #<class 'PIL.PngImagePlugin.PngImageFile'>
"""
# RuntimeError: output with shape [1, 224, 224] doesn't match the broadcast shape [3, 224, 224]
# Passed in as Input a - GrayScale Image - Single Channel Image - OutPut shape is incorrect - thus got the below error ?? 
Traceback (most recent call last):
  File "1_preTrn.py", line 381, in <module>
    img_t = preprocess(img)
  File "/home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages/torchvision/transforms/transforms.py", line 61, in __call__
    img = t(img)
  File "/home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages/torchvision/transforms/transforms.py", line 212, in __call__
    return F.normalize(tensor, self.mean, self.std, self.inplace)
  File "/home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages/torchvision/transforms/functional.py", line 298, in normalize
    tensor.sub_(mean).div_(std)
RuntimeError: output with shape [1, 224, 224] doesn't match the broadcast shape [3, 224, 224]
"""
#spacenet
# img = Image.open("/img_inputs/spacenet.png")
# img_t = preprocess(img)
# print(type(img_t)) 
"""
Traceback (most recent call last):
  File "1_preTrn.py", line 396, in <module>
    img_t = preprocess(img)
  File "/home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages/torchvision/transforms/transforms.py", line 61, in __call__
    img = t(img)
  File "/home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages/torchvision/transforms/transforms.py", line 212, in __call__
    return F.normalize(tensor, self.mean, self.std, self.inplace)
  File "/home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages/torchvision/transforms/functional.py", line 298, in normalize
    tensor.sub_(mean).div_(std)
RuntimeError: The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 0
"""
#spacenet again 
# test_image = Image.open(test_image_name).convert('RGB')
# SOURCE SO -- https://stackoverflow.com/questions/58496858/pytorch-runtimeerror-the-size-of-tensor-a-4-must-match-the-size-of-tensor-b

#img = Image.open("spacenet.png").convert('RGB')
img = Image.open("../img_inputs/dog.jpg").convert('RGB')
img_t = preprocess(img)
#print(type(img_t)) #<class 'torch.Tensor'>
#print(img_t.shape) #torch.Size([3, 224, 224])
import matplotlib.pyplot as plt
import numpy as np
import PIL
#Show image after - preprocess(Transforms that we have done - transforms.Compose)
plt.imshow(np.transpose(img_t,(1, 2, 0))) # after transpose shape == 224,224,3 ( Length , Breath , Channels)
plt.show()
#
batch_t = torch.unsqueeze(img_t, 0)
#print(type(batch_t))##<class 'torch.Tensor'>
resnet.eval()
#print(resnet.eval()) # Same as the Terminal Print for --- #print(resnet) - done ABOVE
out = resnet(batch_t)
#print(out)
with open('../img_inputs/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]
    #print(len(labels)) # 1000
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
print(labels[index[0]], percentage[index[0]].item()) 
_, indices = torch.sort(out, descending=True)
print([(labels[idx], percentage[idx].item()) for idx in indices[0][:5]])
