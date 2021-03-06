
##### Check all systems GO !!!

#
> Visualise PyTorch tensors and layers etc 
- Visdom - Visline -- https://github.com/facebookresearch/visdom#visline
- torch.utils.tensorboard -- TensorBoard from torch >> utils 

```
Another option was to use Crayon and TensorBoard , now Crayon is Archived - https://github.com/torrvision/crayon
Crayon is used here on Kaggle - https://www.kaggle.com/solomonk/pytorch-senet-augmentation-cnn-lb-0-956
```

#
> Deploy PyTorch models with Flask - 
- pytorch-flask-api - https://github.com/avinassh/pytorch-flask-api
- pytorch-flask-api - https://pytorch.org/tutorials/intermediate/flask_rest_api_tutorial.html

#

```
#PyTorch --- BatchNorm2d
# Source == https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/02-intermediate/convolutional_neural_network/main.py#L35-L56

# Convolutional neural network (two convolutional layers)

class ConvNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.fc = nn.Linear(7*7*32, num_classes)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        return out


```
```
(pytorch_venv) dhankar@dhankar-1:~/temp/pytorch/PyTorch_1$ python
Python 3.8.5 (default, Aug  5 2020, 08:36:46) 
[GCC 7.3.0] :: Anaconda, Inc. on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> 
>>> from __future__ import print_function
>>> import torch
>>> x = torch.rand(5, 3)
>>> print(x)
tensor([[0.1965, 0.8001, 0.6781],
        [0.9286, 0.7364, 0.2602],
        [0.1383, 0.6738, 0.3003],
        [0.6685, 0.4664, 0.9403],
        [0.6227, 0.5124, 0.4035]])
>>> 
>>> torch.cuda.is_available()
True
>>> 
```

#

- IMAGE_NET http://imagenet.stanford.edu
- https://github.com/pytorch/vision
- AlexNet - http://mng.bz/lo6z
- ReLU was introduced in this paper by Vinod Nair and Geoffrey E. Hinton in 2010.		https://towardsdatascience.com/what-alexnet-brought-to-the-world-of-deep-learning-46c7974b46fc
- ResNET - Residual Network -- https://arxiv.org/pdf/1512.03385.pdf


#

- pip install torchvision - https://github.com/pytorch/vision

#

- test code dump below - to check the INDEX and VALUE from - ```tensor.max ```

```
tensor([235])
tensor(235)
---AA    ------AA    ------AA    ------AA    ------AA    ------AA    ------AA    ------AA    ------AA    ------AA    ---
tensor([14.5138], grad_fn=<MaxBackward0>) 1.45
--- BB   ------ BB   ------ BB   ------ BB   ------ BB   ------ BB   ------ BB   ------ BB   ------ BB   ------ BB   ---
tensor([3.3339e-05, 9.8777e-06, 2.3446e-05, 2.8386e-05, 6.2726e-06, 2.8400e-05,
        1.6072e-05, 2.9150e-05, 3.8224e-05, 8.9633e-05, 5.0123e-05, 3.0741e-05,
        1.5401e-05, 1.7589e-05, 4.2333e-05, 1.8972e-05, 4.1111e-05, 7.4038e-05,
        4.3250e-05, 2.8442e-05, 1.2220e-05, 7.8632e-05, 2.6467e-06, 5.4897e-05,
        2.1041e-05, 3.3092e-05, 1.9825e-05, 6.1416e-06, 1.0530e-05, 2.6415e-05,
        4.4315e-05, 8.4517e-06, 4.2191e-05, 3.3758e-05, 5.3347e-05, 3.7769e-05,
        2.4521e-05, 1.6589e-05, 3.7809e-05, 6.8475e-06, 3.1787e-05, 2.1667e-05,
        4.5116e-05, 1.2443e-05, 1.7249e-05, 4.3923e-05, 2.0429e-05, 3.2976e-05,
        1.5958e-06, 3.1632e-05, 4.9902e-05, 9.3217e-06, 1.1327e-04, 4.2638e-05,
        1.0383e-04, 2.1193e-05, 3.8393e-06, 5.2748e-05, 9.5103e-05, 5.6516e-05,
        1.2494e-04, 2.8213e-05, 4.3813e-05, 2.4431e-04, 4.9238e-05, 4.0367e-05,
        1.5146e-04, 5.1143e-05, 4.7644e-05, 1.2593e-05, 1.1949e-04, 1.2194e-04,
        6.3299e-05, 2.5208e-05, 3.8676e-05, 3.1843e-05, 5.6887e-06, 1.6821e-04,
        2.3283e-04, 5.7986e-05, 2.1477e-05, 1.8664e-06, 4.4332e-05, 2.3346e-05,
        5.2749e-06, 2.4496e-05, 4.3985e-05, 1.2598e-05, 6.1047e-06, 7.3643e-06,
        4.4431e-06, 1.0101e-05, 5.0529e-05, 6.3905e-05, 1.3098e-04, 4.1077e-05,
        4.0573e-06, 1.5725e-05, 5.1769e-05, 3.0567e-05, 3.9503e-05, 1.0126e-05,
        3.5554e-06, 3.7522e-05, 1.6544e-03, 3.0003e-06, 4.4201e-05, 2.3701e-06,
        1.0704e-05, 2.9969e-06, 7.0037e-06, 4.3253e-05, 4.4913e-05, 1.6420e-04,
        5.8238e-05, 1.0941e-05, 2.1399e-05, 1.7048e-05, 9.1376e-05, 2.7425e-05,
        4.3731e-05, 3.8931e-05, 9.1568e-05, 6.4276e-05, 1.3447e-04, 7.8212e-06,
        5.3207e-05, 7.2779e-05, 3.8062e-05, 1.7411e-05, 1.6273e-05, 6.8985e-06,
        8.5470e-06, 6.2593e-05, 3.8879e-05, 2.5616e-05, 1.8904e-04, 3.0485e-05,
        8.4644e-06, 4.6330e-05, 7.5660e-06, 1.0165e-05, 7.6003e-05, 3.0339e-05,
        2.9986e-06, 3.8774e-05, 5.8029e-06, 4.6663e-06, 9.0647e-06, 3.4799e-06,
        1.8177e-05, 6.5090e-04, 1.3751e-04, 1.2215e-05, 1.2971e-04, 1.0434e-05,
        4.3846e-05, 1.6106e-04, 3.0728e-04, 2.8820e-03, 1.5321e-03, 1.2331e-04,
        5.7625e-04, 2.9784e-02, 5.0755e-04, 5.0705e-04, 1.0715e-03, 4.0888e-03,
        3.8435e-04, 9.4581e-04, 2.5254e-03, 2.9067e-05, 2.9810e-04, 8.8327e-04,
        1.5733e-01, 1.5549e-02, 2.2065e-03, 3.7334e-04, 1.8121e-05, 3.3070e-04,
        3.9701e-04, 3.2956e-04, 5.8416e-03, 1.6208e-04, 3.6633e-04, 2.1436e-04,
        2.8176e-03, 1.2819e-04, 2.2686e-03, 2.0298e-03, 1.1688e-05, 6.0546e-03,
        6.7402e-04, 2.4289e-03, 4.7002e-04, 1.0873e-04, 5.4439e-05, 5.6521e-04,
        5.0551e-04, 1.8307e-04, 1.0648e-03, 3.5187e-04, 2.6062e-04, 8.9395e-06,
        9.9409e-05, 2.0042e-04, 1.8155e-04, 4.7866e-04, 1.5926e-04, 2.9703e-04,
        5.0780e-05, 7.6062e-05, 7.3967e-05, 3.3393e-04, 3.0230e-04, 1.2477e-04,
        1.2700e-04, 1.7346e-05, 1.8226e-04, 4.8264e-05, 5.6229e-04, 7.3212e-05,
        2.4785e-03, 4.3330e-04, 3.4302e-03, 4.5036e-01, 5.9631e-03, 1.1292e-02,
        1.1056e-04, 7.1410e-04, 4.1876e-04, 4.6532e-03, 8.9051e-04, 1.3287e-03,
        2.4341e-03, 9.8724e+01, 1.1021e-03, 1.4729e-04, 6.9063e-04, 2.9947e-04,
        6.1171e-03, 7.5160e-04, 3.3576e-04, 3.3892e-03, 4.6243e-02, 4.7934e-05,
        1.6992e-03, 9.1422e-03, 4.2588e-02, 4.5416e-02, 7.8917e-03, 5.4631e-05,
        3.3930e-05, 1.4386e-03, 1.5205e-04, 1.0772e-01, 1.6452e-03, 5.2194e-04,
        8.4847e-05, 1.1165e-04, 2.0603e-03, 6.3724e-03, 1.3238e-03, 2.2047e-03,
        1.0138e-03, 3.2611e-05, 2.4845e-05, 5.6622e-04, 1.4485e-04, 1.2353e-02,
        2.6048e-03, 9.3872e-03, 7.4614e-03, 2.0258e-02, 1.0224e-02, 5.1307e-03,
        1.9804e-04, 5.2269e-04, 6.1564e-04, 2.1163e-05, 2.4657e-03, 1.3679e-04,
        2.6887e-04, 5.8013e-06, 1.2398e-04, 6.6895e-05, 5.6655e-05, 2.2754e-05,
        4.3498e-05, 1.2471e-05, 2.9185e-05, 1.9011e-04, 3.5083e-04, 1.6900e-04,
        6.0068e-05, 4.0041e-05, 1.8066e-05, 2.8186e-06, 2.5840e-06, 9.8202e-06,
        1.2361e-04, 3.5377e-05, 6.3952e-05, 1.1829e-04, 1.7815e-04, 8.6781e-05,
        2.6079e-04, 1.2947e-04, 6.0337e-04, 6.4094e-04, 8.5231e-05, 4.9287e-04,
        2.7382e-04, 2.6634e-05, 1.2696e-04, 1.0888e-05, 3.5975e-05, 2.5875e-04,
        8.3652e-05, 1.3888e-04, 6.9052e-05, 4.7432e-05, 1.2583e-05, 1.0198e-04,
        1.0094e-05, 3.0178e-05, 2.7587e-05, 1.5054e-04, 2.0840e-05, 1.7399e-05,
        1.4947e-04, 7.6343e-04, 1.5611e-05, 1.2308e-05, 3.9126e-06, 2.1870e-05,
        3.1209e-05, 3.5006e-05, 8.4186e-06, 1.8638e-04, 1.0642e-04, 1.4572e-04,
        3.6594e-05, 8.1172e-06, 1.4724e-05, 2.3915e-05, 4.7114e-05, 2.5184e-05,
        3.3406e-05, 6.7441e-06, 1.9604e-05, 6.0466e-05, 1.2289e-04, 1.6752e-04,
        4.5121e-05, 5.2385e-04, 1.5415e-04, 2.9162e-05, 8.2093e-05, 5.3538e-05,
        6.6903e-06, 1.3924e-05, 2.7320e-05, 4.7038e-05, 1.0607e-05, 4.9825e-06,
        7.4373e-06, 8.1058e-06, 1.0891e-04, 8.2017e-06, 1.8254e-05, 1.0285e-04,
        6.9439e-05, 2.7038e-05, 3.3748e-06, 9.8613e-06, 1.6701e-05, 3.2721e-06,
        2.3841e-05, 5.0641e-05, 1.3319e-05, 4.5530e-05, 4.1902e-05, 1.2085e-05,
        2.1755e-04, 1.3216e-05, 6.3602e-06, 6.0928e-06, 1.2503e-05, 1.3191e-04,
        2.1279e-05, 1.9647e-05, 2.2516e-05, 4.3899e-07, 1.3171e-04, 1.6324e-04,
        3.0675e-06, 4.7915e-06, 9.2250e-05, 1.5184e-05, 8.0771e-05, 1.1283e-04,
        1.7550e-05, 5.6642e-06, 7.5582e-06, 9.8534e-05, 4.4275e-06, 4.4277e-05,
        7.9708e-05, 3.0040e-05, 1.1217e-05, 6.0957e-05, 9.8542e-05, 1.2480e-04,
        8.4187e-06, 1.0586e-05, 8.1039e-05, 1.1928e-04, 5.4042e-05, 1.5915e-05,
        2.7433e-05, 4.5948e-05, 4.2754e-05, 9.2269e-06, 3.4922e-06, 2.6860e-05,
        2.8284e-05, 6.1732e-05, 7.7024e-05, 6.9621e-04, 4.7889e-05, 1.3115e-05,
        1.6764e-05, 1.3293e-04, 5.7041e-05, 9.4793e-06, 4.6091e-05, 2.3729e-05,
        1.7623e-05, 3.7847e-04, 4.4180e-05, 5.8786e-05, 1.7463e-04, 2.2884e-05,
        1.0554e-05, 1.2176e-04, 2.0664e-05, 4.7061e-05, 7.1893e-05, 2.0160e-05,
        2.2582e-05, 2.1051e-05, 9.0809e-06, 1.2339e-05, 1.2287e-05, 4.6955e-05,
        3.8913e-05, 5.3015e-05, 3.3360e-05, 3.5862e-05, 3.7735e-06, 1.9829e-04,
        5.5983e-05, 1.7068e-04, 1.5385e-04, 2.1279e-02, 1.1492e-05, 3.7164e-06,
        1.1374e-05, 9.5877e-05, 7.2761e-05, 2.5169e-04, 2.3681e-04, 8.0329e-05,
        4.3522e-06, 4.6151e-05, 7.9290e-06, 1.6055e-05, 2.0245e-05, 2.3791e-04,
        4.4651e-05, 9.5189e-06, 3.1780e-05, 1.1137e-04, 1.9181e-05, 2.2683e-05,
        6.6639e-06, 1.8849e-04, 5.6000e-05, 1.0371e-05, 1.1886e-04, 2.6325e-05,
        2.7240e-05, 7.4988e-06, 8.6941e-05, 8.0532e-06, 6.6309e-06, 2.1762e-05,
        7.1956e-06, 2.2541e-05, 4.2465e-06, 2.2434e-05, 1.5366e-04, 1.0738e-04,
        1.0579e-04, 3.1874e-05, 3.5550e-05, 1.9572e-05, 2.0895e-04, 1.0944e-05,
        1.3637e-05, 4.0983e-05, 2.4452e-05, 1.3318e-04, 7.3651e-05, 4.2405e-04,
        5.2547e-06, 7.1330e-05, 6.0269e-05, 1.9934e-05, 6.5175e-06, 7.6619e-06,
        5.5286e-04, 7.3997e-05, 2.2059e-04, 7.5321e-06, 5.2431e-05, 8.1221e-05,
        4.7385e-05, 1.2463e-05, 4.3568e-05, 1.2584e-04, 1.8874e-05, 1.7969e-05,
        8.6766e-06, 8.3859e-06, 4.7602e-06, 1.1502e-03, 2.7097e-05, 2.0366e-04,
        6.3208e-06, 6.5717e-05, 6.7592e-05, 4.9435e-05, 1.0404e-04, 2.8758e-05,
        4.4122e-05, 1.3708e-05, 6.4154e-05, 1.9488e-05, 4.2002e-06, 6.3798e-05,
        3.2077e-05, 3.6022e-06, 1.7399e-05, 2.5438e-05, 4.6560e-05, 3.6559e-05,
        7.6512e-06, 5.7540e-05, 4.6069e-04, 2.4527e-05, 2.0679e-05, 4.5615e-05,
        3.7822e-06, 4.7071e-06, 3.8929e-04, 6.5517e-05, 1.4651e-04, 3.8489e-05,
        3.3982e-05, 1.3501e-05, 1.6664e-04, 8.1146e-05, 2.2038e-03, 2.4172e-04,
        9.0406e-05, 4.7709e-05, 5.0574e-05, 1.0066e-04, 1.9602e-05, 7.0483e-05,
        1.0447e-05, 1.6316e-05, 1.5072e-05, 6.3170e-06, 7.4943e-04, 2.5486e-05,
        5.1579e-05, 5.2846e-05, 6.6466e-05, 2.2279e-05, 5.2012e-06, 4.3996e-05,
        1.7642e-05, 7.1580e-05, 2.4679e-05, 1.1802e-04, 9.3051e-05, 1.0015e-04,
        1.4942e-04, 2.5654e-05, 3.7257e-05, 2.2784e-04, 4.0061e-05, 8.0624e-05,
        2.4220e-05, 4.6469e-06, 1.4579e-04, 4.0211e-04, 2.6130e-05, 5.9387e-05,
        4.9547e-06, 9.8026e-05, 2.5320e-05, 6.8328e-05, 3.9220e-05, 3.4687e-05,
        2.4614e-04, 1.4886e-05, 9.7413e-05, 2.9154e-04, 2.7248e-05, 9.2734e-06,
        2.8353e-05, 8.5762e-06, 3.0135e-05, 8.4441e-05, 2.7695e-06, 5.7336e-05,
        1.9868e-04, 1.0041e-05, 3.5433e-04, 8.5093e-05, 1.0416e-05, 6.6615e-05,
        1.0650e-05, 3.0409e-05, 3.5951e-04, 1.6539e-04, 4.6785e-06, 1.5490e-04,
        1.2343e-05, 4.4204e-05, 4.6200e-05, 2.9902e-05, 1.7259e-04, 9.2519e-05,
        5.3020e-06, 2.5901e-05, 8.2692e-05, 2.1454e-05, 6.4866e-04, 3.9062e-05,
        1.1734e-05, 1.0074e-04, 4.1231e-05, 1.6521e-04, 1.7927e-05, 9.0608e-05,
        9.9555e-06, 1.3573e-04, 1.0703e-05, 1.9982e-05, 2.1288e-04, 1.1398e-05,
        1.9706e-04, 2.3310e-04, 6.2331e-06, 1.1784e-05, 1.6425e-05, 2.9270e-05,
        1.9787e-05, 2.3461e-04, 5.0374e-05, 1.1137e-05, 8.1787e-02, 2.9178e-05,
        9.4850e-05, 2.6214e-05, 5.2714e-05, 2.6754e-04, 1.4203e-05, 1.9901e-05,
        1.7002e-05, 3.5766e-06, 2.7795e-06, 1.9617e-06, 5.9035e-05, 1.1387e-05,
        1.2085e-05, 1.6527e-04, 2.2117e-05, 1.6889e-04, 4.2537e-06, 1.6245e-05,
        8.0996e-06, 5.4891e-05, 2.1426e-04, 3.3966e-05, 9.1377e-05, 2.5512e-04,
        7.9768e-05, 5.6318e-05, 2.9943e-05, 1.2845e-05, 1.7748e-05, 2.9648e-05,
        3.4907e-05, 2.1195e-06, 2.0572e-05, 8.6487e-05, 1.9974e-05, 3.8131e-05,
        4.8875e-05, 5.1514e-04, 2.1499e-05, 3.6029e-05, 1.7626e-05, 1.0424e-05,
        1.1166e-05, 7.2706e-06, 1.0334e-04, 4.0008e-05, 5.0285e-06, 1.0885e-04,
        1.0696e-05, 1.1535e-05, 1.4978e-05, 1.1509e-05, 1.5800e-04, 2.1198e-04,
        4.4507e-05, 2.3839e-04, 9.6404e-05, 2.5264e-05, 1.9499e-05, 1.6781e-05,
        1.5678e-05, 1.8714e-05, 8.6384e-06, 1.3626e-05, 3.9278e-05, 2.0287e-04,
        1.9460e-04, 5.1760e-06, 8.5761e-05, 5.4833e-05, 8.7567e-06, 1.6191e-05,
        3.0993e-05, 1.3772e-04, 2.4256e-04, 8.5175e-06, 2.7928e-05, 3.7932e-05,
        7.8000e-05, 4.0899e-05, 3.2421e-05, 4.7808e-05, 8.0923e-06, 2.8447e-04,
        1.3048e-05, 6.5914e-05, 8.1954e-05, 1.4422e-05, 1.0391e-05, 8.1075e-05,
        1.0335e-03, 5.4136e-05, 4.2321e-05, 8.9405e-06, 1.6915e-05, 5.1567e-05,
        5.2217e-05, 4.4473e-05, 8.7263e-05, 3.0890e-05, 2.1404e-05, 2.5129e-05,
        6.3937e-06, 1.9947e-04, 7.5937e-05, 2.0565e-05, 1.8886e-05, 1.9740e-05,
        6.3015e-06, 7.7127e-04, 3.6504e-05, 5.1927e-05, 2.5195e-05, 1.9707e-05,
        3.1679e-04, 1.5704e-05, 1.5454e-05, 5.1621e-05, 4.3768e-05, 2.4499e-05,
        6.6733e-05, 1.3417e-04, 5.4263e-06, 4.9442e-05, 2.1342e-05, 2.4690e-05,
        5.1504e-05, 6.2716e-03, 1.5780e-05, 1.1337e-04, 7.8688e-05, 2.2268e-04,
        2.7987e-04, 1.6141e-05, 1.0299e-04, 9.6838e-05, 2.5207e-05, 4.6710e-05,
        8.0641e-06, 1.4318e-04, 1.1582e-05, 3.0872e-05, 9.1721e-06, 1.8196e-06,
        2.8554e-05, 7.8741e-06, 3.4064e-05, 1.7997e-04, 2.9573e-05, 6.4500e-05,
        3.2376e-05, 1.3230e-05, 6.7776e-05, 6.9102e-06, 1.2662e-05, 3.2629e-05,
        1.2360e-04, 3.4487e-05, 1.6585e-04, 4.9336e-04, 1.1625e-05, 1.5898e-06,
        1.5952e-05, 3.0558e-05, 4.6387e-05, 1.7993e-04, 1.8121e-05, 1.0279e-05,
        2.4892e-05, 2.0885e-04, 2.5335e-05, 9.7843e-06, 1.5335e-05, 2.9531e-04,
        2.0233e-03, 1.2152e-05, 2.7913e-05, 1.6218e-05, 2.5288e-05, 9.4157e-06,
        1.3079e-05, 2.2066e-05, 1.0505e-05, 3.7164e-05, 6.2961e-05, 1.2592e-05,
        1.1662e-05, 1.7641e-05, 2.8103e-05, 1.1798e-05, 1.2267e-04, 2.5579e-05,
        8.9811e-06, 1.8827e-05, 2.6318e-05, 2.2623e-05, 5.7088e-06, 3.0577e-04,
        2.3084e-05, 7.1880e-06, 6.9735e-05, 4.6706e-05, 9.8907e-05, 6.4090e-06,
        5.9489e-05, 1.5571e-04, 1.8579e-05, 7.1832e-05, 6.1100e-06, 1.9731e-05,
        4.8894e-05, 2.2142e-05, 6.4075e-05, 2.6140e-05, 5.2713e-05, 7.5691e-06,
        5.4081e-06, 1.5293e-04, 1.1637e-05, 7.2641e-06, 1.5508e-04, 1.0620e-04,
        1.2660e-05, 6.5618e-05, 9.7999e-05, 1.1883e-05, 3.0730e-04, 1.1805e-05,
        2.2454e-05, 1.0596e-04, 1.7881e-04, 5.9667e-05, 2.0692e-04, 1.4621e-05,
        3.2724e-05, 2.4530e-05, 9.8922e-06, 1.5698e-05, 2.9553e-04, 1.2711e-05,
        2.4724e-05, 2.6791e-05, 1.1388e-05, 1.3997e-04, 4.1722e-06, 6.9492e-05,
        1.1302e-04, 1.1220e-04, 6.2062e-06, 5.1018e-06, 1.1025e-05, 1.3924e-04,
        3.7081e-05, 2.3044e-05, 3.4148e-05, 2.8614e-05, 2.5899e-05, 1.4480e-05,
        8.4443e-06, 2.1880e-05, 7.4145e-06, 2.5774e-04, 8.1975e-06, 3.3388e-05,
        1.3750e-05, 1.9278e-04, 1.2484e-05, 4.6565e-05, 2.0616e-05, 1.8581e-05,
        4.8081e-04, 1.4159e-05, 1.3556e-05, 2.0160e-05, 3.0675e-05, 2.7016e-05,
        5.2755e-05, 7.8972e-06, 9.7400e-06, 4.6357e-05, 2.1526e-05, 4.8727e-05,
        3.3699e-05, 1.7969e-05, 7.2503e-06, 2.8993e-06, 6.9393e-06, 1.3792e-05,
        1.6703e-04, 2.4179e-05, 4.1290e-05, 6.9470e-06, 3.2605e-05, 8.9561e-05,
        1.7389e-05, 5.2295e-06, 6.6123e-06, 1.2500e-04, 4.9224e-06, 1.2022e-04,
        3.1529e-06, 8.0021e-06, 1.9490e-06, 1.2208e-03, 2.2974e-05, 9.4730e-06,
        2.2900e-04, 2.8642e-05, 3.6676e-06, 5.2206e-05, 2.3600e-05, 3.2906e-05,
        9.7542e-06, 3.8454e-06, 5.9476e-06, 4.5020e-06, 8.7768e-05, 2.1648e-04,
        1.1481e-05, 1.2313e-06, 1.5597e-04, 3.3612e-05],
       grad_fn=<MulBackward0>)
--- CC   ------ CC   ------ CC   ------ CC   ------ CC   ------ CC   ------ CC   ------ CC   ------ CC   ------ CC   ---
German shepherd, German shepherd dog, German police dog, alsatian 98.72412872314453
[('German shepherd, German shepherd dog, German police dog, alsatian', 98.72412872314453), ('malinois', 0.4503569006919861), ('Norwegian elkhound, elkhound', 0.1573256552219391), ('Leonberg', 0.10771603882312775), ('muzzle', 0.08178723603487015)]
(pytorch_venv) dhankar@dhankar-1:~/temp/pytorch/PyTorch_1/Scripts_InLineComments$ 
```

#

```
$ pip install albumentations
Processing /home/dhankar/.cache/pip/wheels/d2/e3/0b/99a239413035502833a7b07283894243fddf5ce3aa720ca8dd/albumentations-0.4.6-py3-none-any.whl
Requirement already satisfied: scipy in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from albumentations) (1.5.2)
Processing /home/dhankar/.cache/pip/wheels/13/90/db/290ab3a34f2ef0b5a0f89235dc2d40fea83e77de84ed2dc05c/PyYAML-5.3.1-cp38-cp38-linux_x86_64.whl
Requirement already satisfied: numpy>=1.11.1 in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from albumentations) (1.19.1)
Collecting opencv-python>=4.1.1
  Downloading opencv_python-4.4.0.42-cp38-cp38-manylinux2014_x86_64.whl (49.4 MB)
     |████████████████████████████████| 49.4 MB 27.7 MB/s 
Collecting imgaug>=0.4.0
  Using cached imgaug-0.4.0-py2.py3-none-any.whl (948 kB)
Requirement already satisfied: Pillow in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from imgaug>=0.4.0->albumentations) (7.2.0)
Requirement already satisfied: six in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from imgaug>=0.4.0->albumentations) (1.15.0)
Collecting Shapely
  Downloading Shapely-1.7.1-cp38-cp38-manylinux1_x86_64.whl (1.0 MB)
     |████████████████████████████████| 1.0 MB 56.0 MB/s 
Collecting imageio
  Downloading imageio-2.9.0-py3-none-any.whl (3.3 MB)
     |████████████████████████████████| 3.3 MB 40.8 MB/s 
Requirement already satisfied: matplotlib in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from imgaug>=0.4.0->albumentations) (3.3.1)
Collecting scikit-image>=0.14.2
  Using cached scikit_image-0.17.2-cp38-cp38-manylinux1_x86_64.whl (12.4 MB)
Requirement already satisfied: certifi>=2020.06.20 in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from matplotlib->imgaug>=0.4.0->albumentations) (2020.6.20)
Requirement already satisfied: kiwisolver>=1.0.1 in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from matplotlib->imgaug>=0.4.0->albumentations) (1.2.0)
Requirement already satisfied: python-dateutil>=2.1 in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from matplotlib->imgaug>=0.4.0->albumentations) (2.8.1)
Requirement already satisfied: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.3 in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from matplotlib->imgaug>=0.4.0->albumentations) (2.4.7)
Requirement already satisfied: cycler>=0.10 in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from matplotlib->imgaug>=0.4.0->albumentations) (0.10.0)
Collecting PyWavelets>=1.1.1
  Using cached PyWavelets-1.1.1-cp38-cp38-manylinux1_x86_64.whl (4.4 MB)
Collecting networkx>=2.0
  Downloading networkx-2.5-py3-none-any.whl (1.6 MB)
     |████████████████████████████████| 1.6 MB 56.5 MB/s 
Collecting tifffile>=2019.7.26
  Downloading tifffile-2020.9.3-py3-none-any.whl (148 kB)
     |████████████████████████████████| 148 kB 33.5 MB/s 
Requirement already satisfied: decorator>=4.3.0 in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from networkx>=2.0->scikit-image>=0.14.2->imgaug>=0.4.0->albumentations) (4.4.2)
Installing collected packages: PyYAML, opencv-python, Shapely, imageio, PyWavelets, networkx, tifffile, scikit-image, imgaug, albumentations
Successfully installed PyWavelets-1.1.1 PyYAML-5.3.1 Shapely-1.7.1 albumentations-0.4.6 imageio-2.9.0 imgaug-0.4.0 networkx-2.5 opencv-python-4.4.0.42 scikit-image-0.17.2 tifffile-2020.9.3
(pytorch_venv) dhankar@dhankar-1:~/temp/pytorch/PyTorch_1$ 
```
#

> pip install tensorboard

```
$ pip install tensorboard
Collecting tensorboard
  Downloading tensorboard-2.3.0-py3-none-any.whl (6.8 MB)
     |████████████████████████████████| 6.8 MB 4.6 MB/s 
Collecting grpcio>=1.24.3
  Downloading grpcio-1.32.0-cp38-cp38-manylinux2014_x86_64.whl (3.8 MB)
     |████████████████████████████████| 3.8 MB 30.3 MB/s 
Requirement already satisfied: six>=1.10.0 in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from tensorboard) (1.15.0)
Requirement already satisfied: numpy>=1.12.0 in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from tensorboard) (1.19.1)
Collecting werkzeug>=0.11.15
  Using cached Werkzeug-1.0.1-py2.py3-none-any.whl (298 kB)
Requirement already satisfied: wheel>=0.26; python_version >= "3" in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from tensorboard) (0.34.2)
Requirement already satisfied: requests<3,>=2.21.0 in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from tensorboard) (2.24.0)
Requirement already satisfied: setuptools>=41.0.0 in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from tensorboard) (49.2.0.post20200714)
Collecting markdown>=2.6.8
  Using cached Markdown-3.2.2-py3-none-any.whl (88 kB)
Collecting google-auth-oauthlib<0.5,>=0.4.1
  Using cached google_auth_oauthlib-0.4.1-py2.py3-none-any.whl (18 kB)
Collecting absl-py>=0.4
  Downloading absl_py-0.10.0-py3-none-any.whl (127 kB)
     |████████████████████████████████| 127 kB 31.9 MB/s 
Collecting protobuf>=3.6.0
  Downloading protobuf-3.13.0-cp38-cp38-manylinux1_x86_64.whl (1.3 MB)
     |████████████████████████████████| 1.3 MB 117.0 MB/s 
Collecting google-auth<2,>=1.6.3
  Downloading google_auth-1.21.1-py2.py3-none-any.whl (93 kB)
     |████████████████████████████████| 93 kB 115 kB/s 
Collecting tensorboard-plugin-wit>=1.6.0
  Downloading tensorboard_plugin_wit-1.7.0-py3-none-any.whl (779 kB)
     |████████████████████████████████| 779 kB 15.1 MB/s 
Requirement already satisfied: chardet<4,>=3.0.2 in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard) (3.0.4)
Requirement already satisfied: certifi>=2017.4.17 in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard) (2020.6.20)
Requirement already satisfied: idna<3,>=2.5 in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard) (2.10)
Requirement already satisfied: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /home/dhankar/anaconda3/envs/pytorch_venv/lib/python3.8/site-packages (from requests<3,>=2.21.0->tensorboard) (1.25.10)
Collecting requests-oauthlib>=0.7.0
  Using cached requests_oauthlib-1.3.0-py2.py3-none-any.whl (23 kB)
Collecting rsa<5,>=3.1.4; python_version >= "3.5"
  Downloading rsa-4.6-py3-none-any.whl (47 kB)
     |████████████████████████████████| 47 kB 602 kB/s 
Collecting cachetools<5.0,>=2.0.0
  Downloading cachetools-4.1.1-py3-none-any.whl (10 kB)
Collecting pyasn1-modules>=0.2.1
  Using cached pyasn1_modules-0.2.8-py2.py3-none-any.whl (155 kB)
Collecting oauthlib>=3.0.0
  Using cached oauthlib-3.1.0-py2.py3-none-any.whl (147 kB)
Collecting pyasn1>=0.1.3
  Using cached pyasn1-0.4.8-py2.py3-none-any.whl (77 kB)
Installing collected packages: grpcio, werkzeug, markdown, pyasn1, rsa, cachetools, pyasn1-modules, google-auth, oauthlib, requests-oauthlib, google-auth-oauthlib, absl-py, protobuf, tensorboard-plugin-wit, tensorboard
Successfully installed absl-py-0.10.0 cachetools-4.1.1 google-auth-1.21.1 google-auth-oauthlib-0.4.1 grpcio-1.32.0 markdown-3.2.2 oauthlib-3.1.0 protobuf-3.13.0 pyasn1-0.4.8 pyasn1-modules-0.2.8 requests-oauthlib-1.3.0 rsa-4.6 tensorboard-2.3.0 tensorboard-plugin-wit-1.7.0 werkzeug-1.0.1
(pytorch_venv) dhankar@dhankar-1:~/temp/pytorch/PyTorch_1/Scripts_InLineComments$ 
(pytorch_venv) dhankar@dhankar-1:~/temp/pytorch/PyTorch_1/Scripts_InLineComments$ python 4_alb_aug.py 
(pytorch_venv) dhankar@dhankar-1:~/temp/pytorch/PyTorch_1/Scripts_InLineComments$ 
```
#
