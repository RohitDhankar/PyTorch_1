#conda activate pytorch_venv

## Code Source >> https://github.com/teralion/explore-ml-cnn/blob/master/conv.py

# -*- coding: utf-8 -*-
"""
Neural Net from Scratch - WORK IN PROGRESS
MAIN CODE Source -- Prof Ahlad Kumar YouTube 
Video -- https://www.youtube.com/watch?v=0zbhg79i_Bs
"""
import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image
#import albumentations as alb
import random , cv2 

from utils.utils_logger import setup_logger
#pytorch_logger = setup_logger(module_name='logs_pytorch', folder_name=str('pytorch_logs_dir'))

gray_img_input = cv2.imread('./img_inputs/dog.jpg',cv2.IMREAD_GRAYSCALE)/255
plt.imshow(gray_img_input)#,cmap = 'gray')
plt.title("Gray--DogImage-with-cv2.IMREAD_GRAYSCALE-|  NO-->> cmap")
plt.show() # Green Dog ?
#
plt.imshow(gray_img_input,cmap = 'gray')
plt.title("Gray--DogImage-with-cv2.IMREAD_GRAYSCALE-|cmap =gray")
plt.show() # GRAY Dog - OK
print("gray_img_input.shape--->>" ,gray_img_input.shape)#(595, 1176 )
#
noChange_img_input = cv2.imread('./img_inputs/dog.jpg',cv2.IMREAD_UNCHANGED)
plt.imshow(noChange_img_input)#,cmap = 'gray')
plt.title("OriginalDogImage-with-cv2.IMREAD_UNCHANGED-| NO-->> cmap")
plt.show() # Original Dog - OK
print("noChange_img_input.shape-->>" , noChange_img_input.shape)#(595, 1176 ,3)
#pytorch_logger.info(f"-Logging --img_input.shape_>> : {str(img_input.shape)}")

class Conv_class:
    def __init__(self, num_filters,filter_size,filter_image):
        """[summary]

        Args:
            num_filters ([type]): [description]
            filter_size ([type]): [description]
        """
        self.num_filters = num_filters 
        self.filter_size = filter_size
        self.conv_filter = np.random.randn(num_filters,filter_size,filter_size)/(filter_size*filter_size)
        # self.filters = np.random.randn(num_filters, 3, 3) / 9 
        # self.last_input = None
    
    
    def image_region(self, image):
        '''
        Generator method which will YIELD the image_patch ( Numpy Array) of Dimensions - 7X7
        These - image_patch - numpy arrays will be the TARGET's over which the KERNEL's will be EMBOSSED
        WIP - Count image_patches ... #Maybe == 6,89,130 , from experimenting with Counter == cntImgPatch 
        '''
        height, width = image.shape
        #print(height) # 595 , #(595, 1176)
        #self.image = image # Not required ?? 
        f_size = self.filter_size 
        #print("Func = image_region , Filter Size---> ", f_size) #7
        #cntImgPatch = 0

        for j in range(height - f_size +1):
            for k in range(width - f_size +1):
                image_patch = image[j:(j+f_size), k:(k+f_size)]
                #print(type(image_patch)) #<class 'numpy.ndarray'>
                #print(image_patch.shape) # (7,7)
                #cntImgPatch +=1
                #print(cntImgPatch) # Last print == 6,89,130 
                yield image_patch, j, k 
    
    def forward_prop(self, image):
        '''
        '''
        height, width = image.shape
        f_size = self.filter_size
        num_filters = self.num_filters
        conv_fil = self.conv_filter

        conv_out = np.zeros((height- f_size +1, width- f_size +1, num_filters))
        for image_patch, i, j in self.image_region(image): #Unpacking values ?
            conv_out[i, j] = np.sum(image_patch * conv_fil, axis=(1,2)) 
            #this is the actual convolution
        return conv_out
  
    def backprop(self,dL_dout, learning_rate): 
        '''
        #loss_gradient = dL_dout -- is the output of the MaxPooling Layer
        '''
        dL_dF_params = np.zeros(self.conv_filter.shape) #loss_filters = dL_dF_params
        for image_patch, i, j in self.image_region(self.image): #last_input
            for k in range(self.num_filters):
                dL_dF_params[k] += dL_dout[i, j, k] * image_patch  ##loss_gradient = dL_dout  
        # Filter params update
        self.conv_filter -= learning_rate*dL_dF_params
        #Above - update rule for the Conv Filter - it updates the Weights of the Conv Filter
        return dL_dF_params

conn = Conv_class(18,7) # Count of Filters == 18 , Shape of Filters == 7 X 7 

## OK_below_Uncomment_JAN22
#img_out1 = conn.forward_prop(img_input)
#print(type(img_out1)) #<class 'numpy.ndarray'>
#print(img_out1.shape) # (589, 1170, 18)


# Done -- Loop through various plots - to see effect of Diff Filters
## OK_below_Uncomment_JAN22
# for plt_cnt in range(17):
#     plt.imshow(img_out1[:,:,plt_cnt])
#     plt.show(block=False)
#     plt.pause(1)
#     plt.close()
# FOOBAR_WIP -- Name the Plots so that they relate back to the Filters 
# FOOBAR_WIP -- Try Not to use random Filters ?? 











        
        
# '''
#     Much of the info contained after the convolution is redundant
#     Example: we can find the same edge using an edge detecting filter by shifting 1 pixel from the first found edge location
# '''
# class Pooling:
#     def __init__(self):
#         '''
#             A Max Pooling layer using a pool size of 2
#         '''
#         self.last_input = None
    
    
#     def iterate_regions(self, image):
#         '''
#             Yields non-overlapping 2x2 image regions to pool over.
#             Image is 2d numpy array
#         '''
#         h, w, _ = image.shape
#         new_h = h // 2
#         new_w = w // 2
#         for i in range(new_h):
#             for j in range(new_w):
#                 image_region = image[(i*2):(i*2+2), (j*2):(j*2+2)]
#                 yield image_region, i, j
                
#     def forward(self, input):
#         '''
#             Forward pass of the pooling layer
#             Input is a 3d numpy array with dimensions (h, w, num_filters)
#             Output is a 3d numpy array with dimensions (h//2, w//2, num_filters)
#         '''
#         self.last_input = input
#         h, w, num_filters = input.shape
#         output = np.zeros((h//2, w//2, num_filters))
#         for image_region, i, j in self.iterate_regions(input):
#             output[i, j] = np.amax(image_region, axis=(0, 1)) #we only want to maximize over the height and weight
#         return output
 
#     def backprop(self, loss_gradient):
#         '''
#             Backward pass of the pooling layer
#             Input is a 3d numpy array (h, w, num_filters)
#             Returns a 3d numpy array with (h//2, w//2, num_filters)
#         '''
#         loss_input = np.zeros(self.last_input.shape)
#         for image_region, i, j in self.iterate_regions(self.last_input):
#             h, w, f = image_region.shape
#             amax = np.amax(image_region, axis=(0,1))
#             for i2 in range(h):
#                 for j2 in range(w):
#                     for f2 in range(f):
#                         #we want to copy the gradients loss if the pixel was the max value
#                         if image_region[i2,j2,f2] == amax[f2]:
#                             loss_input[i*2+i2,j*2+j2,f2] = loss_gradient[i, j, f2]
#         return loss_input
        
    
# '''
#     Final layer for the multiclass classification problem.
#     Uses the Softmax function as activation => CNN gains the ability to make predictions
#     10 nodes, each representing a digit
#     The digit represented by the node with the highest probabilty will be the output of the CNN
# '''
# class Softmax:
#     def __init__(self, input_length, nodes):
#          '''
#              Standard fully-connected layer with softmax activation
#          '''
#          self.weights = np.random.randn(input_length, nodes) / input_length #as in convolution layer, divide in order to reduce the variance
#          self.biases = np.zeros(nodes)
#          self.last_input_shape = None
#          self.last_input = None
#          self.last_totals = None
    
#     def forward(self, input):
#         '''
#             Forward pass of the Softmax layer
#             Output is 1d array containing respective probability values
#         '''
#         self.last_input_shape = input.shape
#         input = input.flatten() #makes it easier to work with the input(there's no need for shape anymore)
#         self.last_input = input
#         input_length, nodes = self.weights.shape
#         totals=np.dot(input, self.weights) + self.biases
#         self.last_totals = totals
#         exp = np.exp(totals)
#         return exp / np.sum(exp, axis=0)
    
#     def backprop(self, loss_gradient, learn_rate):
#         '''
#             Backward pass of the Softmax layer
#         '''
#         for i , gradient in enumerate(loss_gradient):
#             if gradient == 0:
#                 continue
#             #e^totals
#             totals_exp = np.exp(self.last_totals)
#             #sum of all e^totals
#             S = np.sum(totals_exp)
#             #gradient of out[i] against totals
#             gradient_out_totals = -totals_exp[i] * totals_exp / (S**2)
#             gradient_out_totals[i]= totals_exp[i] * (S - totals_exp[i]) / (S**2)
#             #gradient against weights/biases/input
#             gradient_weight = self.last_input
#             gradient_bias = 1
#             gradient_input = self.weights
#             #loss against totals
#             loss = gradient * gradient_out_totals
#             #loss against weights/biases/input
#             loss_weight = gradient_weight[np.newaxis].T @ loss[np.newaxis]
#             loss_bias = loss * gradient_bias
#             loss_input = gradient_input @ loss
            
#             #update stages
#             self.weights -= learn_rate * loss_weight
#             self.biases -= learn_rate * loss_bias
#             return loss_input.reshape(self.last_input_shape)
            
            
# class CNN:
#     def __init__(self, convolution, pooling, softmax, learning_rate, epochs):
#         self.conv = convolution #28*28*1 => 26*26*8
#         self.pool = pooling #26*26*8 => 13*13*8
#         self.softmax= softmax # 13*13*8 => 10
#         self.learning_rate = learning_rate 
#         self.epochs = epochs
        
        
#     def forward(self, image, label):
#         out = self.conv.forward((image/255) - 0.5)
#         out = self.pool.forward(out)
#         out = self.softmax.forward(out)
#         loss = -np.log(out[label])
#         acc = 1 if np.argmax(out) == label else 0
#         return out, loss, acc
    
#     def train(self, train_images, train_labels):
#         print('--- Training ---')
#         for epoch in range(self.epochs):
#             print('--- Epoch %d ---'%(epoch + 1))
#             permutation = np.random.permutation(len(train_images))
#             train_images = train_images[permutation]
#             train_labels = train_labels[permutation]
#             loss = 0
#             num_correct = 0
#             for i, (im, label) in enumerate(zip(train_images, train_labels)):
#                 out, l, acc =  self.forward(im, label)
#                 gradient = np.zeros(10)
#                 gradient[label] = -1 / out[label]
#                 gradient = self.softmax.backprop(gradient, self.learning_rate)
#                 gradient = self.pool.backprop(gradient)
#                 self.conv.backprop(gradient, self.learning_rate)    
#                 loss += l
#                 num_correct +=acc
#                 if i % 100 == 99:
#                     print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %(i + 1, loss / 100, num_correct))
#                     loss = 0
#                     num_correct = 0

#     def test(self, test_images, test_labels):
#         print('--- Testing ---')
#         loss = 0 
#         num_correct = 0
#         for i, (im, label) in enumerate(zip(test_images,test_labels)):
#             out, l, acc = self.forward(im, label)
#             loss += l
#             num_correct += acc
#             if i % 100 == 99:
#                 print('[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %(i + 1, loss / 100, num_correct))
#                 loss = 0
#                 num_correct = 0

        
# if __name__=='__main__':
#     train_images = mnist.test_images()[:1000]
#     train_labels = mnist.test_labels()[:1000]
#     test_images = mnist.test_images()[:1000]
#     test_labels = mnist.test_labels()[:1000]
#     conv = Convolution(8)
#     pool = Pooling()
#     softmax = Softmax(13*13*8, 10)
#     learning_rate = 0.01
#     epochs = 3
#     cnn = CNN(conv, pool, softmax, learning_rate, epochs)
#     cnn.train(train_images, train_labels)
#     cnn.test(test_images, test_labels)