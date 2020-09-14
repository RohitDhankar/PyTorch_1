# -*- coding: utf-8 -*-
"""
Neural Net from Scratch
Source -- Prof Ahlad Kumar YouTube Video -- https://www.youtube.com/watch?v=0zbhg79i_Bs
"""
import matplotlib.pyplot as plt
import numpy as np
#from PIL import Image
#import albumentations as alb
import random , cv2 

img = cv2.imread('../img_inputs/dog.jpg',cv2.IMREAD_GRAYSCALE)/255
#img = cv2.imread('../img_inputs/dog.jpg')#Not GrayScale
plt.imshow(img,cmap = 'gray')
plt.show()
print(img.shape)#(595, 1176)

class Conv:
    def __init__(self, num_filters,filter_size):
        '''
        num_filters,filter_size
        '''
        self.num_filters = num_filters 
        self.filter_size = filter_size
        self.conv_filter = np.random.randn(num_filters,filter_size,filter_size)/(filter_size*filter_size)
        # self.filters = np.random.randn(num_filters, 3, 3) / 9 
        # self.last_input = None
    
    
    def image_region(self, image):
        '''
        '''
        height, width = image.shape
        #self.image = image # Not required ?? 
        f_size = self.filter_size 

        for j in range(height - f_size +1):
            for k in range(width - f_size +1):
                image_patch = image[j:(j+f_size), k:(k+f_size)]
                yield image_region, j, k
    
#     def forward(self, input):
#         '''
#             Forward pass of the convolution layer
#             Input is a 2d numpy array
#             Returns a 3d numpy array with dimensions (height, width, num_filters)
#         '''
#         self.last_input = input
#         height, width = input.shape
#         output = np.zeros((height-2, width-2, self.num_filters))
#         for image_region, i, j in self.iterate_regions(input):
#             output[i, j] = np.sum(image_region * self.filters, axis=(1,2)) #this is the actual convolution
#         return output
  
#     def backprop(self, loss_gradient, learning_rate):
#         '''
#             Backward pass of the convolution layer
#         '''
#         loss_filters = np.zeros(self.filters.shape)
#         for image_region, i, j in self.iterate_regions(self.last_input):
#             for f in range(self.num_filters):
#                 loss_filters[f] += loss_gradient[i, j, f] * image_region        
#         self.filters -= learning_rate*loss_filters
        
        
        
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