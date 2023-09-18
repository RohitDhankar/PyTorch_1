### Intro to CNN for Image Processing 


<p align="center">
    <img src="https://github.com/RohitDhankar/PyTorch_1/blob/master/screenCaptures/CNN_AHLAD_KUMAR_2020-09-11%2022-11-27.png" width= "850px">
</p>

<h1 align="center">Intro to CNN for Image Processing - </h1>

> This repository is an absolute intro for - Intro to CNN for Image Processing  **-CNN for Image Processing 2022**
 
> If you are here - kindly feel free to contribute. 


<br/>


### Table of Contents of this repository

- [X] `A-- Intro to CNN ...` 
- [X] `B-- Many thanks to Prof Ahlad Kumar + Soumith Chintala and PyTorch ` 
- [X] `C-- Further explorations with CNN` 
- [X] `D-- Work in Progress` 
- [X] `Work in Progress` 
- [X] `Work in Progress` 
- [X] `Work in Progress` 


<br/>

### References - Always an ongoing effort - Work in Progress

<br/>

- Recommended Further Learning - Reads / YouTube Vids etc 

- PROF.  AHLAD KUMAR - CNN -- Convolution Neural Nets for Image Processing https://www.youtube.com/watch?v=0zbhg79i_Bs&t=1112s

<br/>


## DIPANJAN SARKAR - 
#### DIPANJAN SARKAR --- Session on Convolutional Neural Networks (CNN)-- - https://www.youtube.com/watch?v=YyoSuP_aFN8&t=24s
- DIPANJAN SARKAR ( https://www.linkedin.com/in/dipanzan/ )
#### Link to Video Screencaptures - 
- https://github.com/RohitDhankar/PyTorch_1/tree/master/screenCaptures/Dipanjan_Sarkar_YouTubeVid
- Below notes are sometimes verbatim - i have added my own inputs also 
- Feature extraction -->> Conv Layer >> Conv Filter | Conv Kernels which will build further Feature Maps  
- Pooling Layers -->> Reduce Dimensionality , this compression enhances specific aspects of these Feature Maps
- RELU - Rectified Linear Units - Non Linear Activation functions 
- Overfitting - Predictions on Training data are very good 
- DropOut and BatchNormalization Layers - are used to prevent Overfitting
- Fully Connected Layers -- used as final stage layers for Flattening and Predictions
- CNN's have a Stacked Layer Acrhitecture - Layers are stacked on after another . Output from the previous layers is mostly the Input to the next layers 
- CNN will have CONV LAyers and POOLING layers usually as alternating layers
- One CONV Layer will have Multiple Filters | Kernels - the output of these Filters or Kernels is a Feature MAP 
- The RAW IMAGE is divided into PATCHES of PIXELS - 1 patch could be a 3X3 Pixel Patch (9 PIXELS in Area - 3 LEN and 3 WIDTH)
- All the FILTERS are passed over each PATCH of the IMAGE and for each FILTER passing on Each Patch we have a RESULT which is a DOT PRODUCT
- Thus we will obtain FEATURE MAPS same count as we have FILTERS 
- TODO - convolution stride 
- TODO - SOURCE >> https://arxiv.org/pdf/1409.1556.pdf The image is passed through a stack of convolutional (conv.) layers, where we use filters with a very small receptive field: 3 × 3 (which is the smallest size to capture the notion of left/right, up/down, center) 
- Next layer is the POOLING LAYER - this will Down Sample the  FEATURE MAPS 
- ACTIVATION LAYER -- Non linearity and Back Propagation ( https://en.wikipedia.org/wiki/Backpropagation )
- ReLU (Rectified Linear Unit) activation function -- https://en.wikipedia.org/wiki/Rectifier_(neural_networks) , RELU is better than SIGMOID 
- COLORED IMAGE - will have the usual 3 CHANNELS -- RED | GREEN | BLUE -- RGB ( Refer Image - COLORED_IMAGE_3_Channels_Dipanjan )
- GRAYSCALED IMAGE - will have only 1 Channel and HEIGHT AND WIDTH measures only 
- 
- 

<br/>

## ANDREW NG - 
#### ACTIVATION FUNCTIONS - https://www.youtube.com/watch?v=NkOv_k7r6no 
#### Link to Video Screencaptures - https://github.com/RohitDhankar/PyTorch_1/tree/master/screenCaptures/ANDREW_NG_YouTubeVideo
#### Why Non-linear Activation Functions (C1W3L07) -- https://www.youtube.com/watch?v=NkOv_k7r6no



- RELU - NON LINEAR ACTIVATION functions are required as when there are only LINEAR Activation functions in the hidden layers a Neural Net is alsmost the same as a sigmoid function and nothing more 
- LINEAR ACTIVATION FUNCTIONS -  can be used in the LAST Layers when we have a REGRESSION problem . Like when predicting HOUSE PRICES which are on a continous scale --- $ 0 to $ Some million etc ... We can use a LINEAR ACTIVATION FUNCTION in the Last OutPut layer . 
- 

<br/>

## Referred eBook - FUNDAMENTALS OF DEEP LEARNING 
- https://www.amazon.in/Fundamentals-Deep-Learning-Nikhil-Buduma/dp/1491925612
### AUTHOR - NIKHIL BUDUMA 
### Link to Screencaptures FROM - Referred eBook - FUNDAMENTALS OF DEEP LEARNING 

- layers of neurons that lie sandwiched between the first layer of neurons (input layer) and the last layer of neurons (output layer) are called the hidden layers.
- every layer has the same number of neurons, this is neither necessary nor recommended.
- It is not required that every neuron has its output connected to the inputs of all neurons in the next layer. In fact, selecting which neurons to connect to which other neurons in the next layer is an art that comes from experience.
- In fact, it can be shown that any feed-forward neural network consisting of only linear neurons can be expressed as a network with no hidden layers.
- in order to learn complex relationships, we need to use neurons that employ some sort of nonlinearity.
- 
-
-



<br/>

## Source -- Rocket AI -- 2020Spring_15_NeuralNetworks_CNNs
### AUTHOR - Rocket AI



<br/>


```
Project Directoty Tree --

```

<br/>



<br/>

- Source URL - 

> ...

<br/>

- Source - 

<br/>





<br/>

Rohit Dhankar - https://www.linkedin.com/in/rohitdhankar/




