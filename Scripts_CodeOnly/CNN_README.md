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
- 

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




