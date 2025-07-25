************************************************************************************ Project name: FSRCNN ************************************************************************************ 

This repository is implementation of the new robust architecture called [" Fast Super Resolution Convolutional Neural Network with Residual Block (FSRCNN-RB) "].

===========================================================================================================================================================================

## Differences between our network (FSRCNN-RB) with the original network (FSRCNN)

The overall architecture of our proposed network (FSRCNN-RB) is depicted in Figure 4a. The key distinctions between our network and FSRCNN can be summarized in three aspects. First, unlike FSRCNN, which utilizes a single 
convolutional layer for feature extraction, we have implemented three asymmetric convolutional layers to enhance the convolutional kernels in both horizontal and vertical directions, thereby improving the extraction of local salient 
features. The second key difference is that FSRCNN often produces reconstructed images with artifacts due to the deconvolution layer used in its upsampling module. To address this issue, we incorporate a PixelShuffle operation 
between two convolutional layers within our network architecture (Figure 4d). PixelShuffle, is extensively utilized in image SR processing [68]. This technique enhances the resolution of feature maps by decreasing the number of 
feature channels, which also leads to a reduction in the number of parameters for subsequent convolution operations. PixelShuffle preserves all feature information, effectively mitigating edge blur and artifacts that can arise from 
information loss. To enhance the resolution of output feature maps and facilitate further feature fusion, we implement the PixelShuffle operation in the FSRCNN-RB. PixelShuffle rearrange elements in a tensor according to an 
upscaling factor. In fact, PixelShuffle (r) rearranges elements from a tensor of shape (W, H, Cr2) to a tensor of shape (r × W, r × H, C), where 'r' is an upscaling factor. The upscaling factor indicates how much the original image will 
be enlarged. In our network, the values of W, H, C and r are equal to 32, 32, 2, and 4 respectively. The final difference of our introduced network with the other two architectures is that in conventional neural networks like FSRCNN,
layers learn the input distribution; however, in our proposed network, the implemented residual blocks enable the network to learn the output-input distribution, which is why they are referred to as residual blocks. The structure of 
these residual blocks is illustrated in Figure 4b and 4c, where a direct connection bypassing the intermediate layers is evident. This connection, known as a skip connection, is a fundamental feature of residual blocks.


===========================================================================================================================================================================

## Requirements

- PyTorch 1.13.1
- Numpy 1.21.6
- matplotlib 3.5.3
- h5py 3.8.0
- tqdm 4.67.1
- numpy 1.21.6
- pandas 1.3.5

===========================================================================================================================================================================

# datasets.py: In this file, the Image Dataset class is defined, which can be used to input images into the network and train the network. Of course, it should be noted that the LR and HR images must first be placed in the format of H5
files and the necessary codes for doing this are placed in the Dataset preparation folder.

# utils.py: In this file, the necessary functions for training the network are defined.

# models.py: In this file, the structure of the Fast Super Resolution Convolutional Neural Network with Residual Block (FSRCNN-RB) model is defined in the form of a class, which is used to increase the quality of LR images.

===========================================================================================================================================================================

************************************************************************************ Main Code: FSRCNN_RB.py *********************************************************************************** 

1) In line 50, the address of the H5 file related to the HR images should be written.

2) In line 51, the address of the H5 file related to the LR images should be written.

3) In line 56, the value of learning rate can be set.

4) In line 57, the value of batch-size can be set.

5) In line 58, the number of epochs can be set.

6) In line 84, the type of loss function can be set.

7) In line 86, the type of optimizer can be set.

===========================================================================================================================================================================
#FSRCNN_RB_Test.py: 

Using this file, the network can be tested. For this purpose, the address of the file containing the best weights of the trained network must be written in line 127.
===========================================================================================================================================================================