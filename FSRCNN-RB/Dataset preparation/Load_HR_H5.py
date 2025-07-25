from PIL import Image
import glob
import numpy as np
import time
import cv2
import h5py
from datetime import timedelta


# y = cv2.imread('E:/Pre-Process/Final Images/Q01_5mag_resample_NLMF (1).tif')
y = cv2.imread('E:/Pre-Process/Final Images/dis_a32_x1_y1_z10000.png')
# # y = cv2.imread('E:/Pre-Process/Final Images/Q01_5mag_resample_NLMF.tif')
# y = cv2.imread('E:/Pre-Process/Final Images/FIBSLICE (1).tif')
print('dtype Original Images: ', y.dtype)

Image_Folder_Path = 'E:/Pre-Process/Final Images/'

def Load_2D_Images (Image_Folder_Path):
    # Finding Number of PNG Images
    g = glob.glob1(Image_Folder_Path, "*.tif")
    Number_of__Images = len(g)
    print(f'{Number_of__Images} 2D TIF images were Found')

    # Find w & h of a sample image at index of 0
    im = Image.open(Image_Folder_Path + g[0])
    h, w = im.size
    print(f'2D image size: {h}x{w}')
    PNGarray = np.zeros((Number_of__Images, w, h)).astype(np.float)
    for i, img in enumerate(g):
        PNGarray[i, :, :] = np.array(Image.open(Image_Folder_Path + img))

    return PNGarray

t = time.time()
print('*** Loading the main image ***')
# Loading 2D PNG Images
PNGarray = Load_2D_Images(Image_Folder_Path)
print('dtype Load Images: ', PNGarray.dtype)

PNGarray_New = np.zeros((20000, 128, 128)).astype(np.float)
for i in range (0, 20000):
    PNGarray_New[i] = cv2.normalize(PNGarray[i], None, 0.0, 1.0, cv2.NORM_MINMAX)
print(np.min(PNGarray_New[0]), " , ", np.max(PNGarray_New[0]))
print('dtype Images After Normalization: ', PNGarray_New.dtype)
print('Shape Images After Normalization: ', PNGarray_New.shape)

print(f'{PNGarray_New.shape[0]} subimages with a size of {(128,128)} were generated.')

print('*** Saving h5 database ***')
# Saving the h5 file
h5py_name = "HR_Q01.h5"
hf = h5py.File(h5py_name, 'w')
hf.create_dataset('3D_image', data=PNGarray_New)

hf.close()

elapsed = time.time() - t
print('elapsed time=', str(timedelta(seconds=elapsed)), ' sec')