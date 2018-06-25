import numpy as np
import glob
import os
from PIL import Image
import math

datafolder = '/projects/melanoma/ISIC/Shima/Data/train'
R = 0.0
G = 0.0
B = 0.0
numPixel = 0

# Calculate Mean value
for x in os.walk(datafolder):
    folder = x[0]
    im_list = glob.glob(os.path.join(datafolder, folder, '*.jpg'))
    for im in im_list:
        image = Image.open(im)
        image = np.array(image, dtype=np.float)
        red = image[:, :, 0]/255
        green = image[:, :, 1]/255
        blue = image[:, :, 2]/255
        R += np.sum(red)
        G += np.sum(green)
        B += np.sum(blue)
        numPixel += red.size

print("Mean Red channel: {}".format(R/numPixel))
print("Mean Green channel: {}".format(G/numPixel))
print("Mean Blue channel: {}".format(B/numPixel))



# Calucalte Std 
R_std = 0.0
G_std = 0.0
B_std = 0.0

for x in os.walk(datafolder):
    folder = x[0]
    im_list = glob.glob(os.path.join(datafolder, folder, '*.jpg'))
    for im in im_list:
        image = Image.open(im)
        image = np.array(image, dtype=np.float)
        red = image[:, :, 0]/255
        green = image[:, :, 1]/255
        blue = image[:, :, 2]/255
        R_std += np.sum((red - R/numPixel)**2)
        G_std += np.sum((green - G/numPixel)**2)
        B_std += np.sum((blue - B/numPixel)**2)


Rstd = math.sqrt(R_std/numPixel)
Gstd = math.sqrt(G_std/numPixel)
Bstd = math.sqrt(B_std/numPixel)

print("Std Red channel: {}".format(Rstd))
print("Std Green channel: {}".format(Gstd))
print("Std Blue channel: {}".format(Bstd))
