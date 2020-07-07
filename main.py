'''
'''

import numpy as np
import cv2
import sys


def translate(I, x, y):
    ''' Translate the given image by the given offset in the x and y directions.
    '''
    # Make a transformation matrix
    M = np.array([[1,0,x],[0,1,y]], np.float)
    # Get the number of rows and columns
    rows, cols = I.shape[:2]
    # Shift the image by the given offset. watch out for x,y vs. row,col
    I = cv2.warpAffine(I, M ,(cols,rows))
    return I

def compute_ssd(I1, I2):
    I1 = I1.astype(np.float)
    I2 = I2.astype(np.float)
    ssd = np.sum((I1-I2)*(I1-I2))
    return ssd

def align_images(I1, I2):
    rows, cols = I2.shape[:2]
    bestx = 0
    besty = 0
    minimum = sys.maxsize
    delta = 15
    for x in range(1,2*delta+1,1):
        for y in range(1,2*delta,1):
            z = compute_ssd(I1,translate(I2,x - delta,y - delta))
            temp = translate(I2,x - delta,y - delta)
            text = "Translated X:{},Y:{}".format(x - delta,y - delta)
            text2 = "SSD:{:.2f}".format(z)
            text3 = "Best SSD:{:.2f}".format(minimum)
            cv2.putText(temp,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,.6,(66,66,255))
            cv2.putText(temp,text2,(10,40),cv2.FONT_HERSHEY_SIMPLEX,.6,(66,66,255))
            cv2.putText(temp,text3,(10,60),cv2.FONT_HERSHEY_SIMPLEX,.6,(66,66,255))
            cv2.imshow("fixed?",temp)
            delay = 2
            key = cv2.waitKey(delay)
            if z < minimum:
                minimum = z
                bestx = x - delta
                besty = y - delta
    #cv2.imshow("fixed?",translate(I2,bestx,besty))
    return translate(I2,bestx,besty)

# --------------------------------------------------------------------------

# this is the image to be processed
image_name = 'logs.jpg'

# Read in the input image and convert to grayscale.
img = cv2.imread(image_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
rows, cols = img.shape[:2]

# Extract the color channels.
# Note that The 3 color channels are stacked vertically.
# The blue channel is the top third of the image, the green channel is the
# middle third, and the red channel is the bottom third.
"slice the picture into 3 pieces"
blue = gray[0:rows//3,0:]
green = gray[rows//3:2*(rows//3),0:]
red = gray[2*(rows//3):,0:]
if red.shape != green.shape:
    red = red[0:red.shape[0]-1,0:red.shape[1]]

# First align the green channel with the blue.
fixedGreen = align_images(blue,green)
# Next align the red channel with the blue.
fixedRed = align_images(blue,red)

# Merge the three color channels into a single color image.
final = cv2.merge((blue,fixedGreen,fixedRed))
final = final[20:-20,20:-20]
cv2.imshow("combine",final)

delay = 0
key = cv2.waitKey(delay)
