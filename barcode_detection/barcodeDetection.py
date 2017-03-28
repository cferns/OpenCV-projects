# import the necessary packages
import numpy as np
import cv2
import matplotlib.pyplot as plt
from functionsFolder.transform import four_point_transform

# load the image and convert it to grayscale
imagePATH = './images/barcode_01.jpg'
image = cv2.imread(imagePATH)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# compute the Scharr gradient magnitude representation of the images in both the x and y direction
gradX = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
gradY = cv2.Sobel(gray, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

# subtract the y-gradient from the x-gradient
gradient = cv2.subtract(gradX, gradY)
gradient = cv2.convertScaleAbs(gradient)

# blur - to smooth out high frequency noise in the gradient representation of the image.
blurred = cv2.blur(gradient, (9, 9))
#  threshold - Any pixel in the gradient image that is not greater than 225 is set to 0 (black).
(_, thresh) = cv2.threshold(blurred, 225, 255, cv2.THRESH_BINARY)

#construct a closing kernel and apply it to the thresholded image
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 7))
closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

#cv2.imshow("closed",closed)

# perform a series of erosions and dilations
closed = cv2.erode(closed, None, iterations=4)
closed = cv2.dilate(closed, None, iterations=4)


# find the contours in the thresholded image, then sort the contours
# by their area, keeping only the largest one
(__, cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
c = sorted(cnts, key = cv2.contourArea, reverse = True)[0]

# compute the rotated bounding box of the largest contour
rect = cv2.minAreaRect(c)
box = np.int32(cv2.boxPoints(rect))
pts = box

warped = four_point_transform(image,pts)
cv2.imshow("warped",warped)

warpedGray = cv2.cvtColor(warped,cv2.COLOR_BGR2GRAY)
cv2.imshow("warpedGray",warpedGray)

#Read the barcode here
midRowIndex = int(len(warped)/2)
warpedGrayImage = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)


#sharpening the image
# generating the kernels
kernel_sharpen_1 = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
kernel_sharpen_2 = np.array([[1,1,1], [1,-7,1], [1,1,1]])
kernel_sharpen_3 = np.array([[-1,-1,-1,-1,-1],
                             [-1,2,2,2,-1],
                             [-1,2,8,2,-1],
                             [-1,2,2,2,-1],
                             [-1,-1,-1,-1,-1]]) / 8.0

# applying different kernels to the input image
output_1 = cv2.filter2D(warpedGrayImage, -1, kernel_sharpen_1)
output_2 = cv2.filter2D(warpedGrayImage, -1, kernel_sharpen_2)
warped_EdgeEnhanced = cv2.filter2D(warpedGrayImage, -1, kernel_sharpen_3)
cv2.imshow("warped_EdgeEnhanced",warped_EdgeEnhanced)

hscale = 95/len(warped_EdgeEnhanced[int(len(warped_EdgeEnhanced)/2)])
warpedImageResized = cv2.resize(warped_EdgeEnhanced, None,fx=  hscale, fy=1, interpolation = cv2.INTER_LINEAR  )
cv2.imshow("midRow gray Image",warpedImageResized)

#Selecting mid row
midGrayImage = warpedImageResized[midRowIndex]

#midGrayImage = cv2.resize(midGrayImage,None,fx=4, fy=4, interpolation = cv2.INTER_CUBIC)

cv2.imshow("midRow gray Image",midGrayImage)




for i in range(len(midGrayImage)):
    if midGrayImage[i]>170:
        midGrayImage[i] = 255;
    else:
        midGrayImage[i] = 0;


cv2.imshow("midGrayImageResized gray Image", midGrayImage)

# draw a bounding box arounded the detected barcode and display the
# image
cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
#cv2.imshow("Image", image)
cv2.waitKey(15000)
cv2.destroyAllWindows()