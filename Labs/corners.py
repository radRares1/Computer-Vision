import cv2
import numpy as np
from skimage.morphology import (diamond)

img = cv2.imread('square-rectangle.png')
img1 = cv2.imread('MorpologicalCornerDetection.png',0)
h, w = img1.shape[:2]

print(h)
print(w)

cv2.imshow("image2",img1)

kernel = np.ones((3,3),np.uint8)
#
# gray_image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

# Perform a morphological gradient on the image using a 3x3 rectangular structuring element
kernel = np.ones((3,3),np.uint8)
gradient = cv2.morphologyEx(img1, cv2.MORPH_GRADIENT, kernel)

print(gradient)

cv2.imshow("gradient",gradient)

# Create the structuring elements for cross, diamond, 'x' and rectangle
cross_kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
diamond_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
diamond_kernel[1,0] = 1
diamond_kernel[1,2] = 1
diamond_kernel[0,1] = 1
diamond_kernel[2,1] = 1
x_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
x_kernel[0,0] = 1
x_kernel[0,2] = 1
x_kernel[2,0] = 1
x_kernel[2,2] = 1
square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

img_cpy=gradient

# Perform erosion and dilation on the gradient image using the structuring elements
r1 = cv2.dilate(img_cpy, cross_kernel)
r1 = cv2.erode(r1, diamond_kernel)
r2 = cv2.dilate(img_cpy, x_kernel)
r2 = cv2.erode(r2, square_kernel)

# Subtract the different results to get the corners
result = cv2.absdiff(r1, r2)

cv2.imshow('result', result)

# Threshold the result to remove noise
threshold = 20
ret,thresh = cv2.threshold(result,threshold,255,cv2.THRESH_BINARY)

cv2.imshow('thresh', thresh)

original_img = cv2.imread('MorpologicalCornerDetection.png')

corners = np.column_stack(np.where(thresh > 0))

# Draw a red circle around each corner on the original image
for corner in corners:
    x, y = corner
    cv2.circle(original_img, (y, x), 5, (0, 0, 255), 1)

# Show the final image
cv2.imshow("canvasOutput", original_img)
cv2.waitKey(0)
cv2.destroyAllWindows()


