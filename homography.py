
!wget https://basketball-ml.s3-eu-west-1.amazonaws.com/3DCourtBasketball.jpg
!wget https://basketball-ml.s3-eu-west-1.amazonaws.com/2DCourtBasketball.jpg

import cv2
import numpy as np
 
# Read source image.
im_src = cv2.imread('3DCourtBasketball.jpg')

# Four corners of the 3D court + mid-court circle point in source image
# Start top-left corner and go anti-clock wise + mid-court circle point
pts_src = np.array([[125, 128], [70, 350], [622, 163], [391, 112], [488, 140]])

# Read destination image.
im_dst = cv2.imread('2DCourtBasketball.jpg')

# Four corners of the court + mid-court circle point in destination image 
# Start top-left corner and go anti-clock wise + mid-court circle point
pts_dst = np.array([[8, 7], [8, 355], [631, 355], [631, 7], [320, 224]])

# Calculate Homography
h, status = cv2.findHomography(pts_src, pts_dst)
  
# Warp source image to destination based on homography
im_out = cv2.warpPerspective(im_src, h, (im_dst.shape[1],im_dst.shape[0]))


from google.colab.patches import cv2_imshow
# Display images
cv2_imshow(im_src)
cv2_imshow(im_dst)

cv2_imshow(im_out)
