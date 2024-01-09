import cv2 

import numpy as np 
# Load the two images 
image1 = cv2.imread('right.jpg') 
image2 = cv2.imread('leftnoise.jpg')
# Combine the images horizontally 
combined_image = np.hstack((image2, image1))  
# Display the combined image 
cv2.imshow('Combined Image', combined_image) 
cv2.waitKey(0) 
cv2.destroyAllWindows() 
# Save the combined image 
cv2.imwrite('combined_image.jpg', combined_image)