import numpy as np
import cv2
import matplotlib.pyplot as plt
from objective_functions.py import obj_intensity
img = cv2.imread('3.png')
noise =  np.random.normal(loc=0, scale=1, size=img.shape)
noisy = np.clip((img + noise*0),0,1)
yolo=[]
detr=[]
for i in range(100):
    img = cv2.imread('3.png')
    h, w, channels = img.shape
 
    half = w//2
 
 
    # this will be the first column
    left_part = img[:, :half] 
 
    # [:,:half] means all the rows and
    # all the columns upto index half
 
    # this will be the second column``
    right_part = img[:, half:]  
 
    # [:,half:] means all the rows and all
    # the columns from index half to the end
    # cv2.imshow is used for displaying the image
    #v2.imshow('Left part', left_part)
    #cv2.imshow('Right part', right_part)
 
    # this is horizontal division
    half2 = h//2
 
    top = img[:half2, :]
    bottom = img[half2:, :]
 
    #cv2.imshow('Top', top)
    #cv2.imshow('Bottom', bottom)
 
    # saving all the images
    # cv2.imwrite() function will save the image 
    # into your pc

    cv2.imwrite('right2.jpg', right_part)
    cv2.imwrite('left2.jpg', left_part)
    cv2.waitKey(0)
    image = cv2.imread('left.jpg')[...,::-1]/255.0
    noise =  np.random.normal(loc=0, scale=1, size=img.shape)
    noisy = np.clip((img + noise*.1),0,1)
    plt.imsave('leftnoise1.jpg', noisy)
    # noise overlaid over image
    
    image1 = cv2.imread('right2.jpg') 
    image2 = cv2.imread('leftnoise1.jpg')
    # Combine the images horizontally 
    combined_image = np.hstack((image2, image1))  
    # Display the combined image 
    cv2.imshow('Combined Image', combined_image) 
    cv2.waitKey(0) 
    cv2.destroyAllWindows() 
    # Save the combined image 
    cv2.imwrite('combined_image1.jpg', combined_image)
