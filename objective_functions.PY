import numpy as np
import cv2
#from sklearn.metrics import jaccard_score
img = cv2.imread('combined_image.jpg')[...,::-1]/255.0

noise =  np.random.normal(loc=0, scale=1, size=img.shape)
perturbation = np.clip((img + noise*.1),0,1)

from math import sqrt
def obj_intensity(perturbation):
    sum=0
    for i in perturbation:
        for j in i:
            for k in j:
                sum=k+sum
    return sqrt(sum)

print(obj_intensity(perturbation))
   

def obj_degrad(perturbation, image, object_detector):
   
    if object_detector=='yolo':
        A=0


#def pixelAccuracy(y_pred, y_true):
    #y_pred = np.argmax(np.reshape(y_pred,[N_CLASSES_PASCAL,img_rows,img_cols]),axis=0)
    #y_true = np.argmax(np.reshape(y_true,[N_CLASSES_PASCAL,img_rows,img_cols]),axis=0)
    #y_pred = y_pred * (y_true>0)

    #return 1.0 * np.sum((y_pred==y_true)*(y_true>0)) /  np.sum(y_true>0)
def compute_iou(y_pred, y_true):
     # ytrue, ypred is a flatten vector
     y_pred = y_pred.flatten()
     y_true = y_true.flatten()
     current = confusion_matrix(y_true, y_pred, labels=[0, 1])
     # compute mean iou
     intersection = np.diag(current)
     ground_truth_set = current.sum(axis=1)
     predicted_set = current.sum(axis=0)
     union = ground_truth_set + predicted_set - intersection
     IoU = intersection / union.astype(np.float32)
     return np.mean(IoU)
