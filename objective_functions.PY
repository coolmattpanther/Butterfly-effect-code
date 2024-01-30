import numpy as np
import cv2
#from sklearn.metrics import jaccard_score
img = cv2.imread('combined_image.jpg')[...,::-1]/255.0

noise =  np.random.normal(loc=0, scale=1, size=img.shape)

from ultralytics import YOLO

model = YOLO('yolov8n-seg.pt')

# Make predictions with the model
results = model.predict('3.png')
B=[]
B_2=[]

for result in results:
       # masks, (N, H, W)
    names = model.names
for r in results:
  B= r.masks.xy
  B_2=r.boxes.cls
results1=model.predict('combined_image.jpg')
B_1=[]
B_3=[]

for result in results1:
       # masks, (N, H, W)
    names = model.names
for r in results1:
  B_1= r.masks.xy
  B_3=r.boxes.cls
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

#print(obj_intensity(perturbation))
def get_iou(ground_truth, pred):
    # coordinates of the area of intersection.
    ix1 = np.maximum(ground_truth[0], pred[0])
    iy1 = np.maximum(ground_truth[1], pred[1])
    ix2 = np.minimum(ground_truth[2], pred[2])
    iy2 = np.minimum(ground_truth[3], pred[3])
     
    # Intersection height and width.
    i_height = np.maximum(iy2 - iy1 + 1, np.array(0.))
    i_width = np.maximum(ix2 - ix1 + 1, np.array(0.))
     
    area_of_intersection = i_height * i_width
     
    # Ground Truth dimensions.
    gt_height = ground_truth[3] - ground_truth[1] + 1
    gt_width = ground_truth[2] - ground_truth[0] + 1
     
    # Prediction dimensions.
    pd_height = pred[3] - pred[1] + 1
    pd_width = pred[2] - pred[0] + 1
     
    area_of_union = gt_height * gt_width + pd_height * pd_width - area_of_intersection
     
    iou = area_of_intersection / area_of_union
     
    return iou
#print(get_iou(B[2],B_1[2]))
def obj_degrad(B,B_1,B_2,B_3, object_detector):


    u=0
    if object_detector=='yolo':
       
        A=0
        for i in B_2:
            a_0=0
            for j in B_3:
                k=0
                if i==j:
                    
                    a_0=max(a_0, get_iou(B[u],B_1[k]).all())
                k=k+1
            u=u+1    
        A=A+a_0

    return A
#print(obj_degrad(B,B_1,B_2,B_3, "yolo"))
#print(B_2)
#print(B_3)
#def pixelAccuracy(y_pred, y_true):
    #y_pred = np.argmax(np.reshape(y_pred,[N_CLASSES_PASCAL,img_rows,img_cols]),axis=0)
    #y_true = np.argmax(np.reshape(y_true,[N_CLASSES_PASCAL,img_rows,img_cols]),axis=0)
    #y_pred = y_pred * (y_true>0)

    #return 1.0 * np.sum((y_pred==y_true)*(y_true>0)) /  np.sum(y_true>0)

def obj_dist( B, B_2):
    x=np.empty([375,1242])
    
    for k in range(375):
        for l in range(1242):

            x[k][l]=sqrt((375^2)+(1242^2))
            for i in B:
                for j in i:
                    x[k][l]=min(x[k][l], sqrt((j[0]-k)**2+(j[1]-l)**2))
    sum=0
    for z in x:
        for l in z:
            sum+=l
    neg_avg=(-1)*sum/(375*1242)
    for k in range(375):
        for l in range(1242):
            for i in B:
                for j in i:
                    x[0][0]=neg_avg
    perturbation_max=np.empty([375,1242])
    for k in range(375):
        for l in range(1242):
            perturbation_max[k][l]=max(noise[k][l][0],noise[k][l][1], noise[k][l][2])
            x[k][l]=x[k][l]*perturbation_max[k][l]
    unperturbed_pixel_count=0
    for i in perturbation_max:
        for j in i:
            unperturbed_pixel_count+=j
    new_sum=0
    for i in x:
        for j in i:
            new_sum+=j
    return new_sum/unperturbed_pixel_count

        
print(obj_dist(B,B_2))   
