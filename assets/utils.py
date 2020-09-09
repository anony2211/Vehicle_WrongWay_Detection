'''
Copyright 2020 Vignesh Kotteeswaran <iamvk888@gmail.com>
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

BoundingBox sorting algorithms
'''

import numpy as np
import cv2


def compute_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = np.maximum(boxA[0], boxB[0])
    yA = np.maximum(boxA[1], boxB[1])
    xB = np.minimum(boxA[2], boxB[2])
    yB = np.minimum(boxA[3], boxB[3])
 
    # compute the area of intersection rectangle
    interArea = np.maximum(0, xB - xA + 1) * np.maximum(0, yB - yA + 1)

    #print ('intersection:',interArea)
 
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    #print ('boxA:',boxAArea)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    #print ('boxB:',boxBArea)
        
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / (boxAArea + boxBArea - interArea)
 
    # return the intersection over union value
    
    return iou
     
def nms(box):
  unq_class=np.unique(box[:,0])
  for i in unq_class:
    curr_box=[]
    for j in box:
      if j[0]==i:
        curr_box.append(j)
    ious=[]
    for j in range(len(curr_box)-1):
      ious.append(compute_iou(box[j,2:],box[j+1,2:]))
    print(ious)

def box_filter(box,thresh):
  #print('num boxes:',len(box))
  box_copy=np.copy(box)
  indices=[]
  for i in range(len(box)):
    for j in range(len(box)):
      if i!=j and i<len(box) and j<len(box):
        iou=compute_iou(box[i,2:],box[j,2:])
        #print(iou,i,j,chars[int(box[i,0])-1],chars[int(box[j,0])-1])
      
        if iou>thresh:
          index=np.argmin(np.array([box[i,1],box[j,1]]))
          #print(index,box[i],box[j])
          if index==0:
            #print('*** deleting_'+chars[int(box[i,0])-1])
            #box_del=np.delete(box_copy,i,axis=0)
            indices.append(i)

          if index==1:
            #print('*** deleting_'+chars[int(box[j,0])-1])
            #box_del=np.delete(box_copy,j,axis=0)
            indices.append(j)
      
  #print(indices)
  return np.delete(box,indices,axis=0)

def sorter(dpred,chars):
  s=[i for i in np.argsort(dpred[:,-4])]
  #print(s)
  ordered= np.array([dpred[i,:] for i in s])
  return [chars[int(i-1)] for i in ordered[:,0]]

def xmin_sorter(dpred,chars,return_arg=False):
  s=[i for i in np.argsort(dpred[:,-4])]
  #print(s)
  ordered= np.array([dpred[i,:] for i in s])
  if return_arg:
        return ordered
  else:
    return [chars[int(i-1)] for i in ordered[:,0]]

def ymin_sorter(dpred,chars,return_arg=False):
  s=[i for i in np.argsort(dpred[:,-3])]
  #print(s)
  ordered= np.array([dpred[i,:] for i in s])
  if return_arg:
        return ordered
  else:
    return [chars[int(i-1)] for i in ordered[:,0]]

def bb_sorter(dpred,chars):
    ysort=ymin_sorter(dpred,chars,return_arg=True)
    x_sort=xmin_sorter(dpred,chars=chars,return_arg=True)
    #print(ysort[:,-1]>ysort[-1,-3])
    if x_sort[0,-3]>x_sort[-1,-3]:
        return xmin_sorter(dpred,chars=chars)
    
    if not((ysort[:,-1]>ysort[-1,-3]).all()):
        return xmin_sorter(ysort[ysort[:,-1]<ysort[-1,-3]],chars)+xmin_sorter(ysort[ysort[:,-1]>ysort[-1,-3]],chars)
    else:
        return xmin_sorter(dpred,chars)
