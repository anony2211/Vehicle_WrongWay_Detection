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
'''


from vehicledetector import VehicleDetector
from keras import backend as K
import cv2
import numpy as np
from keras.preprocessing.image import img_to_array,load_img
from matplotlib import pyplot as plt
from centroidtracker import CentroidTracker
import time

K.clear_session()
img_height=640
img_width=384
write=False ## write video output

if write:
  video_output_path='output.mp4'
  fourcc = cv2.VideoWriter_fourcc(*'mp4v') # Be sure to use lower case
  video = cv2.VideoWriter(video_output_path, fourcc, 20.0, (img_height,img_width))


ct = CentroidTracker(3)
videopath='test.mkv'
cap=cv2.VideoCapture(videopath)
sb=VehicleDetector(modelpath='vehicle_detect.h5',
        size=(img_width,img_height))

## Dicts for storing attributes of centroid's states 

initial_value={} ## stores initial value of centroid object for trajectory tracking
state={} ## stores the state of centroid object in positive or negative integers
count={} ## stores num times a centroid object travels in opposite direction (say towards ymax)


start_time=time.time()
frame_counter=0
def calc_fps():
    elapsed_time=time.time()-start_time
    return frame_counter//elapsed_time



if (cap.isOpened()== False): 

  print("Error opening video stream or file")

while(cap.isOpened()):
  ret, frame = cap.read()
  if ret == False:
    print('End of File or File Error')
    break
  if ret == True:
    frame=cv2.resize(frame,(640,384))
    show_frame=frame.copy()
    blob=cv2.cvtColor(cv2.resize(frame,(img_height,img_width)),cv2.COLOR_BGR2RGB).astype(np.float32)
    dpred=sb.forward(blob,filter_=True,filter_thresh=0.1,iou_threshold=0.1,confidence_thresh=0.9)
    
    rects=[]    ## collects list of rectangle bounding box in the image
    if hasattr(dpred,'shape'): ## check if detections are not empty
      for class_,score,coords in zip(dpred[:,0],dpred[:,1],dpred[:,-4:]):
        xmin,ymin,xmax,ymax=[int(i) for i in coords]
        rects.append([xmin,ymin,xmax,ymax])
        
    objects,object_rects = ct.update(rects) ## update coords of detected vehicles to centroid ID assignment
    try:
      for (objectID, coords) in object_rects.items():
        centroid=[int((coords[0]+coords[2])/2),int((coords[1]+coords[3])/2)]
        if objectID not in state.keys():    ## new entry to state dict
          initial_value[objectID]=centroid[1]
          state[objectID]=initial_value[objectID]
          count[objectID]=0
        else:
          state[objectID]=initial_value[objectID]-centroid[1]  ## update centroid's state
        text = "ID {}".format(objectID)

        '''
        check if state of centroid is non-negative . The negativity is determined by Y-axis of centroid
        if negative, the centroid will be monitored for 10 frames the centroid will turn Orange/Yellow
        in output window. If the state is negative for more than 10 frames , the centroid will be flagged
        as moving in wrong directiona nd bounding box turns Red 
        '''
        if state[objectID]>-20:
          cv2.putText(show_frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 2)
          cv2.circle(show_frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
        if state[objectID]<-20:
          #print(objectID,state[objectID],count[objectID])
          if count[objectID]>10:
            count[objectID]+=1
            cv2.putText(show_frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0, 255), 2)
            cv2.circle(show_frame, (centroid[0], centroid[1]), 4, (0, 0,255), -1)
            cv2.rectangle(show_frame, (coords[0],coords[1]), (coords[2],coords[3]), (0,0,255), 2)
          if count[objectID]<=10:
            cv2.putText(show_frame, text, (centroid[0] - 10, centroid[1] - 10),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,165,255), 2)
            cv2.circle(show_frame, (centroid[0], centroid[1]), 4, (0,165,255), -1)
            cv2.rectangle(show_frame, (coords[0],coords[1]), (coords[2],coords[3]), (0,165,255), 2)
            count[objectID]+=1
    except AttributeError:
         pass
    
    if frame_counter%30==0:
      if frame_counter>0:
        fps=calc_fps()
      else:
        fps='NA'
    cv2.putText(show_frame,'FPS:{}'.format(fps), (20,20),cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,255,0), 2)

    
    cv2.imshow('Output',show_frame)
    frame_counter+=1
    if write:
      video.write(show_frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):

      break
if write:
  video.release()
cap.release()
cv2.destroyAllWindows()

