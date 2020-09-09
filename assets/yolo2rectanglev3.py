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

import os
import cv2
from matplotlib import pyplot as plt
from glob import glob
import numpy as np

def yolo2rectangle(x_n,y_n,w_n,h_n,x_height=384,y_height=384):
    #0.923047 0.481250 0.086719 0.445833

    x=(x_n*x_height) 
    y=(y_n*y_height) 
    w=w_n*x_height
    h=h_n*y_height

    #print(x,y,w,h)

    x1_,y1_= x-(w/2),y-(h/2)
    x2_,y2_= x+(w/2),y+(h/2)


    return(int(x1_), int(y1_), int(x2_), int(y2_))

def yolotxt2rectangle(file_,plot_image=False,x_height=None,y_height=None,last_class=None):
    #print(file_)
    #yolotxt2rectangle('/home/vignesh/vk/intozi/ocr_detection/colab/img_hrithik/HR26BH3775.txt',y_height=128,x_height=384,last_class=36)
    assert(x_height!=None),'Invalid x_height dimension '+x_height
    assert(y_height!=None),'Invalid y_height dimension '+y_height
    assert(last_class!=None),'last class must be int but found '+y_height
    if plot_image:
        assert os.path.exists(file_.strip('.txt')+'.jpg'),'invalid_path:'+file_.strip('.txt')+'.jpg'
        img=cv2.imread(file_.strip('.txt')+'.jpg')
        img=cv2.resize(img,(x_height,y_height))
        
    with open(file_,'r') as f:
        a=f.read()
    splits=a.split('\n')
    if splits[-1]=='':
        splits.pop()
        
    data=np.empty(shape=(len(splits),5))
    
    for i in range(len(splits)):
        
        for index,split in enumerate(splits[i].split(' ')):
            data[i,index]=float(split)

    rects=[]
    for data_ in data:
      (x1,y1,x2,y2)=(yolo2rectangle(data_[1],data_[2],data_[3],data_[4],x_height=x_height,y_height=y_height))
      rects.append((x1,y1,x2,y2))
      if plot_image:
        cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
    if plot_image:
      plt.imshow(img)
      plt.show()
    rects=np.array(rects)
    '''
    #usage:
    for file_ in sorted(glob('data/*.txt')):
        yolotxt2rectangle(file_,True)
    returns classes,boxes
    '''
    data[:,1:]=rects
    np.place(data[:,0],data[:,0]==0,last_class)
    np.place(data[:,1],data[:,1]>=x_height,x_height)
    np.place(data[:,3],data[:,3]>=x_height,x_height)
    np.place(data[:,2],data[:,2]>=y_height,y_height)
    np.place(data[:,4],data[:,4]>=y_height,y_height)
    data[data<0]=0
    #data[data>x_height]=x_height
    return data

'''
#usage:

x_scale,y_scale=384,384
#img=cv2.imread('/home/vignesh/vk/intozi/yolo/ccd.jpg') 2 0.577734 0.401389 0.350781 0.222222
img=cv2.imread('data/2.jpg')
img=cv2.resize(img,(384,384))

#0.923047 0.481250 0.086719 0.445833

x_n= 0.923047
y_n= 0.481250
w_n= 0.086719
h_n= 0.445833

(x1,y1),(x2,y2)=yolo2rectangle(x_n,y_n,w_n,h_n,x_scale,y_scale)
#318,127,413,227
cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
plt.imshow(img)
plt.show()
'''
