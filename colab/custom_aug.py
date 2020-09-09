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

Custom Augmentations
'''

import numpy as np
import cv2
import albumentations as A
#aug=A.GaussNoise()

def hshift(img,box,percentage):

    ##img --> (192,384,3)
    #print('hthresh:',percentage)
    backup_box=np.copy(box)
    assert(0.0<percentage<1.0)
    max_h=img.shape[-2]
    max_w=img.shape[-3]
    class_,coords=box[:,0],box[:,-4:]
    h_val=int(percentage*max_h)
    #print(img.shape)
    roi=np.zeros_like(img)
    
    if np.random.randint(0,10,size=(1,))%2==0:
        #print('right')
        roi[:,h_val:max_h]=img[:,0:max_h-h_val]
        box[:,1]+=h_val
        box[:,3]+=h_val
    
        
    else:
        #print('left')
        roi[:,0:max_h-h_val]=img[:,h_val:max_h]
        box[:,1]-=h_val
        box[:,3]-=h_val
    
    
    np.place(box[:,1],box[:,1]>=max_h,max_h)
    np.place(box[:,3],box[:,3]>=max_h,max_h)
    np.place(box[:,1],box[:,1]<=0,0)
    np.place(box[:,3],box[:,3]<=0,0)
    
    refined_box=[]
    counter=0
    for xmin,xmax in zip(box[:,1],box[:,3]):
        if xmax-xmin>10:
            refined_box.append(box[counter,:])
        counter+=1
               
    if len(np.array(refined_box))<1:
      #print('degenerate situation sending backup')
      return img,np.array(backup_box)
    else:     
      return roi,np.array(refined_box)


def vshift(img,box,percentage):
    #print('vthresh:',percentage)

    ##img --> (192,384,3)
    backup_box=np.copy(box)
    assert(0.0<percentage<1.0)
    max_h=img.shape[-2]
    max_w=img.shape[-3]
    class_,coords=box[:,0],box[:,-4:]
    x_val=int(percentage*max_w)
    #print(img.shape)
    roi=np.zeros_like(img)
    
    if np.random.randint(0,10,size=(1,))%2==0:
        #print('down')
        roi[x_val:max_w,:]=img[0:max_w-x_val,:]
        box[:,2]+=x_val
        box[:,4]+=x_val
    
        
    else:
        #print('up')
        roi[0:max_w-x_val,:]=img[x_val:max_w,:]
        box[:,2]-=x_val
        box[:,4]-=x_val
    
    
    np.place(box[:,2],box[:,2]>=max_w,max_w)
    np.place(box[:,4],box[:,4]>=max_w,max_w)
    np.place(box[:,2],box[:,2]<=0,0)
    np.place(box[:,4],box[:,4]<=0,0)
    
    refined_box=[]
    counter=0
    for ymin,ymax in zip(box[:,2],box[:,4]):
        if ymax-ymin>10:
            refined_box.append(box[counter,:])
        counter+=1
               
    if len(np.array(refined_box))<1:
      #print('degenerate situation sending backup')
      return img,np.array(backup_box)
    else:     
      return roi,np.array(refined_box)


def add_gauss(img,div=0,mean=0,devlimit=200,verbose=False):
    if div==0:
        chance=1
        div=100
    else:
      chance=np.random.randint(0,10,size=(1,))
    if chance%div==0:
        std=np.random.randint(50,devlimit,size=(1,))
        gauss=np.random.normal(loc=mean,scale=std,size=img.shape)
        if verbose:
            print(chance,div,std,gauss.min(),gauss.max())
        gauss[gauss<0]=0
        gauss[gauss>=255]=255
        aug_img=aug.apply(img,gauss=gauss)
        return aug_img
    else:
        return img


'''

thresh=np.random.uniform(low=0.05,high=0.3,size=(1,))
file_=random.choice(glob('data/*.txt'))
#img=cv2.imread(file_[:-4]+'.jpg')
#img=cv2.resize(img,(192,384))
img_width,img_height=192,384
img=img_to_array(load_img(path=file_.strip('.txt')+'.jpg',target_size=(img_width,img_height),color_mode='rgb')).astype(np.uint8)
box=yolotxt2rectangle(file_,plot_image=False,y_height=img_width,x_height=img_height)
img,dpred=hshift(img,thresh,box)

for class_,coords in zip(dpred[:,0],dpred[:,-4:]):
    xmin,ymin,xmax,ymax=[int(i) for i in coords]
    print(class_,xmin,ymin,xmax,ymax)
    cv2.rectangle(img, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,0,255), 2)
    #plt.title(str(sorter(dpred)))
  
plt.imshow(img)
plt.show()


img=img_to_array(load_img(path=file_.strip('.txt')+'.jpg',target_size=(img_width,img_height),color_mode='rgb')).astype(np.uint8)
box=yolotxt2rectangle(file_,plot_image=False,y_height=img_width,x_height=img_height)
img,dpred=vshift(img,thresh,box)

for class_,coords in zip(dpred[:,0],dpred[:,-4:]):
    xmin,ymin,xmax,ymax=[int(i) for i in coords]
    print(class_,xmin,ymin,xmax,ymax)
    cv2.rectangle(img, (int(xmin),int(ymin)), (int(xmax),int(ymax)), (0,0,255), 2)
    #plt.title(str(sorter(dpred)))
  
plt.imshow(img)
plt.show()

'''
    
        
        
    
