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

'''
Load the keras SSD detection model and give out the filtered
bounding box predictions
'''

import keras
import numpy as np
from keras.models import load_model
from assets.keras_layer_AnchorBoxes import AnchorBoxes
from assets.ssd_output_decoder import decode_detections
from assets.utils import box_filter




class VehicleDetector():
    def __init__(self,modelpath,size):
        self.modelpath=modelpath
        self.model=self.load(modelpath)
        self.image_size=size
        self.img_height=self.image_size[0]
        self.img_width=self.image_size[1]
        self.normalize_coords=True
        
        
    def temp(self,x,y):
        return keras.losses.binary_crossentropy(x,y)
    
    def load(self,modelpath):
        return load_model(self.modelpath,custom_objects={'AnchorBoxes': AnchorBoxes,
                                               'compute_loss': self.temp})
    def forward(self,img,filter_=True,filter_thresh=0.3,confidence_thresh=0.65,iou_threshold=0.2,return_boxes=False):
        pred=self.model.predict(np.expand_dims(img,axis=0))
        self.dpred=decode_detections(pred,confidence_thresh=confidence_thresh,iou_threshold=iou_threshold,top_k=200,input_coords='centroids',normalize_coords=self.normalize_coords,img_height=self.img_height,img_width=self.img_width)[0]
        if filter_:
            self.dpred=box_filter(self.dpred,filter_thresh)
   
        if len(self.dpred.shape)>1:
            return self.dpred
                
        else:
            return 'No vehicle found'

    
