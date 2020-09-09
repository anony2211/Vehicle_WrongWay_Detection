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

import numpy as np

###ssd pred to yolotxt converter###

def convert(dpred,x_height=1.0,y_height=1.0):
    xmin=dpred[:,2]
    ymin=dpred[:,3]
    xmax=dpred[:,4]
    ymax=dpred[:,5]

    h=(ymax-ymin)/y_height
    w=(xmax-xmin)/x_height
    x=((xmin+xmax)/2)/x_height
    y=((ymin+ymax)/2)/y_height

    yolobox=np.copy(dpred)
    yolobox[:,0]=dpred[:,0].astype(np.int32)
    yolobox[:,2]=x
    yolobox[:,3]=y
    yolobox[:,4]=w
    yolobox[:,5]=h
    return yolobox
    
