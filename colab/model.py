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
from keras.models import Model
from keras.layers import Input, Lambda, Conv2D, SeparableConv2D, MaxPooling2D, BatchNormalization, ELU, Reshape, Concatenate, Activation,Add,GlobalAveragePooling2D,Multiply,UpSampling2D
from keras.regularizers import l2
import keras.backend as K
import utils
from keras_layer_AnchorBoxes import AnchorBoxes


#def swish_fn(x_):
#return x_*Activation('sigmoid')(x_)
#return x_*Activation('tanh')(Activation('softplus')(x_))

def get_shape(z):
    if hasattr(z,'_keras_shape'):
        #print(z._keras_shape[-1])
        return z._keras_shape[-1]
    elif hasattr(z,'shape'):
        #if hasattr(x.shape,'value'):
        #print(z.shape[-1].value)
        return z.shape[-1].value

def se(x_):
    y=GlobalAveragePooling2D()(x_)
    y=Reshape(target_shape=(1,1,int(get_shape(y))))(y)
    y=Conv2D(int(get_shape(y)*0.25),(1,1),padding='same')(y)
    y=swish(y)
    y=Conv2D(int(get_shape(x_)),(1,1),padding='same')(y)
    y=Activation('sigmoid')(y)
    y=Multiply()([x_,y])
    y=Conv2D(int(get_shape(y)),(1,1),padding='same')(y)
    y=swish(y)
    return y

#def se(x_):
#    return Lambda(se_fn, output_shape=x_._keras_shape[1:])(x_)
    
def swish(x_):
    #return Lambda(swish_fn, output_shape=x_._keras_shape[1:])(x_)
    #return Multiply()([x_,Activation('tanh')(Activation('softplus')(x_))])
    return Multiply()([x_,Activation('sigmoid')(x_)])
    #return Activation('relu')(x_)

def build_model(image_size,
                n_classes,
                mode='training',
                redux=1.0,
                l2_regularization=0.0,
                min_scale=0.1,
                max_scale=0.9,
                scales=None,
                aspect_ratios_global=[0.5, 1.0, 2.0],
                aspect_ratios_per_layer=None,
                two_boxes_for_ar1=True,
                steps=None,
                offsets=None,
                clip_boxes=False,
                variances=[1.0, 1.0, 1.0, 1.0],
                coords='centroids',
                normalize_coords=False,
                subtract_mean=None,
                divide_by_stddev=None,
                swap_channels=False,
                confidence_thresh=0.01,
                iou_threshold=0.45,
                top_k=200,
                nms_max_output_size=400,
                return_predictor_sizes=False,
                show_flops=False,
                first_stride=2):

    n_predictor_layers = 4 # The number of predictor conv layers in the network
    n_classes += 1 # Account for the background class.
    l2_reg = l2_regularization # Make the internal name shorter.
    img_height, img_width, img_channels = image_size[0], image_size[1], image_size[2]

    ############################################################################
    # Get a few exceptions out of the way.
    ############################################################################

    if aspect_ratios_global is None and aspect_ratios_per_layer is None:
        raise ValueError("`aspect_ratios_global` and `aspect_ratios_per_layer` cannot both be None. At least one needs to be specified.")
    if aspect_ratios_per_layer:
        if len(aspect_ratios_per_layer) != n_predictor_layers:
            raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == {}, but len(aspect_ratios_per_layer) == {}.".format(n_predictor_layers, len(aspect_ratios_per_layer)))

    if (min_scale is None or max_scale is None) and scales is None:
        raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")
    if scales:
        if len(scales) != n_predictor_layers+1:
            raise ValueError("It must be either scales is None or len(scales) == {}, but len(scales) == {}.".format(n_predictor_layers+1, len(scales)))
    else: # If no explicit list of scaling factors was passed, compute the list of scaling factors from `min_scale` and `max_scale`
        scales = np.linspace(min_scale, max_scale, n_predictor_layers+1)

    if len(variances) != 4: # We need one variance value for each of the four box coordinates
        raise ValueError("4 variance values must be pased, but {} values were received.".format(len(variances)))
    variances = np.array(variances)
    if np.any(variances <= 0):
        raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

    if (not (steps is None)) and (len(steps) != n_predictor_layers):
        raise ValueError("You must provide at least one step value per predictor layer.")

    if (not (offsets is None)) and (len(offsets) != n_predictor_layers):
        raise ValueError("You must provide at least one offset value per predictor layer.")

    ############################################################################
    # Compute the anchor box parameters.
    ############################################################################

    # Set the aspect ratios for each predictor layer. These are only needed for the anchor box layers.
    if aspect_ratios_per_layer:
        aspect_ratios = aspect_ratios_per_layer
    else:
        aspect_ratios = [aspect_ratios_global] * n_predictor_layers

    # Compute the number of boxes to be predicted per cell for each predictor layer.
    # We need this so that we know how many channels the predictor layers need to have.
    if aspect_ratios_per_layer:
        n_boxes = []
        for ar in aspect_ratios_per_layer:
            if (1 in ar) & two_boxes_for_ar1:
                n_boxes.append(len(ar) + 1) # +1 for the second box for aspect ratio 1
            else:
                n_boxes.append(len(ar))
    else: # If only a global aspect ratio list was passed, then the number of boxes is the same for each predictor layer
        if (1 in aspect_ratios_global) & two_boxes_for_ar1:
            n_boxes = len(aspect_ratios_global) + 1
        else:
            n_boxes = len(aspect_ratios_global)
        n_boxes = [n_boxes] * n_predictor_layers

    if steps is None:
        steps = [None] * n_predictor_layers
    if offsets is None:
        offsets = [None] * n_predictor_layers

    ############################################################################
    # Define functions for the Lambda layers below.
    ############################################################################

    def identity_layer(tensor):
        return tensor

    def input_mean_normalization(tensor):
        return tensor - np.array(subtract_mean)

    def input_stddev_normalization(tensor):
        return tensor / np.array(divide_by_stddev)

    def input_channel_swap(tensor):
        if len(swap_channels) == 3:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]]], axis=-1)
        elif len(swap_channels) == 4:
            return K.stack([tensor[...,swap_channels[0]], tensor[...,swap_channels[1]], tensor[...,swap_channels[2]], tensor[...,swap_channels[3]]], axis=-1)

    ############################################################################
    # Build the network.
    ############################################################################

    x = Input(shape=(img_height, img_width, img_channels))
    
    conv1 = Conv2D(int(16*redux), (3,3), strides=(first_stride, first_stride), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv1')(x)
    bn1 = BatchNormalization(axis=3, momentum=0.99, name='bn1')(conv1) # Tensorflow uses filter format [filter_height, filter_width, in_channels, out_channels], hence axis = 3
    #conv1 = ELU(name='elu1')(conv1)
    act1=se(swish(bn1))
    #add1=Add()([conv1,act1])
    #bn1_a = BatchNormalization(axis=3, momentum=0.99, name='bn1_a')(add1)
    #add_act1=swish(bn1_a)
    pool1 = MaxPooling2D(pool_size=(2, 2), name='pool1')(act1)
    '''
    conv2 = Conv2D(48, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv2')(add_act1)
    bn2 = BatchNormalization(axis=3, momentum=0.99, name='bn2')(conv2)
    #conv2 = ELU(name='elu2')(conv2)
    act2=Activation('relu')(bn2)
    add2=Add()([conv2,act2])
    bn2_a = BatchNormalization(axis=3, momentum=0.99, name='bn2_a')(add2)
    add_act2=Activation('relu')(bn2_a)
    #pool2 = MaxPooling2D(pool_size=(2, 2), name='pool2')(add_act2)
    '''
    conv3 = Conv2D(int(16*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv3')(pool1)
    #conv3=SeparableConv2D(filters=int(64*redux),kernel_size=(3,3),padding='same',depthwise_initializer='he_normal', pointwise_initializer='he_normal')(pool1)
    bn3 = BatchNormalization(axis=3, momentum=0.99, name='bn3')(conv3)
    #conv3 = ELU(name='elu3')(conv3)
    act3=swish(bn3)
    add3=Add()([pool1,act3])
    bn3_a = BatchNormalization(axis=3, momentum=0.99, name='bn3_a')(add3)
    add_act3=se(swish(bn3_a))
    pool3 = MaxPooling2D(pool_size=(2, 2), name='pool3')(add_act3)
    
    conv4 = Conv2D(int(32*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv4')(pool3)
    #conv4=SeparableConv2D(filters=int(64*redux),kernel_size=(3,3),padding='same',depthwise_initializer='he_normal', pointwise_initializer='he_normal')(pool3)
    bn4 = BatchNormalization(axis=3, momentum=0.99, name='bn4')(conv4)
    #conv4 = ELU(name='elu4')(conv4)
    act4=swish(bn4)
    #add4=Add()([pool3,act4])
    #bn4_a = BatchNormalization(axis=3, momentum=0.99, name='bn4_a')(add4)
    #add_act4=swish(bn4_a)
    act4=se(act4)
    pool4 = MaxPooling2D(pool_size=(2, 2), name='pool4')(act4)

    conv5 = Conv2D(int(32*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv5')(pool4)
    #conv5=SeparableConv2D(filters=int(48*redux),kernel_size=(3,3),padding='same',depthwise_initializer='he_normal', pointwise_initializer='he_normal')(pool4)
    bn5 = BatchNormalization(axis=3, momentum=0.99, name='bn5')(conv5)
    #conv5 = ELU(name='elu5')(conv5)
    act5=swish(bn5)
    add5=Add()([pool4,act5])
    bn5_a = BatchNormalization(axis=3, momentum=0.99, name='bn5_a')(add5)
    add_act5=swish(bn5_a)
    add_act5=se(add_act5)
    pool5 = MaxPooling2D(pool_size=(2, 2), name='pool5')(add_act5)

    conv6 = Conv2D(int(64*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv6')(pool5)
    #conv6=SeparableConv2D(filters=int(48*redux),kernel_size=(3,3),padding='same',depthwise_initializer='he_normal', pointwise_initializer='he_normal')(pool5)
    bn6 = BatchNormalization(axis=3, momentum=0.99, name='bn6')(conv6)
    #conv6 = ELU(name='elu6')(conv6)
    act6=swish(bn6)
    #add6=Add()([pool5,act6])
    #bn6_a = BatchNormalization(axis=3, momentum=0.99, name='bn6_a')(add6)
    #add_act6=swish(bn6_a)
    act6=se(act6)
    pool6 = MaxPooling2D(pool_size=(2, 2), name='pool6')(act6)

    conv7 = Conv2D(int(64*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='conv7')(pool6)
    #conv7=SeparableConv2D(filters=int(32*redux),kernel_size=(3,3),padding='same',depthwise_initializer='he_normal', pointwise_initializer='he_normal')(pool6)
    bn7 = BatchNormalization(axis=3, momentum=0.99, name='bn7')(conv7)
    #conv7 = ELU(name='elu7')(conv7)
    act7=swish(bn7)
    add7=Add()([pool6,act7])
    bn7_a = BatchNormalization(axis=3, momentum=0.99, name='bn7_a')(add7)
    add_act7=swish(bn7_a)
    add_act7=se(add_act7)
    resize_add_act7=UpSampling2D()(add_act7)
    resize_conv7 = Conv2D(int(64*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='resize_conv7')(resize_add_act7)
    resize_bn7 = BatchNormalization(axis=3, momentum=0.99, name='resize_bn7')(resize_conv7)
    resize_bn7=swish(resize_bn7)
    resize_bn7=se(resize_bn7)

    resize_add7=Add()([act6,resize_bn7])
    resize_add_act6=UpSampling2D()(resize_add7)
    resize_conv6 = Conv2D(int(32*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='resize_conv6')(resize_add_act6)
    resize_bn6 = BatchNormalization(axis=3, momentum=0.99, name='resize_bn6')(resize_conv6)
    resize_bn6=swish(resize_bn6)
    resize_bn6=se(resize_bn6)

    resize_add6=Add()([add_act5,resize_bn6])
    resize_add_act5=UpSampling2D()(resize_add6)
    resize_conv5= Conv2D(int(32*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='resize_conv5')(resize_add_act5)
    resize_bn5= BatchNormalization(axis=3, momentum=0.99, name='resize_bn5')(resize_conv5)
    resize_bn5=swish(resize_bn5)
    resize_bn5=se(resize_bn5)
    
    resize_add5=Add()([act4,resize_bn5])
    resize_add_act4=UpSampling2D()(resize_add5)
    resize_conv4= Conv2D(int(16*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='resize_conv4')(resize_add_act4)
    resize_bn4= BatchNormalization(axis=3, momentum=0.99, name='resize_bn4')(resize_conv4)
    resize_bn4=swish(resize_bn4)
    resize_bn4=se(resize_bn4)
    
    '''
    resize_add4=Add()([add_act3,resize_bn4])
    resize_add_act3=UpSampling2D()(resize_add4)
    resize_conv3= Conv2D(int(16*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='resize_conv3')(resize_add_act3)
    resize_bn3= BatchNormalization(axis=3, momentum=0.99, name='resize_bn3')(resize_conv3)
    resize_bn3=swish(resize_bn3)
    resize_bn3=se(resize_bn3)

    resize_add3=Add()([act1,resize_bn3])
    resize_add_act2=UpSampling2D()(resize_add3)
    resize_conv2= Conv2D(int(8*redux), (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='resize_conv2')(resize_add_act2)
    resize_bn2= BatchNormalization(axis=3, momentum=0.99, name='resize_bn2')(resize_conv2)
    resize_bn2=swish(resize_bn2)
    resize_bn2=se(resize_bn2)

    resize_conv1= Conv2D(1, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='resize_conv1')(resize_bn2)
    resize_bn1= BatchNormalization(axis=3, momentum=0.99, name='resize_bn1')(resize_conv1)
    resize_out=Activation('sigmoid')(resize_bn1)
    
    '''

    classes4 = SeparableConv2D(n_boxes[0] * n_classes, (3, 3), strides=(1, 1), padding="same",depthwise_initializer='he_normal', pointwise_initializer='he_normal',name='classes4')(resize_bn4)
    classes5 = SeparableConv2D(n_boxes[1] * n_classes, (3, 3), strides=(1, 1), padding="same",depthwise_initializer='he_normal', pointwise_initializer='he_normal', name='classes5')(resize_bn5)
    classes6 = SeparableConv2D(n_boxes[2] * n_classes, (3, 3), strides=(1, 1), padding="same",depthwise_initializer='he_normal', pointwise_initializer='he_normal', name='classes6')(resize_bn6)
    classes7 = SeparableConv2D(n_boxes[3] * n_classes, (3, 3), strides=(1, 1), padding="same",depthwise_initializer='he_normal', pointwise_initializer='he_normal', name='classes7')(resize_bn7)

    
    # Output shape of `boxes`: `(batch, height, width, n_boxes * 4)`
    boxes4 = Conv2D(n_boxes[0] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes4')(resize_bn4)
    boxes5 = Conv2D(n_boxes[1] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes5')(resize_bn5)
    boxes6 = Conv2D(n_boxes[2] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes6')(resize_bn6)
    boxes7 = Conv2D(n_boxes[3] * 4, (3, 3), strides=(1, 1), padding="same", kernel_initializer='he_normal', kernel_regularizer=l2(l2_reg), name='boxes7')(resize_bn7)

    # Generate the anchor boxes
    # Output shape of `anchors`: `(batch, height, width, n_boxes, 8)`
    anchors4 = AnchorBoxes(img_height, img_width, this_scale=scales[0], next_scale=scales[1], aspect_ratios=aspect_ratios[0],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[0], this_offsets=offsets[0],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors4')(resize_bn4)
    anchors5 = AnchorBoxes(img_height, img_width, this_scale=scales[1], next_scale=scales[2], aspect_ratios=aspect_ratios[1],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[1], this_offsets=offsets[1],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors5')(resize_bn5)
    anchors6 = AnchorBoxes(img_height, img_width, this_scale=scales[2], next_scale=scales[3], aspect_ratios=aspect_ratios[2],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[2], this_offsets=offsets[2],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors6')(resize_bn6)
    anchors7 = AnchorBoxes(img_height, img_width, this_scale=scales[3], next_scale=scales[4], aspect_ratios=aspect_ratios[3],
                           two_boxes_for_ar1=two_boxes_for_ar1, this_steps=steps[3], this_offsets=offsets[3],
                           clip_boxes=clip_boxes, variances=variances, coords=coords, normalize_coords=normalize_coords, name='anchors7')(resize_bn7)

    # Reshape the class predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, n_classes)`
    # We want the classes isolated in the last axis to perform softmax on them
    classes4_reshaped = Reshape((-1, n_classes), name='classes4_reshape')(classes4)
    classes5_reshaped = Reshape((-1, n_classes), name='classes5_reshape')(classes5)
    classes6_reshaped = Reshape((-1, n_classes), name='classes6_reshape')(classes6)
    classes7_reshaped = Reshape((-1, n_classes), name='classes7_reshape')(classes7)
    # Reshape the box coordinate predictions, yielding 3D tensors of shape `(batch, height * width * n_boxes, 4)`
    # We want the four box coordinates isolated in the last axis to compute the smooth L1 loss
    boxes4_reshaped = Reshape((-1, 4), name='boxes4_reshape')(boxes4)
    boxes5_reshaped = Reshape((-1, 4), name='boxes5_reshape')(boxes5)
    boxes6_reshaped = Reshape((-1, 4), name='boxes6_reshape')(boxes6)
    boxes7_reshaped = Reshape((-1, 4), name='boxes7_reshape')(boxes7)
    # Reshape the anchor box tensors, yielding 3D tensors of shape `(batch, height * width * n_boxes, 8)`
    anchors4_reshaped = Reshape((-1, 8), name='anchors4_reshape')(anchors4)
    anchors5_reshaped = Reshape((-1, 8), name='anchors5_reshape')(anchors5)
    anchors6_reshaped = Reshape((-1, 8), name='anchors6_reshape')(anchors6)
    anchors7_reshaped = Reshape((-1, 8), name='anchors7_reshape')(anchors7)

    # Concatenate the predictions from the different layers and the assosciated anchor box tensors
    # Axis 0 (batch) and axis 2 (n_classes or 4, respectively) are identical for all layer predictions,
    # so we want to concatenate along axis 1
    # Output shape of `classes_concat`: (batch, n_boxes_total, n_classes)
    classes_concat = Concatenate(axis=1, name='classes_concat')([classes4_reshaped,
                                                                 classes5_reshaped,
                                                                 classes6_reshaped,
                                                                 classes7_reshaped])

    # Output shape of `boxes_concat`: (batch, n_boxes_total, 4)
    boxes_concat = Concatenate(axis=1, name='boxes_concat')([boxes4_reshaped,
                                                             boxes5_reshaped,
                                                             boxes6_reshaped,
                                                             boxes7_reshaped])

    # Output shape of `anchors_concat`: (batch, n_boxes_total, 8)
    anchors_concat = Concatenate(axis=1, name='anchors_concat')([anchors4_reshaped,
                                                                 anchors5_reshaped,
                                                                 anchors6_reshaped,
                                                                 anchors7_reshaped])

    # The box coordinate predictions will go into the loss function just the way they are,
    # but for the class predictions, we'll apply a softmax activation layer first
    classes_softmax = Activation('softmax', name='classes_softmax')(classes_concat)

    # Concatenate the class and box coordinate predictions and the anchors to one large predictions tensor
    # Output shape of `predictions`: (batch, n_boxes_total, n_classes + 4 + 8)
    predictions = Concatenate(axis=2, name='predictions')([classes_softmax, boxes_concat, anchors_concat])

    
    model = Model(inputs=x, outputs=predictions)

    if show_flops:
        utils.show_flops(model,image_size)

    
    return model




