
from __future__ import print_function

from keras import backend as K
  
from keras.models import *
from keras.layers import *

## 2D MODELS
def u_net_both(shape, nb_filters_0=4, exp=1, conv_size=3, initialization='glorot_uniform', activation="relu", output_channels=4, augmentation='Gaussian',rate_augmentation=.1):
   """U-Net model.

   Standard U-Net model, plus optional gaussian noise.
   Note that the dimensions of the input images should be
   multiples of 32.

   Arguments:
   shape: image shape, in the format (nb_channels, x_size, y_size).
   nb_filters_0 : initial number of filters in the convolutional layer.
   exp : should be equal to 0 or 1. Indicates if the number of layers should be constant (0) or increase exponentially (1).
   conv_size : size of convolution.
   initialization: initialization of the convolutional layers.
   activation: activation of the convolutional layers.
   sigma_noise: standard deviation of the gaussian noise layer. If equal to zero, this layer is deactivated.
   output_channels: number of output channels.

   Returns:
   U-Net model - it still needs to be compiled.

   Reference:
   U-Net: Convolutional Networks for Biomedical Image Segmentation
   Olaf Ronneberger, Philipp Fischer, Thomas Brox
   MICCAI 2015

   Credits:
   The starting point for the code of this funcions comes from:
   https://github.com/jocicmarko/ultrasound-nerve-segmentation
   by Marko Jocic
   """

   if K.image_data_format() == 'channels_first':
       channel_axis = 1
   else:
       channel_axis = 3

   inputs = Input(shape)
   noise = GaussianNoise(rate_augmentation)(inputs)
   conv1 = Conv2D(nb_filters_0, conv_size, activation=activation,padding='same', kernel_initializer=initialization, name="conv1_1")(noise)
   conv1 = Conv2D(nb_filters_0, conv_size, activation=activation, padding='same', kernel_initializer=initialization, name="conv1_2")(conv1)
   pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
   conv2 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,padding='same', kernel_initializer=initialization, name="conv2_1")(pool1)
   conv2 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,padding='same', kernel_initializer=initialization, name="conv2_2")(conv2)
   pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

   conv3 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,padding='same', kernel_initializer=initialization, name="conv3_1")(pool2)
   conv3 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,padding='same', kernel_initializer=initialization, name="conv3_2")(conv3)
   pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

   conv4 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv4_1")(pool3)
   conv4 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv4_2")(conv4)
   pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

   conv5 = Conv2D(nb_filters_0 * 2**(4 * exp), conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv5_1")(pool4)
   conv5 = Conv2D(nb_filters_0 * 2**(4 * exp), conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv5_2")(conv5)

   up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=channel_axis)
   conv6 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv6_1")(up6)
   conv6 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv6_2")(conv6)

   up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=channel_axis)
   conv7 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv7_1")(up7)
   conv7 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv7_2")(conv7)

   up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=channel_axis)
   conv8 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv8_1")(up8)
   conv8 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv8_2")(conv8)

   up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=channel_axis)
   conv9 = Conv2D(nb_filters_0, conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv9_1")(up9)
   conv9 = Conv2D(nb_filters_0, conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv9_2")(conv9)


   conv10 = Conv2D(output_channels, 1, activation='sigmoid', name="conv_out")(conv9)
    
   up6_2 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=channel_axis)
   conv6_2 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv6_1_2")(up6_2)
   conv6_2 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv6_2_2")(conv6_2)

   up7_2 = concatenate([UpSampling2D(size=(2, 2))(conv6_2), conv3], axis=channel_axis)
   conv7_2 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv7_1_2")(up7_2)
   conv7_2 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv7_2_2")(conv7_2)

   up8_2 = concatenate([UpSampling2D(size=(2, 2))(conv7_2), conv2], axis=channel_axis)
   conv8_2 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv8_1_2")(up8_2)
   conv8_2 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv8_2_2")(conv8_2)

   up9_2 = concatenate([UpSampling2D(size=(2, 2))(conv8_2), conv1], axis=channel_axis)
   conv9_2 = Conv2D(nb_filters_0, conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv9_1_2")(up9_2)
   conv9_2 = Conv2D(nb_filters_0, conv_size, activation=activation,
                  padding='same', kernel_initializer=initialization, name="conv9_2_2")(conv9_2)


   conv10_2 = Conv2D(output_channels, 1, activation='sigmoid', name="conv_out_2")(conv9_2)

   return Model(inputs, outputs=[conv10,conv10_2])


def u_net(shape, nb_filters_0=4, exp=1, conv_size=3, initialization='glorot_uniform', activation="relu", output_channels=4, rate_augmentation=0.1):
    """U-Net model.

    Standard U-Net model, plus optional gaussian noise.
    Note that the dimensions of the input images should be
    multiples of 32.

    Arguments:
    shape: image shape, in the format (nb_channels, x_size, y_size).
    nb_filters_0 : initial number of filters in the convolutional layer.
    exp : should be equal to 0 or 1. Indicates if the number of layers should be constant (0) or increase exponentially (1).
    conv_size : size of convolution.
    initialization: initialization of the convolutional layers.
    activation: activation of the convolutional layers.
    sigma_noise: standard deviation of the gaussian noise layer. If equal to zero, this layer is deactivated.
    output_channels: number of output channels.

    Returns:
    U-Net model - it still needs to be compiled.

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    MICCAI 2015

    Credits:
    The starting point for the code of this funcions comes from:
    https://github.com/jocicmarko/ultrasound-nerve-segmentation
    by Marko Jocic
    """

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3

    inputs = Input(shape)
    noise = GaussianNoise(rate_augmentation)(inputs)
    conv1 = Conv2D(nb_filters_0, conv_size, activation=activation,padding='same', kernel_initializer=initialization, name="conv1_1")(noise)
    conv1 = Conv2D(nb_filters_0, conv_size, activation=activation, padding='same', kernel_initializer=initialization, name="conv1_2")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,padding='same', kernel_initializer=initialization, name="conv2_1")(pool1)
    conv2 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,padding='same', kernel_initializer=initialization, name="conv2_2")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,padding='same', kernel_initializer=initialization, name="conv3_1")(pool2)
    conv3 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,padding='same', kernel_initializer=initialization, name="conv3_2")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv4_1")(pool3)
    conv4 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv4_2")(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(nb_filters_0 * 2**(4 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv5_1")(pool4)
    conv5 = Conv2D(nb_filters_0 * 2**(4 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv5_2")(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4], axis=channel_axis)
    conv6 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv6_1")(up6)
    conv6 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv6_2")(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3], axis=channel_axis)
    conv7 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv7_1")(up7)
    conv7 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv7_2")(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2], axis=channel_axis)
    conv8 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv8_1")(up8)
    conv8 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv8_2")(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1], axis=channel_axis)
    conv9 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv9_1")(up9)
    conv9 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv9_2")(conv9)


    conv10 = Conv2D(output_channels, 1, activation='sigmoid', name="conv_out")(conv9)

    return Model(inputs, conv10)
    

def u_net6(shape, filters=4,int_space=4,output_channels=4,rate=.3,conv_size=3):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    inputs = Input(shape)
    inputN = GaussianNoise(rate)(inputs)
    pool1 = conv2dLeakyDownProj(inputN, filters,int_space, f_size = conv_size)
    pool1 = Dropout(rate)(pool1)
    pool2 = conv2dLeakyDownProj(pool1, filters,int_space, f_size = conv_size)
    pool2 = Dropout(rate)(pool2)
    pool3 = conv2dLeakyDownProj(pool2, filters,int_space, f_size = conv_size)
    pool3 = Dropout(rate)(pool3)
    pool4 = conv2dLeakyDownProj(pool3, filters,int_space, f_size = conv_size)
    pool4 = Dropout(rate)(pool4)
    pool5 = conv2dLeakyDownProj(pool4, filters,int_space, f_size = conv_size)
    pool5 = Dropout(rate)(pool5)
    up4 = concatenate([UpSampling2D(size=(2, 2))(pool5), pool4], axis=channel_axis)
    conv4 =conv2dLeakyProj(up4,filters,int_space, f_size = conv_size)
    conv4 =Dropout(rate)(conv4)
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv4), pool3], axis=channel_axis)
    conv3 =conv2dLeakyProj(up3,filters,int_space, f_size = conv_size)
    conv3 =Dropout(rate)(conv3)
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv3), pool2], axis=channel_axis)
    conv2 =conv2dLeakyProj(up2,filters,int_space, f_size = conv_size)
    conv2 =Dropout(rate)(conv2)
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv2), pool1], axis=channel_axis)
    conv1 =conv2dLeakyProj(up1,filters,int_space, f_size = conv_size)
    conv1 =Dropout(rate)(conv1)
    up0 = concatenate([UpSampling2D(size=(2, 2))(conv1), inputs], axis=channel_axis)
    conv0 = conv2dLeaky(up0,output_channels, f_size = conv_size)
    #conv0  = GaussianNoise(rate)(conv0)
    return Model(inputs, conv0)

    

def conv2dLeakyDownProj(layer_input, filters, output,f_size=3):
    d = Conv2D(filters, kernel_size=f_size, strides=2,padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(output, kernel_size=1,padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    return d

def conv2dLeaky(layer_input, filters, f_size=3):
    d = Conv2D(filters, kernel_size=f_size,padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    return d

def conv2dLeakyProj(layer_input, filters,output, f_size=3):
    d = Conv2D(filters, kernel_size=f_size,padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv2D(output, kernel_size=1,padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    return d

def u_net6_both(shape, filters=4,int_space=4,output_channels=4,rate=.3):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    inputs = Input(shape)
    #inputN = GaussianNoise(rate)(inputs)
    pool1 = conv2dLeakyDownProj(inputs, filters,int_space)
    pool1 = Dropout(rate)(pool1)
    pool2 = conv2dLeakyDownProj(pool1, filters,int_space)
    pool2 = Dropout(rate)(pool2)
    pool3 = conv2dLeakyDownProj(pool2, filters,int_space)
    pool3 = Dropout(rate)(pool3)
    pool4 = conv2dLeakyDownProj(pool3, filters,int_space)
    pool4 = Dropout(rate)(pool4)
    pool5 = conv2dLeakyDownProj(pool4, filters,int_space)
    pool5 = Dropout(rate)(pool5)
    up4 = concatenate([UpSampling2D(size=(2, 2))(pool5), pool4], axis=channel_axis)
    conv4 =conv2dLeakyProj(up4,filters,int_space)
    conv4 =Dropout(rate)(conv4)
    up3 = concatenate([UpSampling2D(size=(2, 2))(conv4), pool3], axis=channel_axis)
    conv3 =conv2dLeakyProj(up3,filters,int_space)
    conv3 =Dropout(rate)(conv3)
    up2 = concatenate([UpSampling2D(size=(2, 2))(conv3), pool2], axis=channel_axis)
    conv2 =conv2dLeakyProj(up2,filters,int_space)
    conv2 =Dropout(rate)(conv2)
    up1 = concatenate([UpSampling2D(size=(2, 2))(conv2), pool1], axis=channel_axis)
    conv1 =conv2dLeakyProj(up1,filters,int_space)
    conv1 =Dropout(rate)(conv1)
    up0 = concatenate([UpSampling2D(size=(2, 2))(conv1), inputs], axis=channel_axis)
    conv0 = conv2dLeaky(up0,output_channels)

    #up4 = concatenate([UpSampling2D(size=(2, 2))(pool5), pool4], axis=channel_axis)
    conv4_2 =conv2dLeakyProj(up4,filters,int_space)
    conv4_2 =Dropout(rate)(conv4_2)
    up3_2 = concatenate([UpSampling2D(size=(2, 2))(conv4_2), pool3], axis=channel_axis)
    conv3_2 =conv2dLeakyProj(up3_2,filters,int_space)
    conv3_2 =Dropout(rate)(conv3)
    up2_2 = concatenate([UpSampling2D(size=(2, 2))(conv3_2), pool2], axis=channel_axis)
    conv2_2 =conv2dLeakyProj(up2,filters,int_space)
    conv2_2 =Dropout(rate)(conv2_2)
    up1_2 = concatenate([UpSampling2D(size=(2, 2))(conv2_2), pool1], axis=channel_axis)
    conv1_2 =conv2dLeakyProj(up1,filters,int_space)
    conv1_2 =Dropout(rate)(conv1_2)
    up0_2 = concatenate([UpSampling2D(size=(2, 2))(conv1_2), inputs], axis=channel_axis)
    conv0_2 = conv2dLeaky(up0_2,output_channels)


    #conv0  = GaussianNoise(rate)(conv0)
    return Model(inputs, outputs=[conv0,conv0_2])


def u_net3(shape, nb_filters_0=32, exp=1, conv_size=3, initialization='glorot_uniform', activation="relu", sigma_noise=0, output_channels=1):
    """U-Net model, with three layers.

    U-Net model using 3 maxpoolings/upsamplings, plus optional gaussian noise.

    Arguments:
    shape: image shape, in the format (nb_channels, x_size, y_size).
    nb_filters_0 : initial number of filters in the convolutional layer.
    exp : should be equal to 0 or 1. Indicates if the number of layers should be constant (0) or increase exponentially (1).
    conv_size : size of convolution.
    initialization: initialization of the convolutional layers.
    activation: activation of the convolutional layers.
    sigma_noise: standard deviation of the gaussian noise layer. If equal to zero, this layer is deactivated.
    output_channels: number of output channels.

    Returns:
    U-Net model - it still needs to be compiled.

    Reference:
    U-Net: Convolutional Networks for Biomedical Image Segmentation
    Olaf Ronneberger, Philipp Fischer, Thomas Brox
    MICCAI 2015

    Credits:
    The starting point for the code of this funcions comes from:
    https://github.com/jocicmarko/ultrasound-nerve-segmentation
    by Marko Jocic
    """

    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
        
    inputs = Input(shape)
    conv1 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv1_1")(inputs)
    conv1 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv1_2")(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv2_1")(pool1)
    conv2 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv2_2")(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv3_1")(pool2)
    conv3 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv3_2")(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv4_1")(pool3)
    conv4 = Conv2D(nb_filters_0 * 2**(3 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv4_2")(conv4)

    up5 = concatenate(
        [UpSampling2D(size=(2, 2))(conv4), conv3], axis=channel_axis)
    conv5 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv5_1")(up5)
    conv5 = Conv2D(nb_filters_0 * 2**(2 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv5_2")(conv5)

    up6 = concatenate(
        [UpSampling2D(size=(2, 2))(conv5), conv2], axis=channel_axis)
    conv6 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv6_1")(up6)
    conv6 = Conv2D(nb_filters_0 * 2**(1 * exp), conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv6_2")(conv6)

    up7 = concatenate(
        [UpSampling2D(size=(2, 2))(conv6), conv1], axis=channel_axis)
    conv7 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv7_1")(up7)
    conv7 = Conv2D(nb_filters_0, conv_size, activation=activation,
                   padding='same', kernel_initializer=initialization, name="conv7_2")(conv7)

    if sigma_noise > 0:
        conv7 = GaussianNoise(sigma_noise)(conv7)

    conv10 = Conv2D(output_channels, 1, activation='sigmoid', name="conv_out")(conv7)

    return Model(inputs, conv10)
        


# LOSS FUNCTIONS
import tensorflow as tf
def mask_jaccard_loss_mean_smooth(y_true, y_pred, smooth=10):
    """
    Mean Jaccard Loss per classes.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    """
    y_true=K.clip(y_true,0,1)
    y_pred=K.clip(y_pred,0,1)
    print(y_true.shape)
    mask = K.sum(y_true, axis=[0,1])
    full_mask = tf.stack([mask, mask, mask, mask, mask]) 
    
    #print("SHAPE",y_true.shape)
    intersection = y_true * y_pred 
    union = (y_true + y_pred - intersection)* full_mask
    union = K.sum(union,axis=[3,2])
    intersection = K.sum(intersection,axis=[3,2])
    jac = (intersection + smooth) / (union + (smooth))
    #print('Jacc Shape',jac.shape)
    jac = K.mean(jac,axis=1)
    return 1-jac

def jaccard_loss_mean_smooth(y_true, y_pred, smooth=10):
    """
    Mean Jaccard Loss per classes.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    """
    y_true=K.clip(y_true,0,1)
    y_pred=K.clip(y_pred,0,1)
    print("SHAPE",y_true.shape)
    intersection = y_true * y_pred
    union= y_true + y_pred - intersection
    union =K.sum(union,axis=[3,2])
    intersection = K.sum(intersection,axis=[3,2])
    jac = (intersection + smooth) / (union + (smooth))
    print('Jacc Shape',jac.shape)
    jac = K.mean(jac,axis=1)
    return 1-jac

def jaccard_loss_mean_smooth_old(y_true, y_pred, smooth=10):
    """
    Mean Jaccard Loss per classes.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    """
    y_true=K.clip(y_true,0,1)
    y_pred=K.clip(y_pred,0,1)
    intersection = y_true * y_pred
    union= y_true + y_pred - intersection
    union =K.sum(union,axis=[3,2])
    intersection = K.sum(intersection,axis=[3,2])
    jac = (intersection + 1) / (union + (smooth))
    jac = K.mean(jac,axis=1)
    return 1-jac

def jaccard_distance(y_true, y_pred, smooth=100):
    """Jaccard distance for semantic segmentation.
    Also known as the intersection-over-union loss.
    This loss is useful when you have unbalanced numbers of pixels within an image
    because it gives all classes equal weight. However, it is not the defacto
    standard for image segmentation.
    For example, assume you are trying to predict if
    each pixel is cat, dog, or background.
    You have 80% background pixels, 10% dog, and 10% cat.
    If the model predicts 100% background
    should it be be 80% right (as with categorical cross entropy)
    or 30% (with this loss)?
    The loss has been modified to have a smooth gradient as it converges on zero.
    This has been shifted so it converges on 0 and is smoothed to avoid exploding
    or disappearing gradient.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    # Arguments
        y_true: The ground truth tensor.
        y_pred: The predicted tensor
        smooth: Smoothing factor. Default is 100.
    # Returns
        The Jaccard distance between the two tensors.
    # References
        - [What is a good evaluation measure for semantic segmentation?](
           http://www.bmva.org/bmvc/2013/Papers/paper0032/paper0032.pdf)
    """
    
    y_true=K.clip(y_true,0,1)
    y_pred=K.clip(y_pred,0,1) 
    intersection = y_true * y_pred
    union= y_true + y_pred - intersection
    union =K.sum(union,axis=[3,2])
    intersection = K.sum(intersection,axis=[3,2])
    jac = (intersection + smooth) / (union + (smooth))

    return (1 - jac) * smooth

def jaccard_pow_coef(y_true, y_pred, smooth = 10):
    p_value = 2.0
    print("p_value", p_value)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    term_true = K.sum(K.pow(y_true_f, p_value))
    term_pred = K.sum(K.pow(y_pred_f, p_value))

    union = term_true + term_pred - intersection
    return (intersection + smooth) / (union + smooth)

def jaccard2_coef(y_true, y_pred, smooth=10):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    union = K.sum(y_true_f * y_true_f) + K.sum(y_pred_f * y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def jaccard2_loss(y_true, y_pred, smooth = 10):
    loss_value = 1 - jaccard_pow_coef(y_true, y_pred, smooth)
    print('shape', loss_value.shape)
    return loss_value


#################################################################################
# 3D MODELS

def conv3dLeakyDownProj(layer_input, filters, output,f_size=3):
    d = Conv3D(filters, kernel_size=f_size, strides=(1,2,2),padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv3D(output, kernel_size=1,padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    return d

def conv3dLeaky(layer_input, filters, f_size=3):
    d = Conv3D(filters, kernel_size=f_size,padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    return d

def conv3dLeakyProj(layer_input, filters,output, f_size=3):
    d = Conv3D(filters, kernel_size=f_size,padding='same')(layer_input)
    d = LeakyReLU(alpha=0.2)(d)
    d = Conv3D(output, kernel_size=1,padding='same')(d)
    d = LeakyReLU(alpha=0.2)(d)
    return d


def u_net6_3D_both(shape, filters=4,int_space=4,output_channels=4,rate=.3,noise=0.5):
    if K.image_data_format() == 'channels_first':
        channel_axis = 1
    else:
        channel_axis = 3
    inputs = Input(shape)
    noise = GaussianNoise(rate)(inputs)
    pool1 = conv3dLeakyDownProj(noise, filters,int_space)
    pool1 = Dropout(rate)(pool1)
    pool2 = conv3dLeakyDownProj(pool1, filters,int_space)
    pool2 = Dropout(rate)(pool2)
    pool3 = conv3dLeakyDownProj(pool2, filters,int_space)
    pool3 = Dropout(rate)(pool3)
    pool4 = conv3dLeakyDownProj(pool3, filters,int_space)
    pool4 = Dropout(rate)(pool4)
    pool5 = conv3dLeakyDownProj(pool4, filters,int_space)
    pool5 = Dropout(rate)(pool5)
    up4 = concatenate([UpSampling3D(size=(1,2,2))(pool5), pool4], axis=channel_axis)
    conv4 =conv3dLeakyProj(up4,filters,int_space)
    conv4 =Dropout(rate)(conv4)
    up3 = concatenate([UpSampling3D(size=(1,2,2))(conv4), pool3], axis=channel_axis)
    conv3 =conv3dLeakyProj(up3,filters,int_space)
    conv3 =Dropout(rate)(conv3)
    up2 = concatenate([UpSampling3D(size=(1,2,2))(conv3), pool2], axis=channel_axis)
    conv2 =conv3dLeakyProj(up2,filters,int_space)
    conv2 =Dropout(rate)(conv2)
    up1 = concatenate([UpSampling3D(size=(1,2,2))(conv2), pool1], axis=channel_axis)
    conv1 =conv3dLeakyProj(up1,filters,int_space)
    conv1 =Dropout(rate)(conv1)
    up0 = concatenate([UpSampling3D(size=(1,2,2))(conv1), inputs], axis=channel_axis)
    conv0 = conv3dLeaky(up0,output_channels)

    #up4 = concatenate([UpSampling2D(size=(2, 2))(pool5), pool4], axis=channel_axis)
    #conv4_2 =conv3dLeakyProj(up4,filters,int_space)
    #conv4_2 =Dropout(rate)(conv4_2)
    #up3_2 = concatenate([UpSampling3D(size=(1, 2,2))(conv4_2), pool3], axis=channel_axis)
    #conv3_2 =conv3dLeakyProj(up3_2,filters,int_space)
    #conv3_2 =Dropout(rate)(conv3)
    #up2_2 = concatenate([UpSampling3D(size=(1, 2,2))(conv3_2), pool2], axis=channel_axis)
    #conv2_2 =conv3dLeakyProj(up2,filters,int_space)
    #conv2_2 =Dropout(rate)(conv2_2)
    #up1_2 = concatenate([UpSampling3D(size=(1, 2,2))(conv2_2), pool1], axis=channel_axis)
    #conv1_2 =conv3dLeakyProj(up1,filters,int_space)
    #conv1_2 =Dropout(rate)(conv1_2)
    #up0_2 = concatenate([UpSampling3D(size=(1, 2,2))(conv1_2), inputs], axis=channel_axis)
    #conv0_2 = conv3dLeaky(up0_2,output_channels)


    #conv0  = GaussianNoise(rate)(conv0)
    return Model(inputs, outputs=conv0)


def jaccard_loss_mean_smooth3D(y_true, y_pred, smooth=10):
    """
    Mean Jaccard Loss per classes.
    Jaccard = (|X & Y|)/ (|X|+ |Y| - |X & Y|)
            = sum(|A*B|)/(sum(|A|)+sum(|B|)-sum(|A*B|))
    The jaccard distance loss is usefull for unbalanced datasets. This has been
    shifted so it converges on 0 and is smoothed to avoid exploding or disapearing
    gradient.
    Ref: https://en.wikipedia.org/wiki/Jaccard_index
    """
    y_true=K.clip(y_true,0,1)
    y_pred=K.clip(y_pred,0,1)
    intersection = y_true * y_pred
    union= y_true + y_pred - intersection
    #union =K.sum(union,axis=[3,2,1]) 08/04/2020
    union =K.sum(union,axis=[2,3,4])
    #intersection = K.sum(intersection,axis=[3,2,1]) 08/04/2020
    intersection = K.sum(intersection,axis=[2,3,4])
    jac = (intersection + smooth) / (union + (smooth))
    print('Jacc Shape',jac.shape)
    jac = K.mean(jac,axis=1)
    return 1-jac




