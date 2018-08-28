import os

import numpy as np
import tensorflow as tf

from CityscapesHandler import CityscapesHandler
from PixelNet import PixelNet


def draw_results(image, idx, res):
    h = image.shape[0]
    w = image.shape[1]
    for k in range(0, h):
        for i in range(0, w):
            image[k, i, 0] = 0
            image[k, i, 1] = 0
            image[k, i, 2] = 0
            if(res[k * w + i] < 10):
                image[k, i, 0] = res[k * w + i] / 10.0 * 255
            elif(res[k * w + i] < 20):
                image[k, i, 1] = res[k * w + i] / 10.0 * 255
            else:
                image[k, i, 2] = res[k * w + i] / 10.0 * 255
    
    return image
    
img_idx = 3
n_classes = 30
input_image_shape = (224, 224) #(width, height)
model_path = os.path.dirname('./model/checkpoint')


csh = CityscapesHandler()
train_x, train_y = csh.getTrainSet(img_idx + 1, shape=input_image_shape)
train_x, train_y = train_x[img_idx, None], train_y[img_idx, None]
train_y = train_y[:, :, :, None]
input_image_shape = train_x[0].shape


with tf.Graph().as_default():
    images = tf.placeholder(tf.float32, shape=[1, input_image_shape[0], input_image_shape[1], 3], name='images')

    
    #preprocess
    r, g, b = tf.split(axis=3, num_or_size_splits=3, value=images)
    VGG_MEAN = [103.939, 116.779, 123.68]
    bgr = tf.concat(values=[b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)   
    
    pn = PixelNet()
    logits = pn.build(images=bgr, num_classes=n_classes)
    predictions = tf.argmax(logits, 1)

    #load model
    pixelnet_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)  
    sess = tf.Session()
    saver = tf.train.Saver(pixelnet_vars)  
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)  

    feed_dict = {images: train_x}

    print("start prediction")
    res = sess.run([predictions], feed_dict=feed_dict)
    res = res[0]

    print("prediction done - draw results")

    result_image = np.full((input_image_shape[0], input_image_shape[1], 3), 255)
    result_image = draw_results(result_image, None, res)
            
    csh.displayImage(np.concatenate((train_x[0], result_image), axis=0).astype(np.uint8))
