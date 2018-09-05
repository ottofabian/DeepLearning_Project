import os

import numpy as np
import tensorflow as tf

from CityscapesHandler import CityscapesHandler
from PixelNet import PixelNet


def label_to_color(num_classes, value):
    num_classes = float(num_classes)
    ratio = 2 * value / num_classes
    b = int(max(0, 255 * (1 - ratio)))
    r = int(max(0, 255 * (ratio - 1)))
    g = 255 - b - r
    return r, g, b


def draw_results(image, res, num_classes):
    h = image.shape[0]
    w = image.shape[1]
    for k in range(0, h):
        for i in range(0, w):
            r, g, b = label_to_color(num_classes, res[k * w + i])
            image[k, i, 0] = r
            image[k, i, 1] = g
            image[k, i, 2] = b

    return image


img_idx = 6
n_classes = 30
input_image_shape = (224, 224)  # (width, height)
model_path = os.path.dirname('./model/checkpoint')

csh = CityscapesHandler()
train_x, train_y = csh.getValSet(img_idx + 1, shape=input_image_shape)
train_x, train_y = train_x[img_idx, None], train_y[img_idx, None]
train_y = train_y[:, :, :, None]
input_image_shape = train_x[0].shape

with tf.Graph().as_default():
    images = tf.placeholder(tf.float32, shape=[1, input_image_shape[0], input_image_shape[1], 3], name='images')

    pn = PixelNet()
    train_bgr_norm = pn.preprocess_images(images)

    logits = pn.build(images=train_bgr_norm, num_classes=n_classes)
    predictions = tf.argmax(logits, 1)

    # load model
    pixelnet_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    sess = tf.Session()
    saver = tf.train.Saver(pixelnet_vars)
    ckpt = tf.train.get_checkpoint_state(model_path)
    saver.restore(sess, ckpt.model_checkpoint_path)

    print("start prediction")
    res = sess.run([predictions], feed_dict={images: train_x})[0]
    # TODO make this work
    # TODO do this for all images
    # csh.savePrediction(res)

    print("prediction done - draw results")

    result_image = np.full((input_image_shape[0], input_image_shape[1], 3), 255)
    result_image = draw_results(result_image, res, n_classes)

    csh.displayImage(np.concatenate((train_x[0], result_image), axis=0).astype(np.uint8))
