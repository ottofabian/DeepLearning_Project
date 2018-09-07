import os

import numpy as np
import tensorflow as tf

from CityscapesHandler import CityscapesHandler
from PixelNet import PixelNet

csh = CityscapesHandler()
n_images = 5
n_classes = csh.getNumTrainIDLabels() + 2
input_image_shape = (224, 224)  # (width, height)
model_path = os.path.dirname('./model/checkpoint')

val_x, val_y, filenames, filenames_labels = csh.getValSet(n_images, shape=input_image_shape, withFilenames=True)

prediction_filenames = csh.fromInputFilenamesToPredictionFilenames(filenames)

val_y = val_y[:, :, :, None]
val_y[val_y == 255] = n_classes - 1
val_y[val_y == -1] = n_classes - 1

input_image_shape = val_x[0].shape


def draw_results(res):
    image = np.full((input_image_shape[0], input_image_shape[1], 3), 255)
    for k in range(0, image.shape[0]):
        for i in range(0, image.shape[1]):
            r, g, b = csh.getColorFromLabelId(res[k, i])
            image[k, i, 0] = r
            image[k, i, 1] = g
            image[k, i, 2] = b

    return image


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

    for k in range(n_images):
        print("start prediction " + str(k + 1) + "/" + str(n_images))

        input_x = val_x[k, None]

        res = sess.run([predictions], feed_dict={images: input_x})[0]
        res = np.reshape(res, (input_image_shape[0], input_image_shape[1]))
        res[res == (n_classes - 1)] = 255

        print("prediction done - start saving output")
        csh.savePrediction(res, prediction_filenames[k], image_shape=(2048, 1024))
        print("saving output done")

        # result_image = draw_results(res)
        # csh.displayImage(np.concatenate((val_x[k], result_image), axis=0).astype(np.uint8))
