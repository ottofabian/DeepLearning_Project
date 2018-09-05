import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

from vgg16 import vgg_16


class PixelNet:
    def generate_sample_idxs(self, input_image_shape, n_images, pixel_sample_size):
        idx = np.empty((0, 3))
        for k in range(n_images):
            pxl_idx = np.random.choice(input_image_shape[0] * input_image_shape[1], size=pixel_sample_size,
                                       replace=False).reshape(pixel_sample_size, 1)
            pxl_idx = np.concatenate((pxl_idx / input_image_shape[1], pxl_idx % input_image_shape[1]), axis=1).astype(
                np.int)
            img_idx = np.full(pixel_sample_size, k)
            comb = np.insert(pxl_idx, 0, img_idx, axis=1)
            idx = np.append(idx, comb, axis=0)
        return idx.astype(np.int)

    def preprocess_images(self, images):
        r, g, b = tf.split(axis=3, num_or_size_splits=3, value=images)
        VGG_MEAN = [103.939, 116.779, 123.68]
        return tf.concat(values=[b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)

    def interpolate_bilinear(self, I, C):
        top_left = tf.cast(tf.floor(C), tf.int32)
        top_right = tf.cast(tf.concat([C[:, 0:1], tf.floor(C[:, 1:2]), tf.ceil(C[:, 2:3])], 1), tf.int32)
        bottom_left = tf.cast(tf.concat([C[:, 0:1], tf.ceil(C[:, 1:2]), tf.floor(C[:, 2:3])], 1), tf.int32)
        bottom_right = tf.cast(tf.ceil(C), tf.int32)

        values_at_top_left = tf.gather_nd(I, top_left)
        values_at_top_right = tf.gather_nd(I, top_right)
        values_at_bottom_left = tf.gather_nd(I, bottom_left)
        values_at_bottom_right = tf.gather_nd(I, bottom_right)

        # Varies between 0.0 and 1.0.
        horizontal_offset = C[:, 1] - tf.cast(top_left[:, 1], tf.float32)
        horizontal_offset = tf.reshape(horizontal_offset, [-1, 1])
        horizontal_offset = tf.tile(horizontal_offset, [1, I.shape[3]])

        horizontal_interpolated_top = (
            ((1.0 - horizontal_offset) * values_at_top_left)
            + (horizontal_offset * values_at_top_right))

        horizontal_interpolated_bottom = (
            ((1.0 - horizontal_offset) * values_at_bottom_left)
            + (horizontal_offset * values_at_bottom_right))

        vertical_offset = C[:, 2] - tf.cast(top_left[:, 2], tf.float32)
        vertical_offset = tf.reshape(vertical_offset, [-1, 1])
        vertical_offset = tf.tile(vertical_offset, [1, I.shape[3]])

        interpolated_result = (
            ((1.0 - vertical_offset) * horizontal_interpolated_top)
            + (vertical_offset * horizontal_interpolated_bottom))

        return interpolated_result

    def random_sampling(self, features, labels, index):
        with tf.name_scope('RandomSampling'):
            shape = features[0].shape[1:-1]
            if index is None:
                upsampled = [features[0]]
                for i in range(1, len(features)):
                    upsampled.append(tf.image.resize_bilinear(features[i], shape))
                vector = tf.concat(upsampled, axis=-1)
                vector = tf.reshape(vector, (shape[0] * shape[1], vector.shape[3]))
                label = None
            else:
                # decide for one:
                # ------------------------------
                # previous implementation (no sparsity):
                # upsampled = [features[0]]
                # for i in range(1, len(features)):
                # upsampled.append(tf.image.resize_bilinear(features[i], shape))
                # samples = [tf.gather_nd(feature, index) for feature in upsampled]
                # ------------------------------
                # sparse sampling:
                samples = []
                for f in features:
                    idx_image = tf.cast(index[:, 0], dtype=tf.float32)
                    idx1 = tf.cast(index[:, 1] / features[0].shape[1], dtype=tf.float32) * (
                    tf.cast(f.shape[1], dtype=tf.float32) - 1)
                    idx2 = tf.cast(index[:, 2] / features[0].shape[2], dtype=tf.float32) * (
                    tf.cast(f.shape[2], dtype=tf.float32) - 1)
                    idx_scaled = tf.stack([idx_image, idx1, idx2], axis=1)

                    samples.append(self.interpolate_bilinear(f, idx_scaled))
                    # ------------------------------
                vector = tf.concat(samples, axis=-1)
                label = tf.gather_nd(labels, index)

            return vector, label

    def build(self, images, num_classes, labels=None, index=None):
        with tf.name_scope('PixelNet'):

            out, layers_out = vgg_16(images,
                                     num_classes=num_classes,
                                     is_training=True,
                                     dropout_keep_prob=0.5,
                                     spatial_squeeze=False,
                                     scope='vgg_16',
                                     fc_conv_padding='VALID',
                                     global_pool=False)

            features = []
            features.append(layers_out["vgg_16/conv1/conv1_2"])
            features.append(layers_out["vgg_16/conv2/conv2_2"])
            features.append(layers_out["vgg_16/conv3/conv3_3"])
            features.append(layers_out["vgg_16/conv4/conv4_3"])
            features.append(layers_out["vgg_16/conv5/conv5_3"])
            features.append(layers_out["vgg_16/fc7"])

            x, y = self.random_sampling(features, labels, index)
            with tf.name_scope('MLP'):
                x = slim.fully_connected(x, 4096, activation_fn=tf.nn.relu,
                                         weights_initializer=tf.truncated_normal_initializer(0.0, 0.001), scope='fc1')
                x = slim.dropout(x, 0.5, scope='dropout1')
                x = slim.fully_connected(x, 4096, activation_fn=tf.nn.relu,
                                         weights_initializer=tf.truncated_normal_initializer(0.0, 0.001), scope='fc2')
                x = slim.dropout(x, 0.5, scope='dropout2')
                x = slim.fully_connected(x, num_classes, activation_fn=tf.nn.relu,
                                         weights_initializer=tf.truncated_normal_initializer(0.0, 0.001), scope='fc3')
            if labels is not None:
                return x, y
            else:
                return x
