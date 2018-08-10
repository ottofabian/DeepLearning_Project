import numpy as np
import tensorflow as tf

from CityscapesHandler import CityscapesHandler


class PixelNet:
    def __init__(self, vgg16_npy_path=None):
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

    def random_sampling(self, features, labels, index):
        with tf.name_scope('RandomSampling'):
            shape = features[0].shape[1:-1]
            upsampled = [features[0]]
            for i in range(1, len(features)):
                upsampled.append(tf.image.resize_bilinear(features[i], shape))
            if index is None:
                vector = tf.concat(upsampled, axis=-1)
                label = None
            else:
                label = tf.gather_nd(labels, index)
                sampled = [tf.gather_nd(feature, index) for feature in upsampled]
                vector = tf.concat(sampled, axis=-1)
        return vector, label

    # pixel sampling from original implementation
    def _sample_pixels(self, gt, samplesize, pad):
        # (sample locations and get the labels)
        (y, x) = (gt < 255.0).nonzero()
        v = gt[y, x]
        lv = len(v)
        c = np.arange(lv)
        if samplesize <= lv:
            inds = np.random.choice(c, size=samplesize, replace=False)
        else:
            inds = np.random.choice(c, size=samplesize, replace=True)
        y = y[inds]
        x = x[inds]
        labs = v[inds]
        locs = np.array([y, x]).transpose() + pad
        return locs, labs

    def run(self, images, labels, index, num_classes):

        features = []

        # VGG part
        # maybe replace this one with 3,3 conv layers
        with tf.name_scope('conv_1'):
            self.conv1_1 = self.conv_layer(images, "conv1_1")
            self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
            features.append(self.conv1_2)
            self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        with tf.name_scope('conv_2'):
            self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
            self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
            features.append(self.conv2_2)
            self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        with tf.name_scope('conv_3'):
            self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
            self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
            self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
            features.append(self.conv3_3)
            self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        with tf.name_scope('conv_4'):
            self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
            self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
            self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
            features.append(self.conv4_3)
            self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        with tf.name_scope('conv_5'):
            self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
            self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
            self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
            features.append(self.conv5_3)
            self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        # conv replacement of VGG's last fc layers
        with tf.name_scope('conv_6'):
            self.conv6_1 = tf.layers.conv2d(self.pool5, 4096, 7, padding='SAME', activation=tf.nn.relu)

        with tf.name_scope('conv_7'):
            self.conv7_1 = tf.layers.conv2d(self.conv6_1, 4096, 1, padding='SAME', activation=tf.nn.relu)

        features.append(self.conv7_1)

        # sampling layer
        x, y = self.random_sampling(features, labels, index)

        with tf.name_scope('MLP'):
            x = tf.layers.dropout(x, 0.5, name='dropout1')
            x = tf.layers.dense(x, 4096, activation=tf.nn.relu, name='fc1')
            x = tf.layers.dropout(x, 0.5, name='dropout2')
            x = tf.layers.dense(x, 4096, activation=tf.nn.relu, name='fc2')
            x = tf.layers.dropout(x, 0.5, name='dropout3')
            x = tf.layers.dense(x, num_classes, activation=tf.nn.relu, name='fc3')

        self.data_dict = None

        if labels is not None:
            return x, y
        else:
            return x

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filter = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filter, [1, 1, 1, 1], padding='SAME')

            bias = self.get_bias(name)
            return tf.nn.relu(tf.nn.bias_add(conv, bias))

    # def fc_layer(self, bottom, name):
    #     with tf.variable_scope(name):
    #         shape = bottom.get_shape().as_list()
    #         dim = 1
    #         for d in shape[1:]:
    #             dim *= d
    #         x = tf.reshape(bottom, [-1, dim])
    #
    #         weights = self.get_fc_weight(name)
    #         biases = self.get_bias(name)
    #
    #         fc = tf.nn.bias_add(tf.matmul(x, weights), biases)
    #
    #         return fc

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")

        # def get_fc_weight(self, name):
        #     return tf.constant(self.data_dict[name][0], name="weights")


n_images = 5
n_steps = 10
n_classes = 30
pixel_sample_size = 10

csh = CityscapesHandler()
train_x, train_y = csh.getTrainSet(n_images)
train_y = train_y[:, :, :, None]

with tf.Graph().as_default():
    images = tf.placeholder(tf.float32, shape=[n_images, 160, 320, 3], name='images')
    labels = tf.placeholder(tf.int32, shape=[n_images, 160, 320, 1], name='labels')
    # TODO maybe change this to [...,3]
    # indexing is still problematic, due to memory issues.
    index = tf.placeholder(tf.int32, shape=[pixel_sample_size, 1], name='index')

    pn = PixelNet('./data/vgg16.npy')
    logits, y = pn.run(images=images, labels=labels, index=index, num_classes=n_classes)

    y = tf.one_hot(y, n_classes)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)
    idx = np.random.choice(160 * 320, size=pixel_sample_size, replace=False).reshape(pixel_sample_size, 1)
    # idx = np.tile(idx, (3, 1)).T

    for step in range(n_steps):
        feed_dict = {images: train_x, labels: train_y, index: idx}

        _, loss_value = sess.run([train_op, loss],
                                 feed_dict=feed_dict)

        print('Step %d: loss = %.2f' % (step, loss_value))
