import numpy as np
import tensorflow as tf
<<<<<<< HEAD
import tensorflow.contrib.slim as slim

from vgg16 import vgg_16

class PixelNet:  
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
        horizontal_offset= tf.reshape(horizontal_offset, [-1, 1])        
        horizontal_offset = tf.tile(horizontal_offset, [1, I.shape[3]])
        
        horizontal_interpolated_top = (
            ((1.0 - horizontal_offset) * values_at_top_left)
            + (horizontal_offset * values_at_top_right))
            
        horizontal_interpolated_bottom = (
            ((1.0 - horizontal_offset) * values_at_bottom_left)
            + (horizontal_offset * values_at_bottom_right))            
            
        vertical_offset = C[:, 2] - tf.cast(top_left[:, 2], tf.float32) 
        vertical_offset= tf.reshape(vertical_offset, [-1, 1])
        vertical_offset = tf.tile(vertical_offset, [1,I.shape[3]])
        
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
                    
=======

from CityscapesHandler import CityscapesHandler

np.set_printoptions(threshold=np.inf)


def interpolate_bipolar(I, C):
    top_left = tf.cast(tf.floor(C), tf.int32)
    top_right = tf.cast(tf.concat([tf.floor(C[:, 0:1]), tf.ceil(C[:, 1:2])], 1), tf.int32)
    bottom_left = tf.cast(tf.concat([tf.ceil(C[:, 0:1]), tf.floor(C[:, 1:2])], 1), tf.int32)
    bottom_right = tf.cast(tf.ceil(C), tf.int32)

    values_at_top_left = get_values_at_coordinates(I, top_left)
    values_at_top_right = get_values_at_coordinates(I, top_right)
    values_at_bottom_left = get_values_at_coordinates(I, bottom_left)
    values_at_bottom_right = get_values_at_coordinates(I, bottom_right)

    # Varies between 0.0 and 1.0.
    horizontal_offset = C[:, 0] - tf.cast(top_left[:, 0], tf.float32)
    horizontal_offset = tf.reshape(horizontal_offset, [-1, 1])
    horizontal_offset = tf.tile(horizontal_offset, [1, I.shape[3]])

    horizontal_interpolated_top = (
            ((1.0 - horizontal_offset) * values_at_top_left)
            + (horizontal_offset * values_at_top_right))

    horizontal_interpolated_bottom = (
            ((1.0 - horizontal_offset) * values_at_bottom_left)
            + (horizontal_offset * values_at_bottom_right))

    vertical_offset = C[:, 1] - tf.cast(top_left[:, 1], tf.float32)
    vertical_offset = tf.reshape(vertical_offset, [-1, 1])
    vertical_offset = tf.tile(vertical_offset, [1, I.shape[3]])

    interpolated_result = (
            ((1.0 - vertical_offset) * horizontal_interpolated_top)
            + (vertical_offset * horizontal_interpolated_bottom))

    return interpolated_result


def get_values_at_coordinates(input, coordinates):
    input_as_vector = tf.reshape(input, [input.shape[0], -1, input.shape[3]])
    coordinates_as_indices = (coordinates[:, 0] * tf.shape(input)[1]) + coordinates[:, 1]
    result = tf.gather(input_as_vector, coordinates_as_indices, axis=1)
    return result


class PixelNet:
    def __init__(self, vgg16_npy_path=None):
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

    def sample_sparse(self, features, index):
        samples = []

        for f in features:
            idx_scaled = index / features[0].shape[1]
            idx_scaled = tf.cast(idx_scaled, dtype=tf.float32) * (tf.cast(f.shape[1], dtype=tf.float32) - 1)
            samples.append(interpolate_bipolar(f, idx_scaled))

        return tf.concat(samples, axis=-1)

    def sample_dense(self, features, index):
        shape = features[0].shape[1:-1]
        upsampled = [features[0]]

        for i in range(1, len(features)):
            upsampled.append(tf.image.resize_bilinear(features[i], shape))

        sampled = [tf.transpose(tf.gather_nd(tf.transpose(feature, [1, 2, 0, 3]), index), [1, 0, 2]) for feature in
                   upsampled]

        return tf.concat(sampled, axis=-1)

    def random_sampling(self, features, labels, index, pred):
        with tf.name_scope('RandomSampling'):
            if index is None:
                # TODO: Wo kommt das upsampled her ?
>>>>>>> d88887cc3030bd04dbe5adb20aaec8f3605bd435
                vector = tf.concat(upsampled, axis=-1)
                vector = tf.reshape(vector, (shape[0] * shape[1], vector.shape[3]))
                label = None
            else:
<<<<<<< HEAD
                #decide for one:
                #------------------------------
                #previous implementation (no sparsity):
                # upsampled = [features[0]]
                # for i in range(1, len(features)):
                    # upsampled.append(tf.image.resize_bilinear(features[i], shape))
                #samples = [tf.gather_nd(feature, index) for feature in upsampled]
                #------------------------------
                #sparse sampling:
                samples = []
                for f in features:
                    idx_image = tf.cast(index[:, 0], dtype=tf.float32)
                    idx1 = tf.cast(index[:, 1] / features[0].shape[1], dtype=tf.float32) * (tf.cast(f.shape[1], dtype=tf.float32) - 1)
                    idx2 = tf.cast(index[:, 2] / features[0].shape[2], dtype=tf.float32) * (tf.cast(f.shape[2], dtype=tf.float32) - 1)
                    idx_scaled = tf.stack([idx_image, idx1, idx2], axis = 1)
                    
                    samples.append(self.interpolate_bilinear(f, idx_scaled))                    
                #------------------------------ 
                vector = tf.concat(samples, axis=-1)
                label = tf.gather_nd(labels, index)
            
            return vector, label


    def run(self, images, num_classes, labels=None, index=None):
        with tf.name_scope('PixelNet'):
            
            out, layers_out = vgg_16(images,
                                     num_classes=num_classes,
                                     is_training=True,
                                     dropout_keep_prob=0.5,
                                     spatial_squeeze=False,
                                     scope='vgg_16',
                                     fc_conv_padding='SAME',
                                     global_pool=False)
           
            features = []
            features.append(layers_out["vgg_16/conv1/conv1_2"])
            features.append(layers_out["vgg_16/conv2/conv2_2"])
            features.append(layers_out["vgg_16/conv3/conv3_3"])
            features.append(layers_out["vgg_16/conv4/conv4_3"])
            features.append(layers_out["vgg_16/conv5/conv5_3"])
            features.append(layers_out["vgg_16/dropout7"])

            x, y = self.random_sampling(features, labels, index)
            with tf.name_scope('MLP'):
                x = slim.fully_connected(x, 4096, activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(0.0, 0.001), scope='fc1')
                x = slim.dropout(x, 0.5, scope='dropout1')
                x = slim.fully_connected(x, 4096, activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(0.0, 0.001), scope='fc2')
                x = slim.dropout(x, 0.5, scope='dropout2')
                x = slim.fully_connected(x, num_classes, activation_fn=tf.nn.relu, weights_initializer=tf.truncated_normal_initializer(0.0, 0.001), scope='fc3')
            if labels is not None:
                return x, y
            else:
                return x
=======
                # index = tf.Print(index, [index], message="index", summarize = 200)
                label = tf.transpose(tf.gather_nd(tf.transpose(labels, [1, 2, 0]), index), [1, 0])
                # label = tf.Print(label, [label], message="label", summarize = 200)
                vector = tf.cond(pred, lambda: self.sample_dense(features, index),
                                 lambda: self.sample_sparse(features, index))

        return vector, label

    def run(self, images, labels, index, num_classes, pred):

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
        x, y = self.random_sampling(features, labels, index, pred)

        with tf.name_scope('MLP'):
            # x = tf.layers.dropout(x, 0.5, name='dropout1')
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

    def get_conv_filter(self, name):
        return tf.constant(self.data_dict[name][0], name="filter")

    def get_bias(self, name):
        return tf.constant(self.data_dict[name][1], name="biases")


csh = CityscapesHandler()

batch_size = 10
n_steps = 50
n_classes = csh.getNumLabels()
pixel_sample_size = 2000
max_img = -1

train_x, train_y = csh.getTrainSet(max_img)
print(train_y[0])

train_x_tf = tf.constant(train_x, dtype=tf.float32)
train_y_tf = tf.constant(train_y, dtype=tf.int32)

with tf.Graph().as_default():
    images = tf.placeholder(tf.float32, shape=[batch_size, 224, 224, 3], name='images')
    labels = tf.placeholder(tf.int32, shape=[batch_size, 224, 224], name='labels')

    # TODO maybe change this to [...,3]use
    # indexing is still problematic, due to memory issues.
    index = tf.placeholder(tf.int32, shape=[None, 2], name='index')
    pred = tf.placeholder(tf.bool)

    pn = PixelNet('./data/vgg16.npy')

    logits, y = pn.run(images=images, labels=labels, index=index, num_classes=n_classes, pred=pred)

    # logits = tf.Print(logits, [logits], message="logits", summarize=2000)

    result = tf.argmax(logits, 2)

    # y = tf.Print(y, [logits[0][0]], message="result", summarize = 20000)

    y = tf.one_hot(y, n_classes)

    # y = tf.Print(y, [y[0][0]], message="y      ", summarize = 20000)

    with tf.variable_scope('Loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        accuracy = tf.Print(loss, data=[loss], message="loss:")

    with tf.variable_scope('Accuracy'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(result, y), tf.float32))
        accuracy = tf.Print(accuracy, data=[accuracy], message="accuracy:")

    optimizer = tf.train.AdamOptimizer(0.0001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    sess = tf.Session()

    sess.run([
        tf.local_variables_initializer(),
        tf.global_variables_initializer(),
    ])

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # test = np.full((224, 224, 3), 255)
    for step in range(n_steps):
        # samples pixels new for each epoch

        idx = np.random.choice(224 * 224, size=pixel_sample_size, replace=False).reshape(pixel_sample_size, 1)
        idx = np.concatenate((idx / 224, idx % 224), axis=1).astype(np.int)

        # returns array of x,y coordinates for sampled pixels
        # idx = csh.samplePixels(numSamples=pixel_sample_size,imageShape=(224,224))

        batch_images, batch_labels = tf.train.batch([train_x_tf, train_y_tf], batch_size=batch_size, capacity=300,
                                                    enqueue_many=True, allow_smaller_final_batch=True)

        feed_dict = {images: batch_images, labels: batch_labels, index: idx, pred: False}

        _, loss_value, res = sess.run([train_op, loss, result], feed_dict=feed_dict)

        print('step %d - loss: %.2f' % (step, loss_value))

        test_x, test_y = csh.getValSet(batch_size)
        feed_dict = {images: batch_images, labels: batch_labels, index: idx, pred: True}

        sess.run(accuracy, feed_dict=feed_dict)

    # for k in range(0, len(idx)):
    #     test[idx[k, 0], idx[k, 1], 0] = res[0][k] / 27.0 * 255
    #     test[idx[k, 0], idx[k, 1], 1] = 0
    #     test[idx[k, 0], idx[k, 1], 2] = 0

    # csh.displayImage(np.concatenate((train_x[0], test), axis=0).astype(np.uint8))

    quit()
    # ------------------------------------------------------------------------

    # simple visualization of training result if n_images = 1
    pixel_sample_size = 224 * 224
    idx = np.random.choice(224 * 224, size=pixel_sample_size, replace=False).reshape(pixel_sample_size, 1)
    idx = np.concatenate((idx / 224, idx % 224), axis=1).astype(np.int)

    feed_dict = {images: train_x, labels: train_y, index: idx, pred: True}
    res = sess.run([result], feed_dict=feed_dict)
    res = res[0]

    test = np.full((224, 224, 3), 255)
    for k in range(0, len(idx)):
        test[idx[k, 0], idx[k, 1], 0] = res[0][k] / 27.0 * 255
        test[idx[k, 0], idx[k, 1], 1] = 0
        test[idx[k, 0], idx[k, 1], 2] = 0

    csh.displayImage(np.concatenate((train_x[0], test), axis=0).astype(np.uint8))
>>>>>>> d88887cc3030bd04dbe5adb20aaec8f3605bd435
