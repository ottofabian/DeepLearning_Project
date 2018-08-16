import numpy as np
import tensorflow as tf
from PIL import Image

np.set_printoptions(threshold=np.inf)

from CityscapesHandler import CityscapesHandler

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
    
    horizontal_interpolated_top = (
        ((1.0 - horizontal_offset) * values_at_top_left)
        + (horizontal_offset * values_at_top_right))
        
    horizontal_interpolated_bottom = (
        ((1.0 - horizontal_offset) * values_at_bottom_left)
        + (horizontal_offset * values_at_bottom_right))
        
    vertical_offset = C[:, 1] - tf.cast(top_left[:, 1], tf.float32)

    interpolated_result = (
        ((1.0 - vertical_offset) * horizontal_interpolated_top)
        + (vertical_offset * horizontal_interpolated_bottom))
      
    return interpolated_result
    
# original implementation for a single image 
# def get_values_at_coordinates(input, coordinates):
    # input_as_vector = tf.reshape(input, [-1])
    # coordinates_as_indices = (coordinates[:, 0] * tf.shape(input)[1]) + coordinates[:, 1]
    # return tf.gather(input_as_vector, coordinates_as_indices)

#implementation for multiple feature maps
def get_values_at_coordinates(input, coordinates):
    input_as_vector = tf.reshape(input, [input.shape[1] * input.shape[2], input.shape[3]])
    coordinates_as_indices = (coordinates[:, 0] * tf.shape(input)[1]) + coordinates[:, 1]
    return tf.gather(input_as_vector, coordinates_as_indices)
    
    
class PixelNet:
    def __init__(self, vgg16_npy_path=None):
        self.data_dict = np.load(vgg16_npy_path, encoding='latin1').item()

    def random_sampling(self, features, labels, index):
        with tf.name_scope('RandomSampling'):
            shape = features[0].shape[1:-1]
            # upsampled = [features[0]]
            
            # for i in range(1, len(features)):
                # upsampled.append(tf.image.resize_bilinear(features[i], shape))
            
            for e in features:
                print(e)
                
            if index is None:
                vector = tf.concat(upsampled, axis=-1)
                label = None
            else:
                samples = []
                
                index = tf.Print(index, [index], message="original index :")
                
                it = 0
                for f in features:
                    it += 1
                    
                    idx_scaled = index / features[0].shape[1]
                    idx_scaled = tf.cast(idx_scaled, dtype=tf.float32) * tf.cast(f.shape[1], dtype=tf.float32)                    
                    
                    idx_scaled = tf.Print(idx_scaled, [idx_scaled], message="idx_scaled " + str(it) + ": ")
                    
                    samples.append(interpolate_bipolar(f, idx_scaled))
                      
                vector = tf.concat(samples, axis=-1)            
            
                label = tf.transpose(tf.gather_nd(tf.transpose(labels, [1, 2, 0, 3]), index), [1, 0, 2])
                
                # sampled = [tf.transpose(tf.gather_nd(tf.transpose(feature, [1, 2, 0, 3]), index), [1, 0, 2]) for feature
                           # in upsampled]
                # vector = tf.concat(sampled, axis=-1)
                
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
        
        #x = tf.Print(x, [x])
        

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


n_images = 1
n_steps = 7
n_classes = 5
pixel_sample_size = 1

csh = CityscapesHandler()
train_x, train_y = csh.getTrainSet(n_images)
train_y = train_y[:, :, :, None]


with tf.Graph().as_default():
    images = tf.placeholder(tf.float32, shape=[n_images, 224, 224, 3], name='images')
    labels = tf.placeholder(tf.int32, shape=[n_images, 224, 224, 1], name='labels')
    
    # TODO maybe change this to [...,3]
    # indexing is still problematic, due to memory issues.
    index = tf.placeholder(tf.int32, shape=[None, 2], name='index')
    
    pn = PixelNet('./data/vgg16.npy')

    logits, y = pn.run(images=images, labels=labels, index=index, num_classes=n_classes)

    result = tf.argmax(logits, 1)

    y = tf.one_hot(y, n_classes)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer(0.0001)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)

    for step in range(n_steps):
        # samples pixels new for each epoch
        idx = np.random.choice(224 * 224, size=pixel_sample_size, replace=False).reshape(pixel_sample_size, 1)
        idx = np.concatenate((idx / 224, idx % 224), axis=1).astype(np.int)
        
        feed_dict = {images: train_x, labels: train_y, index: idx}

        _, loss_value = sess.run([train_op, loss],
                                      feed_dict=feed_dict)
        
        print('step %d - loss: %.2f' % (step, loss_value))

    # ------------------------------------------------------------------------

    # simple visualization of training result if n_images = 1
    pixel_sample_size = 224 * 224
    idx = np.random.choice(224 * 224, size=pixel_sample_size, replace=False).reshape(pixel_sample_size, 1)
    idx = np.concatenate((idx / 224, idx % 224), axis=1).astype(np.int)

    feed_dict = {images: train_x, labels: train_y, index: idx}
    _, loss_value, res = sess.run([train_op, loss, result],
                                  feed_dict=feed_dict)

    test = np.full((224, 224, 3), 255)
    out = np.full((224, 224, 1), 255)
    for k in range(0, len(idx)):
        test[idx[k, 0], idx[k, 1], 0] = res[0][k] / 30.0 * 255
        test[idx[k, 0], idx[k, 1], 1] = 0
        test[idx[k, 0], idx[k, 1], 2] = 0
        
        out[idx[k, 0], idx[k, 1]] = res[0][k]

    csh.displayImage(np.concatenate((train_x[0], test), axis=0).astype(np.uint8))
