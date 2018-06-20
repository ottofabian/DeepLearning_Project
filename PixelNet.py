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
                sampled = [tf.gather_nd(feature, index) for feature in upsampled]
                vector = tf.concat(sampled, axis=-1)
                label = tf.gather_nd(labels, index)
        return vector, label

    def run(self, images, labels, index, num_classes):

        features = []

        # VGG part
        # maybe replace this one with 3,3 conv layers
        self.conv1_1 = self.conv_layer(images, "conv1_1")
        self.conv1_2 = self.conv_layer(self.conv1_1, "conv1_2")
        self.pool1 = self.max_pool(self.conv1_2, 'pool1')

        self.conv2_1 = self.conv_layer(self.pool1, "conv2_1")
        self.conv2_2 = self.conv_layer(self.conv2_1, "conv2_2")
        self.pool2 = self.max_pool(self.conv2_2, 'pool2')

        self.conv3_1 = self.conv_layer(self.pool2, "conv3_1")
        self.conv3_2 = self.conv_layer(self.conv3_1, "conv3_2")
        self.conv3_3 = self.conv_layer(self.conv3_2, "conv3_3")
        self.pool3 = self.max_pool(self.conv3_3, 'pool3')

        self.conv4_1 = self.conv_layer(self.pool3, "conv4_1")
        self.conv4_2 = self.conv_layer(self.conv4_1, "conv4_2")
        self.conv4_3 = self.conv_layer(self.conv4_2, "conv4_3")
        self.pool4 = self.max_pool(self.conv4_3, 'pool4')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        self.conv5_1 = self.conv_layer(self.pool4, "conv5_1")
        self.conv5_2 = self.conv_layer(self.conv5_1, "conv5_2")
        self.conv5_3 = self.conv_layer(self.conv5_2, "conv5_3")
        self.pool5 = self.max_pool(self.conv5_3, 'pool5')

        features.append(self.pool5)

        # self.conv6_1 = self.conv_layer(self.pool4, "conv6_1")
        # self.conv6_2 = self.conv_layer(self.conv5_1, "conv6_2")
        # self.conv6_3 = self.conv_layer(self.conv5_2, "conv6_3")
        # self.pool6 = self.max_pool(self.conv5_3, 'pool6')
        #
        # self.conv7_1 = self.conv_layer(self.pool4, "conv7_1")
        # self.conv7_2 = self.conv_layer(self.conv5_1, "conv7_2")
        # self.conv7_3 = self.conv_layer(self.conv5_2, "conv7_3")
        # self.pool7 = self.max_pool(self.conv5_3, 'pool7')

        x, y = random_sampling(features, labels, index)

        with tf.name_scope('MLP'):
            x = tf.layers.dense(x, 4096, activation_fn=tf.nn.relu, scope='fc1')
            x = tf.layers.dropout(x, 0.5, scope='dropout1')
            x = tf.layers.dense(x, 4096, activation_fn=tf.nn.relu, scope='fc2')
            x = tf.layers.dropout(x, 0.5, scope='dropout2')
            x = tf.layers.dense(x, num_classes, activation_fn=tf.nn.relu, scope='fc3')

        self.data_dict = None

        if labels is not None:
            return x, y
        else:
            return x

    def max_pool(self, bottom, name):
        return tf.nn.max_pool(bottom, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME', name=name)

    def conv_layer(self, bottom, name):
        with tf.variable_scope(name):
            filt = self.get_conv_filter(name)

            conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')

            conv_biases = self.get_bias(name)
            bias = tf.nn.bias_add(conv, conv_biases)

            relu = tf.nn.relu(bias)
            return relu

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

# Todo
n_classes = 100000000
csh = CityscapesHandler()
train_x, train_y = csh.getTrainSet(n_images)

# Todo is this already done???
# train_x = train_x.reshape((n_images, 224, 224, 3))

with tf.Graph().as_default():
    images = tf.placeholder(tf.float32, shape=[n_images, 224, 224, 3], name='images')
    labels = tf.placeholder(tf.int32, shape=[n_images, 224, 224, 1], name='labels')
    index = tf.placeholder(tf.int32, shape=[4000, 3], name='index')

    pn = PixelNet('./data/vgg16.npy')
    logits, y = pn.run(images=images, labels=labels, index=index, num_classes=n_classes)

    y = tf.one_hot(y, num_classes)
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
    loss = tf.reduce_mean(cross_entropy)

    optimizer = tf.train.AdamOptimizer()
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())

    init = tf.global_variables_initializer()
    sess = tf.Session()

    sess.run(init)

    for step in range(n_steps):
        # Todo set the index for sampling
        feed_dict = {images: train_x, labels: train_y, index: ???}

        _, loss_value = sess.run([train_op, loss],
                                 feed_dict=feed_dict)

        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
