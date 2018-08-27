import numpy as np
import tensorflow as tf

from CityscapesHandler import CityscapesHandler
from PixelNet import PixelNet


# draw results at given pixel indexes into an image, to give a litte insight into the training process
def draw_results(image, idx, res):
    for k in range(0, len(idx)):
        image[idx[k, 1], idx[k, 2], 0] = 0
        image[idx[k, 1], idx[k, 2], 1] = 0
        image[idx[k, 1], idx[k, 2], 2] = 0
        if(res[k] < 9):
            image[idx[k, 1], idx[k, 2], 0] = res[k] / 9.0 * 255
        elif(res[k] < 18):
            image[idx[k, 1], idx[k, 2], 1] = res[k] / 9.0 * 255
        else:
            image[idx[k, 1], idx[k, 2], 2] = res[k] / 9.0 * 255
    
    return image
    

#TODO: - Some configuration seems to be different to the original implementation.
#      - Some adjustments at image acquisition and training process for mini-batch training (e.g. model save steps, interruptable batch acquisition)
#      - which image ratio for training?
#      - pixel sampling is much faster than before but still takes much memory, can be optimized
#      - (globally handle the hyperparams below (own file), such that they don't need to be set in both train_PixelNet.py and predict_PixelNet.py)
#      (- download/implementation of resnet instead of VGG, see link below)
#      ...
   
path_vgg16_vars = "./data/vgg_16.ckpt" #downloadable at https://github.com/tensorflow/models/tree/master/research/slim
model_save_path = "./model/pixelnet"
n_images = 5
n_steps = 150
n_classes = 30
pixel_sample_size = 2000
lr = 0.00003
input_image_shape = (224, 224) #(width, height)

csh = CityscapesHandler()
train_x, train_y = csh.getTrainSet(n_images, shape=input_image_shape)
#train_y = train_y[:, :, :, None]

input_image_shape = train_x[0].shape

train_x_tf = tf.constant(train_x, dtype=tf.float32)
train_y_tf = tf.constant(train_y, dtype=tf.int32)

with tf.Graph().as_default():
    images = tf.placeholder(tf.float32, shape=[None, input_image_shape[0], input_image_shape[1], 3], name='images')
    labels = tf.placeholder(tf.int32, shape=[None, input_image_shape[0], input_image_shape[1], 1], name='labels')
    index = tf.placeholder(tf.int32, shape=[None, 3], name='index')
    
    #preprocess
    r, g, b = tf.split(axis=3, num_or_size_splits=3, value=images)
    VGG_MEAN = [103.939, 116.779, 123.68]
    bgr = tf.concat(values=[b - VGG_MEAN[0], g - VGG_MEAN[1], r - VGG_MEAN[2]], axis=3)   
    
    pn = PixelNet()
    logits, y = pn.run(images=bgr, num_classes=n_classes, labels=labels, index=index)
    #y = tf.one_hot(y, n_classes)
    
    vgg_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_16')[:-2]   
    pixelnet_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    
    result = tf.argmax(logits, 1)
    
    
    with tf.variable_scope('Loss'):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=logits)
        loss = tf.reduce_mean(cross_entropy)
        accuracy = tf.Print(loss, data=[loss], message="loss:")

    with tf.variable_scope('Accuracy'):
        #?
        accuracy = tf.reduce_mean(tf.cast(tf.equal(result, y), tf.float32))
        accuracy = tf.Print(accuracy, data=[accuracy], message="accuracy:")
        

    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
    
    #load VGG model
    saver = tf.train.Saver(vgg_vars)
    sess = tf.Session()

    sess.run([
        tf.local_variables_initializer(),
        tf.global_variables_initializer(),
    ])
    
    saver.restore(sess, path_vgg16_vars)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
       
    result_image = np.full((input_image_shape[0], input_image_shape[1], 3), 255)
    print("start training")
    try:
        for step in range(n_steps):
        
            # TODO: this implementation only choses 2000 pixels from the first image in batch, adapt such that 2000 pixels per image
            #       are chosen (or 2000 pixels over the whole batch? don't think so)
            idx = np.random.choice(input_image_shape[0] * input_image_shape[1], size=pixel_sample_size, replace=False).reshape(pixel_sample_size, 1)
            idx = np.concatenate((idx / input_image_shape[1], idx % input_image_shape[1]), axis=1).astype(np.int)
            idx = np.insert(idx,0, 0, axis=1)
            
            batch_images, batch_labels = tf.train.batch([train_x_tf, train_y_tf], batch_size=batch_size, capacity=300,
                                                    enqueue_many=True, allow_smaller_final_batch=True)
            
            feed_dict = {images: train_x, labels: train_y, index: idx}
            _, loss_value, res = sess.run([train_op, loss_mean, result], feed_dict=feed_dict)
            
            print('step %d - loss: %.2f' % (step, loss_value))

            
            test_x, test_y = csh.getValSet(batch_size)
            feed_dict = {images: batch_images, labels: batch_labels, index: idx, pred: True}

            sess.run(accuracy, feed_dict=feed_dict)
            
            
            result_image = draw_results(result_image, idx, res)        
            if(step % 30 == 0):
                csh.displayImage(np.concatenate((train_x[0], result_image), axis=0).astype(np.uint8))
        
    except KeyboardInterrupt:
        #interrupts training process after pressing ctrl+c
        pass
        
    csh.displayImage(np.concatenate((train_x[0], result_image), axis=0).astype(np.uint8))
    print("training done")
    
    #save new model to disk
    print("start saving model")
    saver = tf.train.Saver(pixelnet_vars, save_relative_paths = True)
    save_path = saver.save(sess, model_save_path)
    print("Model saved in path: %s" % save_path)

