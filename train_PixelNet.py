import numpy as np
import tensorflow as tf

from CityscapesHandler import CityscapesHandler
from PixelNet import PixelNet

   
path_vgg16_vars = "./data/vgg_16.ckpt" #downloadable at https://github.com/tensorflow/models/tree/master/research/slim
model_save_path = "./model/pixelnet"

n_train_images = 2975  # max: 2975
n_val_images = 500    # max: 500
size_batch = 5
n_batches = int(n_train_images / size_batch)
n_steps = 150
n_classes = 30
pixel_sample_size = 2000 # per image
lr = 0.00003
input_image_shape = (320, 160) #(width, height)
valid_after_n_steps = 3


csh = CityscapesHandler()
train_x, train_y = csh.getTrainSet(n_train_images, shape=input_image_shape)
train_y = train_y[:, :, :, None]

val_x, val_y = csh.getValSet(n_val_images, shape=input_image_shape)
val_y = val_y[:, :, :, None]

input_image_shape = train_x[0].shape


with tf.Graph().as_default():
    train_images = tf.placeholder(tf.float32, shape=[None, input_image_shape[0], input_image_shape[1], 3], name='train_images')
    train_labels = tf.placeholder(tf.int32, shape=[None, input_image_shape[0], input_image_shape[1], 1], name='train_labels')
    
    val_images = tf.placeholder(tf.float32, shape=[None, input_image_shape[0], input_image_shape[1], 3], name='val_images')
    val_labels = tf.placeholder(tf.int32, shape=[None, input_image_shape[0], input_image_shape[1], 1], name='val_labels')
    
    index = tf.placeholder(tf.int32, shape=[None, 3], name='index') 
    batch_size = tf.placeholder(tf.int64)
    
    train_dataset = tf.data.Dataset.from_tensor_slices((train_images,train_labels)).shuffle(buffer_size=n_batches).batch(batch_size).repeat()
    val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels)).batch(batch_size).repeat()
    
    
    iter = tf.data.Iterator.from_structure(train_dataset.output_types, train_dataset.output_shapes)
    
    batch_images, batch_labels = iter.get_next()
    train_init_op = iter.make_initializer(train_dataset)
    val_init_op = iter.make_initializer(val_dataset)
    
    pn = PixelNet()
  
    #preprocess
    train_bgr_norm = pn.preprocess_images(batch_images)
  
    logits, y = pn.build(images=train_bgr_norm, num_classes=n_classes, labels=batch_labels, index=index)
    y_one_hot = tf.one_hot(y, n_classes)
    y = tf.reshape(y, [-1])
    
    vgg_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='vgg_16')[:-2]   
    pixelnet_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    
    predictions = tf.argmax(logits, 1)

    
    cross_entropy = tf.nn.softmax_cross_entropy_with_logits_v2(labels=y_one_hot, logits=logits)
    loss_mean = tf.reduce_mean(cross_entropy)

    # iou_mean, conf_mat = tf.metrics.mean_iou (labels=y, predictions=predictions, num_classes=n_classes)

    optimizer = tf.train.AdamOptimizer(lr)
    train_op = optimizer.minimize(loss=cross_entropy, global_step=tf.train.get_global_step())
    
    #load VGG model
    saver = tf.train.Saver(vgg_vars)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer()) 
    sess.run(tf.local_variables_initializer())     
    saver.restore(sess, path_vgg16_vars)

    print("start training")
    try:
    
        sess.run(train_init_op, feed_dict={train_images: train_x, train_labels: train_y, batch_size: size_batch})        

        for step in range(n_steps):
            tot_loss = 0
            for _ in range(n_batches):           
                idx = pn.generate_sample_idxs(input_image_shape, size_batch, pixel_sample_size)            
                __, loss_value = sess.run([train_op, loss_mean], feed_dict={index: idx})
                tot_loss += loss_value
                print("--batch ", _, "/", n_batches)
            print("step: {}, Loss: {:.4f}".format(step, tot_loss / n_batches))

            if(step % valid_after_n_steps == 0):
                sess.run(val_init_op, feed_dict={val_images: val_x, val_labels: val_y, batch_size: n_val_images})
                
                idx = pn.generate_sample_idxs(input_image_shape, n_val_images, pixel_sample_size)
                #sess.run([conf_mat], feed_dict={index: idx})
                # loss_value, iou  = sess.run([loss_mean, iou_mean], feed_dict={index: idx})
                loss_value = sess.run([loss_mean], feed_dict={index: idx})
                loss_value = loss_value[0]
                print("Validation Loss: {:.4f}".format(loss_value))
                # print('Validation (IOU mean): {:4f}'.format(iou))
                
                sess.run(train_init_op, feed_dict={train_images: train_x, train_labels: train_y, batch_size: size_batch})
                
    except KeyboardInterrupt:
        #interrupts training process after pressing ctrl+c
        pass
        
    print("training done")
    
    #save new model to disk
    print("start saving model")
    saver = tf.train.Saver(pixelnet_vars, save_relative_paths = True)
    save_path = saver.save(sess, model_save_path)
    print("Model saved in path: %s" % save_path)

