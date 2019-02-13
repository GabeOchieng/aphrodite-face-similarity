import numpy as np
import tensorflow as tf
from base.BaseModel import BaseModel
from layers.Basics import batch_normalization
from layers.Convolutions import conv_layer
from layers.Convolutions import fully_connected
from layers.Convolutions import maxpool
from layers.Losses import contrastive_loss, DBL_loss
from layers.Losses import euclidean_distance

# TODO
class ConvNet(BaseModel):
    def __init__(self, data_loader, config):
        super(ConvNet, self).__init__(config)
        self.data_loader = data_loader
        self.n_classes = config.model.num_classes
        self.logits = None
        self.logits_argmax = None
        self.loss = None
        self.acc = None
        self.optimizer = None
        self.train_step = None
        self.build_model(config)
        self.init_saver()

    def cnn_networks(self, x, reuse=False):

        with tf.name_scope('network') as scope:

            paddings = tf.constant([
                [0, 0], [1, 1], 
                [1, 1], [0, 0]])
            
            pads1 = tf.pad(
                x, paddings=paddings, 
                mode='REFLECT', name='pads1')
            conv1 = conv_layer(
                pads1, filters=32, k_size=7,
                stride=1, padding='SAME',
                scope_name='conv1', reuse=reuse)
            pool1 = maxpool(
                conv1, k_size=2, stride=2,
                padding='SAME', scope_name='pool1')
            norm1 = batch_normalization(
                pool1, num_filters=32, eps=1e-05, 
                trainable=self.trainable, 
                scope_name='norm1')

            pads2 = tf.pad(
                norm1, paddings=paddings, 
                mode='REFLECT', name='pads2')
            conv2 = conv_layer(
                pads2, filters=64, k_size=5,
                stride=1, padding='SAME',
                scope_name='conv2', reuse=reuse)
            pool2 = maxpool(
                conv2, k_size=2, stride=2,
                padding='SAME', scope_name='pool2')
            norm2 = batch_normalization(
                pool2, num_filters=64, eps=1e-05, 
                trainable=self.trainable, 
                scope_name='norm2')

            pads3 = tf.pad(
                norm2, paddings=paddings, 
                mode='REFLECT', name='pads3')
            conv3 = conv_layer(
                pads3, filters=128, k_size=3,
                stride=1, padding='SAME',
                scope_name='conv3', reuse=reuse)
            pool3 = maxpool(
                conv3, k_size=2, stride=2,
                padding='SAME', scope_name='pool3')
            norm3 = batch_normalization(
                pool3, num_filters=128, eps=1e-05, 
                trainable=self.trainable, 
                scope_name='norm2')

            pads4 = tf.pad(
                norm3, paddings=paddings, 
                mode='REFLECT', name='pads4')
            conv4 = conv_layer(
                pads4, filters=256, k_size=1,
                stride=1, padding='SAME',
                scope_name='conv4', reuse=reuse)
            pool4 = maxpool(
                conv4, k_size=2, stride=2,
                padding='SAME', scope_name='pool4')
            norm4 = batch_normalization(
                pool4, num_filters=256, eps=1e-05, 
                trainable=self.trainable, 
                scope_name='norm2')

            pads5 = tf.pad(
                norm4, paddings=paddings, 
                mode='REFLECT', name='pads5')
            conv5 = conv_layer(
                pads5, filters=2, k_size=1,
                stride=1, padding='SAME',
                scope_name='conv5', reuse=reuse)
            pool5 = maxpool(
                conv5, k_size=2, stride=2,
                padding='SAME', scope_name='pool5')

            cur_dim = pool5.get_shape()
            pool5_dim = cur_dim[1] * cur_dim[2] * cur_dim[3]
            pool5_flatten = tf.reshape(pool5, shape=[-1, pool5_dim])

            fc6 = fully_connected(
                pool5_flatten, out_dim=500,
                scope_name='fc6', activation=tf.nn.relu)

            fc7 = fully_connected(
                fc6, out_dim=500,
                scope_name='fc7', activation=tf.nn.relu)

            fc8 = fully_connected(
                fc7, out_dim=5,
                scope_name='fc8', activation=None)

        return fc8

    def build_model(self, config):
        self.global_step_tensor = tf.Variable(
            0, trainable=False, name='global_step')
        self.global_step_inc = self.global_step_tensor.assign(
            self.global_step_tensor)

        self.global_epoch_tensor = tf.Variable(
            0, trainable=False, name='global_epoch')
        self.global_epoch_inc = self.global_epoch_tensor.assign(
            self.global_epoch_tensor + 1)

        with tf.name_scope('inputs') as scope:

            self.x_1 = tf.placeholder(
                'float32',
                shape=[
                    None,
                    config.image.size,
                    config.image.size,
                    config.image.channel
                ], name='x_1')

            self.x_2 = tf.placeholder(
                'float32',
                shape=[
                    None,
                    config.image.size,
                    config.image.size,
                    config.image.channel
                ], name='x_2')

            self.y = tf.placeholder(
                'int32', shape=[None], name='y')

            self.trainable = tf.placeholder(
                'bool', name='trainable')

            tf.add_to_collection('inputs', self.x_1)
            tf.add_to_collection('inputs', self.x_2)
            tf.add_to_collection('inputs', self.y)

            self.cnn_1 = self.cnn_networks(self.x_1)
            self.cnn_2 = self.cnn_networks(self.x_2, reuse=True)

            tf.add_to_collection('cnn_1', self.cnn_1)
            tf.add_to_collection('cnn_2', self.cnn_2)

        with tf.name_scope('loss') as scope:
            self.loss = contrastive_loss(
                input_1=self.cnn_1,
                input_2=self.cnn_2,
                label=self.y,
                margin=0.5,
                eps=1e-6)

        with tf.name_scope('train_step') as scope:
            self.optimizer = tf.train.AdamOptimizer(
                config.model.learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(
                    self.loss, global_step=self.global_step_tensor)

        # with tf.name_scope('accuracy'):
        #     prediction = tf.nn.sigmoid(self.logits, name='prediction')
        #     correct_prediction = tf.equal(
        #         tf.argmax(
        #             prediction, axis=1), tf.cast(self.y, dtype=tf.int64))
        #     self.accuracy = tf.reduce_mean(
        #         tf.cast(correct_prediction, tf.float32))

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('loss', self.loss)
        # tf.add_to_collection('accuracy', self.accuracy)

    def init_saver(self):
        self.saver = tf.train.Saver(
            max_to_keep=self.config.saved_model.max_to_keep,
            save_relative_paths=True)
