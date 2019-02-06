import numpy as np
import tensorflow as tf
from base.BaseModel import BaseModel
from layers.Convolutions import conv_layer
from layers.Losses import contrastive_loss
from layers.losses import euclidean_distance


class ConvNet(BaseModel):
    def __init__(self, data_loader,
                 config, trainable=True, retrain='complete'):
        super(VGG16, self).__init__(config)
        self.data_loader = data_loader
        self.n_classes = config.num_classes
        self.data_dict = np.load(
            config.weights,
            encoding='latin1').item()
        self.retrain = retrain
        self.trainable = trainable
        self.logits = None
        self.logits_argmax = None
        self.loss = None
        self.acc = None
        self.optimizer = None
        self.train_step = None
        self.build_model(config)
        self.init_saver()

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
                    config.image_shape,
                    config.image_shape,
                    config.image_channels
                    ],
                name='x')
            self.x_2 = tf.placeholder(
                'float32',
                shape=[
                    None,
                    config.image_shape,
                    config.image_shape,
                    config.image_channels
                    ],
                name='x')
            self.y = tf.placeholder('int32', shape=[None], name='y')

            tf.add_to_collection('inputs', self.x)
            tf.add_to_collection('inputs', self.y)

            self.cnn_1 = cnn_networks(self.x_1)
            self.cnn_2 = cnn_networks(self.x_2, reuse=True)

            tf.add_to_collection('cnn_1', self.cnn_1)
            tf.add_to_collection('cnn_2', self.cnn_2)

        with tf.name_scope('loss') as scope:
            self.loss = contrastive_loss(
                input_1=self.cnn_1,
                input_2=self.cnn_2,
                y=self.y,
                margin=0.5,
                eps=0.2)

        with tf.name_scope('train_step') as scope:
            self.optimizer = tf.train.GradientDescentOptimizer(
                config.learning_rate)

            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

            with tf.control_dependencies(update_ops):
                self.train_step = self.optimizer.minimize(
                    self.loss, global_step=self.global_step_tensor)

        with tf.name_scope('accuracy'):
            prediction = euclidean_distance(self.cnn_1, self.cnn_2)
            correct_prediction = tf.equal(
                tf.round(
                    prediction), tf.cast(self.y, dtype=tf.int64))
            self.accuracy = tf.reduce_mean(
                tf.cast(correct_prediction, tf.float32))

        tf.add_to_collection('train', self.train_step)
        tf.add_to_collection('loss', self.loss)
        tf.add_to_collection('acc', self.accuracy)

    def cnn_networks(x, reuse=False):
        with tf.name_scope('network') as scope:
            conv1 = self.conv_layer(
                x, filters=32, k_size=7,
                stride=1, padding='SAME',
                scope_name='conv1', reuse=reuse)
            pool1 = self.maxpool(
                conv1, k_size=2, stride=2,
                padding='SAME', scope_name='pool1')

            conv2 = self.conv_layer(
                pool1, filters=64, k_size=5,
                stride=1, padding='SAME',
                scope_name='conv2', reuse=reuse)
            pool2 = self.maxpool(
                conv1, k_size=2, stride=2,
                padding='SAME', scope_name='pool2')

            conv3 = self.conv_layer(
                pool1, filters=128, k_size=3,
                stride=1, padding='SAME',
                scope_name='conv3', reuse=reuse)
            pool3 = self.maxpool(
                conv1, k_size=2, stride=2,
                padding='SAME', scope_name='pool3')

            conv4 = self.conv_layer(
                pool1, filters=256, k_size=1,
                stride=1, padding='SAME',
                scope_name='conv4', reuse=reuse)
            pool4 = self.maxpool(
                conv1, k_size=2, stride=2,
                padding='SAME', scope_name='pool4')

            conv5 = self.conv_layer(
                pool1, filters=2, k_size=1,
                stride=1, padding='SAME',
                scope_name='conv5', reuse=reuse)
            pool5 = self.maxpool(
                conv1, k_size=2, stride=2,
                padding='SAME', scope_name='pool5')

            cur_dim = pool5.get_shape()
            pool5_dim = cur_dim[1] * cur_dim[2] * cur_dim[3]
            pool5_flatten = tf.reshape(pool5, shape=[-1, pool5_dim])

        return pool5_flatten

    def init_saver(self):
        self.saver = tf.train.Saver(
            max_to_keep=self.config.max_to_keep,
            save_relative_paths=True)
