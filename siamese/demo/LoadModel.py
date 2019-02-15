import os
import cv2
import numpy as np
import tensorflow as tf
from model.ConvNet import ConvNet
from layers.Losses import euclidean_distance
from utils.Parser import process_config
from dataloader.SiameseDataLoader import SiameseDataLoader


checkpoint = tf.train.latest_checkpoint(
    'experiments/aphrodite/checkpoint/')

tf.reset_default_graph()

def print_all_variables_name():
    for v in tf.global_variables():
        print(v)
        print(v.name)

def print_all_operations_name():
    for n in tf.get_default_graph().as_graph_def().node:
        print(n.name)

with tf.Session() as sess:
    saver = tf.train.import_meta_graph(checkpoint + '.meta')
    saver.restore(sess, checkpoint)
    graph = tf.get_default_graph()

    inputs = graph.get_tensor_by_name('inputs/x_1:0')
    trainable = graph.get_tensor_by_name('inputs/trainable:0')
    embeddings = graph.get_collection('cnn_1')

    beta = graph.get_tensor_by_name('inputs/network/norm3/beta:0')
    gamma = graph.get_tensor_by_name('inputs/network/norm3/gamma:0')

    # print_all_variables_name()
    # print_all_operations_name()

    im_1 = cv2.imread('../dataset/att_faces/train/s12/2.pgm')
    im_2 = cv2.imread('../dataset/att_faces/train/s13/2.pgm')

    im_1 = cv2.cvtColor(im_1, cv2.COLOR_BGR2GRAY)
    im_2 = cv2.cvtColor(im_2, cv2.COLOR_BGR2GRAY)

    im_1 = cv2.bitwise_not(im_1)
    im_2 = cv2.bitwise_not(im_2)

    im_1 = cv2.resize(im_1, (100, 100))
    im_2 = cv2.resize(im_2, (100, 100))

    im_1 = np.expand_dims(np.expand_dims(im_1, axis=0), axis=3)
    im_2 = np.expand_dims(np.expand_dims(im_2, axis=0), axis=3)

    print(im_1.shape)
    print(im_2.shape)

    cnn1 = sess.run(
        embeddings, 
        feed_dict={inputs: im_1, trainable: True})
    cnn2 = sess.run(
        embeddings, 
        feed_dict={inputs: im_2, trainable: True})

    distance = np.linalg.norm(np.array(cnn1)-np.array(cnn2))

    print(cnn1)
    print(cnn2)
    print(distance)
