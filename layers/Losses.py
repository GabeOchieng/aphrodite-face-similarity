import tensorflow as tf


def contrastive_loss(input_1, input_2, y, margin, eps):
    with tf.name_scope('contrastive_loss'):
        distance = tf.sqrt(eps + tf.reduce_sum(
            tf.pow(input_1 - input_2, 2), 1, keepdims=True))
        similarity = y * tf.square(distance)
        disimilarity = (1 - y) * tf.square(
            tf.maximum(margin - distance, 0))
        loss = tf.reduce_mean(disimilarity + similarity) / 2
    return loss


def triplet_loss(acnchor, positive, negative):
    d_pos = tf.reduce_sum(
        tf.square(anchor_output - positive_output), 1)
    d_neg = tf.reduce_sum(
        tf.square(anchor_output - negative_output), 1)

    loss = tf.maximum(0., margin + d_pos - d_neg)
    loss = tf.reduce_mean(loss)
    
    return loss


def eculidean_distance(input_1, input_2):
    return tf.sqrt(
        tf.reduce_sum(tf.pow(input_1, input_2, 2), 1, keepdims=True))
