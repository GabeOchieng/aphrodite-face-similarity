import tensorflow as tf


def contrastive_loss(input_1, input_2, label, margin, eps):
    with tf.name_scope('contrastive_loss') as scope:
        y = tf.to_float(label)
        distance = euclidean_distance(
            input_1, input_2, eps)
        similarity = y * tf.square(distance)
        disimilarity = (1.0 - y) * tf.square(
            tf.maximum((margin - distance), 0.0))
        loss = tf.reduce_mean(
            disimilarity + similarity) / 2.0
    return loss


def DBL_loss(input_1, input_2, label, margin, eps):
    with tf.name_scope('distance_based_logistic') as scope:
        y = tf.expand_dims(tf.to_float(label), axis=1)
        distance = euclidean_distance(
            input_1, input_2, eps)
        p = (1.0 + tf.exp(-margin)) / (1 + tf.exp(distance - margin))
        loss = tf.losses.log_loss(y, p)
    return loss


def triplet_loss(acnchor, positive, negative):
    d_pos = tf.reduce_sum(
        tf.square(anchor_output - positive_output), 1)
    d_neg = tf.reduce_sum(
        tf.square(anchor_output - negative_output), 1)

    loss = tf.maximum(0., margin + d_pos - d_neg)
    loss = tf.reduce_mean(loss)
    
    return loss


def euclidean_distance(input_1, input_2, eps=1e-6):
    with tf.name_scope('euclidean_distance') as scope:
        distance = tf.sqrt(
            eps + tf.reduce_sum(
                tf.square(tf.subtract(
                    input_1, input_2)), axis=1, keepdims=True))
    return distance
