import tensorflow as tf


def conv_layer(inputs, filters, k_size,
               stride, padding, scope_name, 
               reuse=False, active=True):

    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        in_channels = inputs.shape[-1]

        kernel = tf.get_variable(
            'kernel',
            [k_size, k_size, in_channels, filters],
            initializer=tf.contrib.layers.xavier_initializer())
        biases = tf.get_variable(
            'biases', [filters],
            initializer=tf.random_normal_initializer())
        conv = tf.nn.bias_add(
            tf.nn.conv2d(
                inputs,
                kernel,
                strides=[1, stride, stride, 1],
                padding=padding),
            biases)

        if active:
            f_conv = tf.nn.relu(conv)

    return f_conv


def maxpool(inputs, k_size, stride, 
            padding='VALID', scope_name='maxpool'):

    with tf.name_scope(scope_name) as scope:
        pool = tf.nn.max_pool(
            inputs, 
            ksize=[1, k_size, k_size, 1],
            strides=[1, stride, stride, 1], 
            padding=padding)

    return pool


def fully_connected(inputs, out_dim,
                    scope_name, reuse,
                    activation=tf.nn.sigmoid):

    with tf.variable_scope(scope_name, reuse=reuse) as scope:
        in_dim = inputs.shape[-1]
        w = tf.get_variable(
            'weights', [in_dim, out_dim],
            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(
            'biases', [out_dim],
            initializer=tf.random_normal_initializer())
        out = tf.nn.bias_add(tf.matmul(inputs, w), b, name='linear')

        if activation:
            out = activation(out, name='activation')
        return out
