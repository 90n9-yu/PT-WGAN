import numpy as np
import tensorflow as tf
from tensorflow.layers import conv3d, conv3d_transpose, dense
from tensorflow.nn import relu
from tensorflow.contrib.layers import xavier_initializer as init
from tensorflow.contrib.layers import flatten


def generator(inputs):
    '''
    generator (3D)

    inputs: [N, 9, 64, 64, 1]
    '''
    skip_1 = inputs
    output_1 = conv3d(inputs, 32, (3, 3, 3), 1, 'same', use_bias=False, kernel_initializer=init(), name='conv1')
    output_1 = relu(output_1)
    output_1 = conv3d(output_1, 32, (3, 3, 3), 1, 'same', use_bias=False, kernel_initializer=init(), name='conv2')
    output_1 = relu(output_1)
    skip_2 = output_1
    output_2 = conv3d(output_1, 32, (1, 3, 3), 1, 'same', use_bias=False, kernel_initializer=init(), name='conv3')
    output_2 = relu(output_2)
    output_2 = conv3d(output_2, 32, (1, 3, 3), 1, 'same', use_bias=False, kernel_initializer=init(), name='conv4')
    output_2 = relu(output_2)
    skip_3 = output_2
    output_3 = conv3d(output_2, 32, (1, 3, 3), 1, 'same', use_bias=False, kernel_initializer=init(), name='conv5')
    output_3 = relu(output_3)
    output_3 = conv3d_transpose(output_3, 32, (1, 3, 3), 1, 'same', use_bias=False, kernel_initializer=init(),
                                name='deconv1')
    output_3 += skip_3
    output_3 = relu(output_3)
    output_4 = conv3d_transpose(output_3, 32, (1, 3, 3), 1, 'same', use_bias=False, kernel_initializer=init(),
                                name='deconv2')
    output_4 = relu(output_4)
    output_4 = conv3d_transpose(output_4, 32, (1, 3, 3), 1, 'same', use_bias=False, kernel_initializer=init(),
                                name='deconv3')
    output_4 += skip_2
    output_4 = relu(output_4)
    output_5 = conv3d_transpose(output_4, 32, (3, 3, 3), 1, 'same', use_bias=False, kernel_initializer=init(),
                                name='deconv4')
    output_5 = relu(output_5)
    output_5 = conv3d_transpose(output_5, 1, (3, 3, 3), 1, 'same', use_bias=False, kernel_initializer=init(),
                                name='deconv5')
    output_5 += skip_1
    output_5 = relu(output_5)
    return output_5


def leaky_relu(inputs, alpha):
    return 0.5 * (1 + alpha) * inputs + 0.5 * (1-alpha) * tf.abs(inputs)


def discriminator(inputs):
    outputs = conv3d(inputs, 64, 3, strides=(1, 2, 2), padding='valid', kernel_initializer=init(), name='conv1')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = conv3d(outputs, 128, 3, strides=(1, 2, 2), padding='valid', kernel_initializer=init(), name='conv2')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = conv3d(outputs, 256, 3, strides=(1, 2, 2), padding='valid', kernel_initializer=init(), name='conv3')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = conv3d(outputs, 512, 3, strides=(1, 2, 2), padding='valid',  kernel_initializer=init(), name='conv4')
    outputs = leaky_relu(outputs, alpha=0.2)

    outputs = flatten(outputs)
    outputs = dense(outputs, units=1024, name='dense1')
    outputs = leaky_relu(outputs, alpha=0.2)
    outputs = dense(outputs, units=1, name='dense2')
    return outputs