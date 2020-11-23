import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from model import generator
from config import parser
from util import normalization


def main(args):
    os.makedirs(args.testing_output_path, exist_ok=True)
    X = tf.placeholder(dtype=tf.float32, shape=[1, 9, 256, 256, 1])

    with tf.variable_scope('generator') as scope:
        Y_ = generator(X)

    sess = tf.Session()
    saver = tf.train.Saver()
    sess_path = args.checkpoint_path + 'Epoch=' + str(args.start_epoch) + args.model_name + '.ckpt'
    saver.restore(sess, sess_path)

    num_file = len(os.listdir(args.testing_input_path))
    for i in range(1, num_file+1):
        input_slice = np.load(os.path.join(args.testing_input_path, str(i)+'.npy'))
        input_slice = np.expand_dims(input_slice, axis=0)
        input_slice = normalization(input_slice, args.normalization_model)

        output_slice = sess.run(Y_, feed_dict={X: input_slice})

        output_slice = np.squeeze(output_slice, axis=(0, -1))
        output_slice = output_slice[4, ...]

        np.save(os.path.join(args.testing_output_path, str(i) + '_output.npy'), output_slice)


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    main(args)
