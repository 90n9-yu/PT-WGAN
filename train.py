import os
import csv
import h5py
import time
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.train import AdamOptimizer as Optimizer
from sklearn.utils import shuffle

from config import parser
from model import generator, discriminator
from util import ckpt_to_numpy, normalization, cal_nrmse, cal_ssim, cal_psnr


def main(args):
    if os.path.exists(args.result_record_path):
        os.remove(args.result_record_path)
    else:
        with open(args.result_record_path, 'w') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Epoch', 'Training Time', 'Learning Rate_G', 'Learning Rate_D',
                                 'Train_Loss_G', 'Train_Loss_D', 'W_Distance'
                                 'Train_MSE', 'Train_NRMSE', 'Train_SSIM', 'Train_PSNR'])
    os.makedirs(args.checkpoint_path, exist_ok=True)
    os.makedirs(args.log_path, exist_ok=True)
    os.makedirs(args.save_path, exist_ok=True)

    X = tf.placeholder(dtype=tf.float32,
                       shape=[args.batch_size, args.image_depth, args.image_size, args.image_size, args.image_channel])
    Y = tf.placeholder(dtype=tf.float32,
                       shape=[args.batch_size, args.image_depth, args.image_size, args.image_size, args.image_channel])

    with tf.variable_scope('generator') as scope:
        Y_ = generator(X)

    alpha = tf.random_uniform(shape=[args.batch_size, 1], minval=0., maxval=1.)

    with tf.variable_scope('discriminator') as scope:
        disc_real = discriminator(Y)
        scope.reuse_variables()
        disc_fake = discriminator(Y_)
        interpolates = alpha * tf.reshape(Y, [args.batch_size, -1]) + \
                       (1 - alpha) * tf.reshape(Y_, [args.batch_size, -1])
        interpolates = tf.reshape(interpolates, [args.batch_size, args.image_depth, args.image_size, args.image_size, 1])
        gradients = tf.gradients(discriminator(interpolates), [interpolates])[0]

    slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis=1))
    gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
    difference = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)

    gen_cost = -tf.reduce_mean(disc_fake)

    disc_loss = difference + 10 * gradient_penalty

    gen_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='generator')
    disc_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='discriminator')

    if args.normalization_model == 1:
        max_val = 1.0
    elif args.normalization_model == 2:
        max_val = 4.52

    mse_cost = tf.reduce_sum(tf.squared_difference(Y_, Y)) / (args.batch_size * args.image_depth * args.image_size * args.image_size)
    nrmse_cost = tf.placeholder(dtype=tf.float32, shape=[])
    ssim_cost = tf.placeholder(dtype=tf.float32, shape=[])
    psnr_cost = tf.placeholder(dtype=tf.float32, shape=[])

    gen_loss = gen_cost + args.lambda_m * mse_cost

    summary_mse = tf.summary.scalar('MSE', mse_cost)
    summary_nrmse = tf.summary.scalar('NRMSE', nrmse_cost)
    summary_ssim = tf.summary.scalar('SSIM', ssim_cost)
    summary_psnr = tf.summary.scalar('PSNR', psnr_cost)
    summary_loss_g = tf.summary.scalar('Gen_Loss', gen_loss)
    summary_loss_d = tf.summary.scalar('Dis_Loss', disc_loss)
    summary_w_distance = tf.summary.scalar('W_Distance', difference)

    lr_G = tf.placeholder(tf.float32, shape=[])
    lr_D = tf.placeholder(tf.float32, shape=[])

    gen_train_op = Optimizer(learning_rate=lr_G, beta1=args.beta_1, beta2=args.beta_2).minimize(
        gen_loss, var_list=gen_params)
    disc_train_op = Optimizer(learning_rate=lr_D, beta1=args.beta_1, beta2=args.beta_2).minimize(
        disc_loss, var_list=disc_params)

    sess = tf.Session()
    writer = tf.summary.FileWriter(args.log_path, sess.graph)
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver(max_to_keep=None)

    print("Loading data")
    start_time = time.time()
    File = h5py.File(args.training_data_path, 'r')
    inputs, labels = np.array(File['inputs']), np.array(File['labels'])
    inputs = normalization(inputs, args.normalization_model)
    labels = normalization(labels, args.normalization_model)
    File.close()
    print('Loading data takes {:.3f}s.'.format(time.time() - start_time))

    print('Number of parameters 2.5D-WGAN network: {} '.
          format(np.sum([np.prod(v.get_shape().as_list()) for v in tf.trainable_variables(scope='generator')])))

    print("Transferring CNN(MSE) model to WGAN")
    generate_weight = ckpt_to_numpy(args.pretrained_ckpt_path, save_name=args.save_path + 'MSE')
    g_weights = pickle.load(open(generate_weight, 'rb'))
    v_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'deconv1', 'deconv2', 'deconv3', 'deconv4', 'deconv5']
    for i in range(len(g_weights)):
        sess.run(gen_params[i].assign(g_weights[v_names[i]]))

    print("Start training ... ")
    global_step = 1
    for epoch in range(args.start_epoch, args.end_epoch + 1):
        val_lr_G = args.learning_rate_G / np.sqrt((epoch + 1))
        val_lr_D = args.learning_rate_D / (epoch + 1)

        inputs, labels = shuffle(inputs, labels)
        num_batches = inputs.shape[0] // args.batch_size

        train_running_mse = 0.0
        train_running_nrmse = 0.0
        train_running_ssim = 0.0
        train_running_psnr = 0.0
        train_running_loss_G = 0.0
        train_running_loss_D = 0.0
        train_running_w_distance = 0.0

        start_time = time.time()
        for i in range(num_batches):
            for j in range(args.disc_iters):
                idx = np.random.permutation(inputs.shape[0])
                batch_inputs = inputs[idx[:args.batch_size]]
                batch_labels = labels[idx[:args.batch_size]]
                sess.run([disc_train_op], feed_dict={X: batch_inputs,
                                                     Y: batch_labels,
                                                     lr_D: val_lr_D})

            batch_inputs = inputs[i * args.batch_size: (i + 1) * args.batch_size]
            batch_labels = labels[i * args.batch_size: (i + 1) * args.batch_size]

            _, gen_loss_, disc_loss_, difference_, mse_cost_, batch_outputs,\
            summary_loss_g_, summary_loss_d_, summary_w_distance_, summary_mse_ = \
                sess.run([gen_train_op, gen_loss, disc_loss, difference, mse_cost, Y_,
                          summary_loss_g, summary_loss_d, summary_w_distance, summary_mse],
                         feed_dict={X: batch_inputs,
                                    Y: batch_labels,
                                    lr_G: val_lr_G})

            nrmse_cost_ = cal_nrmse(batch_labels, batch_outputs, model=args.normalization_model)
            ssim_cost_ = cal_ssim(batch_labels, batch_outputs, model=args.normalization_model)
            psnr_cost_ = cal_psnr(batch_labels, batch_outputs, model=args.normalization_model)

            nrmse_cost_, ssim_cost_, psnr_cost_, summary_nrmse_, summary_ssim_, summary_psnr_ = sess.run([
                nrmse_cost, ssim_cost, psnr_cost, summary_nrmse, summary_ssim, summary_psnr],
                feed_dict={nrmse_cost: nrmse_cost_,
                           ssim_cost: ssim_cost_,
                           psnr_cost: psnr_cost_})

            train_running_mse += mse_cost_
            train_running_nrmse += nrmse_cost_
            train_running_ssim += ssim_cost_
            train_running_psnr += psnr_cost_
            train_running_loss_G += gen_loss_
            train_running_loss_D += disc_loss_
            train_running_w_distance += difference_

            writer.add_summary(summary_mse_, global_step)
            writer.add_summary(summary_nrmse_, global_step)
            writer.add_summary(summary_ssim_, global_step)
            writer.add_summary(summary_psnr_, global_step)
            writer.add_summary(summary_loss_g_, global_step)
            writer.add_summary(summary_loss_d_, global_step)
            writer.add_summary(summary_w_distance_, global_step)

            global_step += 1

        train_mse = train_running_mse / num_batches
        train_nrmse = train_running_nrmse / num_batches
        train_ssim = train_running_ssim / num_batches
        train_psnr = train_running_psnr / num_batches
        train_loss_G = train_running_loss_G / num_batches
        train_loss_D = train_running_loss_D / num_batches
        train_w_distance = train_running_w_distance / num_batches

        training_time = time.time() - start_time

        print('Epoch: {}, Training_time: {:.2f}s, Learning_rate_G: {}, Learning_rate_D: {}'.format(
            epoch + 1, training_time, val_lr_G, val_lr_D)
              + '\nTrain_Loss_G: {}, Train_Loss_D: {}, Wasserstein_Distance: {}'.format(
            train_loss_G, train_loss_D, train_w_distance)
              + '\nTrain_MSE: {:.8f}, Train_NRMSE: {:.8f}, Train_SSIM: {:.3f}, Train_PSNR: {:.3f}'.format(
            train_mse, train_nrmse, train_ssim, train_psnr))

        with open(args.result_record_path, 'a') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow([epoch+1, round(training_time, 2), val_lr_G, val_lr_D,
                                 train_loss_G, train_loss_D, train_w_distance,
                                 round(train_mse, 8), round(train_nrmse, 8),
                                 round(train_ssim, 3), round(train_psnr, 3)])
        saver.save(sess, args.checkpoint_path + 'Epoch=' + str(epoch + 1) + args.model_name + '.ckpt')
    sess.close()


if __name__ == '__main__':
    args = parser.parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_index
    main(args)