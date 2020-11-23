import pickle
import numpy as np
import tensorflow as tf
from skimage.measure import compare_mse, compare_nrmse, compare_ssim, compare_psnr


def ckpt_to_numpy(checkpoint_dir, save_name = 'tmp_generate_weight'):
    v_names = ['conv1', 'conv2', 'conv3', 'conv4', 'conv5', 'deconv1', 'deconv2', 'deconv3', 'deconv4', 'deconv5']

    weights = dict()
    with tf.Session() as sess:
        for var_name, _ in tf.contrib.framework.list_variables(checkpoint_dir):
            var_name_list = var_name.split('/')
            if len(var_name_list) == 3 and var_name_list[0] == 'generator':
                weights[var_name_list[1]] = tf.contrib.framework.load_variable(checkpoint_dir, var_name)

    pickle.dump(weights, open(save_name, 'wb'))
    return save_name


def normalization(array, model):
    """
    :param array: array (N, 9, 64, 64, 1)
    :param model: 1 - [0, 1], 2 - [0, 4.515]
    :return: normalized array
    """
    if model == 1:
        array = (array - 0.0) / (32767.0 - 0.0)
        return array
    elif model == 2:
        array[array <= 1] = 1
        array = np.log10(array)
        return array
    else:
        raise ValueError('Invalid number for normalization model selection.')


def cal_mse(labels, outputs, model):
    if model == 1:
        data_range = 1
    elif model == 2:
        data_range = 4.52
    else:
        raise ValueError('Invalid number for normalization model selection.')
    labels = np.squeeze(labels, axis=-1)
    outputs = np.squeeze(outputs, axis=-1)
    num_slices = labels.shape[0] * labels.shape[1]
    running_mse = 0.0
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            running_mse += compare_mse(labels[i, j, ...], outputs[i, j, ...])
    avg_mse = running_mse / num_slices
    return avg_mse


def cal_nrmse(labels, outputs, model):
    if model == 1:
        data_range = 1
    elif model == 2:
        data_range = 4.52
    else:
        raise ValueError('Invalid number for normalization model selection.')
    labels = np.squeeze(labels, axis=-1)
    outputs = np.squeeze(outputs, axis=-1)
    num_slices = labels.shape[0] * labels.shape[1]
    running_nrmse = 0.0
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            running_nrmse += compare_nrmse(labels[i, j, ...], outputs[i, j, ...])
    avg_nrmse = running_nrmse / num_slices
    return avg_nrmse


def cal_ssim(labels, outputs, model):
    if model == 1:
        data_range = 1
    elif model == 2:
        data_range = 4.52
    else:
        raise ValueError('Invalid number for normalization model selection.')
    labels = np.squeeze(labels, axis=-1)
    outputs = np.squeeze(outputs, axis=-1)
    num_slices = labels.shape[0] * labels.shape[1]
    running_ssim = 0.0
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            running_ssim += compare_ssim(labels[i, j, ...], outputs[i, j, ...], data_range=data_range)
    avg_ssim = running_ssim / num_slices
    return avg_ssim


def cal_psnr(labels, outputs, model):
    if model == 1:
        data_range = 1
    elif model == 2:
        data_range = 4.52
    else:
        raise ValueError('Invalid number for normalization model selection.')
    labels = np.squeeze(labels, axis=-1)
    outputs = np.squeeze(outputs, axis=-1)
    num_slices = labels.shape[0] * labels.shape[1]
    running_psnr = 0.0
    for i in range(labels.shape[0]):
        for j in range(labels.shape[1]):
            running_psnr += compare_psnr(labels[i, j, ...], outputs[i, j, ...], data_range=data_range)
    avg_psnr = running_psnr / num_slices
    return avg_psnr