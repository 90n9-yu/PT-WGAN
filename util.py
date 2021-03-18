import os
import pydicom as dcm
import numpy as np
from skimage.measure import compare_mse, compare_nrmse, compare_psnr, compare_ssim


def slices_stack(index, source_path):
    """
    :param index: index of start slice
    :param source_path: path of the folder stored dicom files
    :return: np.array (9, 256, 256, 1)
    """
    slices = []
    for i in range(index-8, index+1):
        slices.append(dcm.read_file(os.path.join(source_path, str(i)+'.dcm')).pixel_array)
    slices = np.array(slices)
    slices = np.expand_dims(slices, axis=-1)
    return slices


def get_patches(index, input_store_path, label_store_path, input_array, label_array):
    """
    :param index: start index of patches
    :param input_store_path: store folder path of input patches
    :param label_store_path: store folder path of label patches
    :param input_array: input array (9, 256, 256, 1)
    :param label_array: label array (9, 256, 256, 1)
    :param patch_size: patch size
    :param stride: patch window stride
    :return: end index of patches
    """
    init_center_x = 64 // 2 - 1
    init_center_y = 64 // 2 - 1
    num_zeros = input_array.shape[0] * 64 * 64 * 0.8
    for coordinate_y in range(init_center_y, input_array.shape[1] - 64 // 2, 16):
        for coordinate_x in range(init_center_x, input_array.shape[1] - 64 // 2, 16):
            input_patch = input_array[:, coordinate_x - 31: coordinate_x + 33, coordinate_y - 31: coordinate_y + 33, :]
            label_patch = label_array[:, coordinate_x - 31: coordinate_x + 33, coordinate_y - 31: coordinate_y + 33, :]
            if np.sum(input_patch == 0) > num_zeros or np.sum(label_patch == 0) > num_zeros:
                pass
            else:
                np.save(os.path.join(input_store_path, str(index) + '.npy'), input_patch)
                np.save(os.path.join(label_store_path, str(index) + '.npy'), label_patch)
                index += 1
    return index


def normalization(array, model):
    """
    :param array: array (N, 64, 64, 1)
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