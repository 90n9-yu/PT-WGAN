import os
import numpy as np
from util import slices_stack, get_patches


dcm_input_folder = r"D:\Team's Server\Project\TRPMS\Data\Dicom\input"
dcm_label_folder = r"D:\Team's Server\Project\TRPMS\Data\Dicom\label"

train_npy_input_folder = r"D:\Team's Server\Project\TRPMS\Data\Numpy\Train\1\input"
train_npy_label_folder = r"D:\Team's Server\Project\TRPMS\Data\Numpy\Train\1\label"

test_npy_input_folder = r"D:\Team's Server\Project\TRPMS\Data\Numpy\Test\1\input"
test_npy_label_folder = r"D:\Team's Server\Project\TRPMS\Data\Numpy\Test\1\label"

train_indexes = ['1', '2', '3', '5']
test_indexes = ['4']

###########################################################
# Process training data: slices stack and extract patches #
###########################################################

start_index = 1
for i in train_indexes:
    input_dir = os.path.join(dcm_input_folder, str(i))
    label_dir = os.path.join(dcm_label_folder, str(i))
    num_file = len(os.listdir(input_dir))
    for j in range(9, num_file+1):
        input_slices = slices_stack(j, input_dir)
        label_slices = slices_stack(j, label_dir)
        start_index = get_patches(start_index, train_npy_input_folder, train_npy_label_folder,
                                  input_slices, label_slices)


######################################
# Process testing data: slices stack #
######################################

start_index = 1
for i in test_indexes:
    input_dir = os.path.join(dcm_input_folder, str(i))
    label_dir = os.path.join(dcm_label_folder, str(i))
    num_file = len(os.listdir(input_dir))
    for j in range(9, num_file+1):
        input_slices = slices_stack(j, input_dir)
        np.save(os.path.join(test_npy_input_folder, str(start_index)+'.npy'), input_slices)
        label_slices = slices_stack(j, label_dir)
        np.save(os.path.join(test_npy_label_folder, str(start_index)+'.npy'), label_slices)
        start_index += 1