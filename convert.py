import os
import h5py
import numpy as np

train_npy_input_folder = r"D:\Team's Server\Project\TRPMS\Data\Numpy\Train\1\input"
train_npy_label_folder = r"D:\Team's Server\Project\TRPMS\Data\Numpy\Train\1\label"
h5py_file_path = r"D:\Team's Server\Project\TRPMS\Data\Numpy\Train\1\training_40000.h5"


indexes = np.random.choice(range(1, 165420), 40000, replace=False)

input_patches = []
label_patches = []

for i in indexes:
    input_patch = np.load(os.path.join(train_npy_input_folder, str(i) + '.npy'))
    input_patches.append(input_patch)
    label_patch = np.load(os.path.join(train_npy_label_folder, str(i) + '.npy'))
    label_patches.append(label_patch)

input_patches = np.array(input_patches)
label_patches = np.array(label_patches)
print(input_patches.shape)
print(label_patches.shape)

File = h5py.File(h5py_file_path, 'w')
File['inputs'] = input_patches
File['labels'] = label_patches
File.close()