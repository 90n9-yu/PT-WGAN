import argparse

parser = argparse.ArgumentParser()

parser.add_argument('--gpu-index', type=str, default='0', help='available GPU index')

parser.add_argument('--batch-size', type=int, default=64, help='batch size')
parser.add_argument('--image-depth', type=int, default=9, help='depth of the image')
parser.add_argument('--image-size', type=int, default=64, help='height or length of the image')
parser.add_argument('--image-channel', type=int, default=1, help='number of channels of the image')

parser.add_argument('--start-epoch', type=int, default=0,
                    help='If you choose to train the network from scratch, set it to 0.'
                         'Otherwise, set it to the corresponding serial number of the model you plan to load.')
parser.add_argument('--end-epoch', type=int, default=100, help='The maximum number of epochs.')
parser.add_argument('--disc-iters', type=int, default=4,
                    help='Asynchronous training generator and discriminator.'
                         'The number of training iteration of discriminator')

parser.add_argument('--checkpoint-path', type=str, default="",
                    help='The storage path of the checkpoint file.')
parser.add_argument('--log-path', type=str, default="",
                    help='The storage path of the TensorBoard log file.')

parser.add_argument('--learning-rate-G', type=float, default=1e-5, help='The learning rate of generator.')
parser.add_argument('--learning-rate-D', type=float, default=1e-4, help='The learning rate of discriminator.')
parser.add_argument('--beta-1', type=float, default=0.9, help='Hyperparameter beta_1 for the optimizer.')
parser.add_argument('--beta-2', type=float, default=0.999, help='Hyperparameter beta_2 for the optimizer.')
parser.add_argument('--lambda-m', type=float, default=1e7, help='Hyperparameter for MSE loss.')

parser.add_argument('--normalization-model', type=int, default=1,
                    help='Selection for data normalization.'
                         'If set it to 1, the value range of the data is [0, 1].'
                         'If set it to 2, the value range of the data is [0, 4.52]')

parser.add_argument('--training-data-path', type=str,
                    default="",
                    help='The storage path of the training data (.h5).')
parser.add_argument('--pretrained-ckpt-path', type=str,
                    default="",
                    help='The storage path of the pretrained checkpoint file.')
parser.add_argument('--save-path', type=str,
                    default="",
                    help='The storage path of the trained checkpoint file.')

parser.add_argument('--testing-input-path', type=str, default="",
                    help='The storage path of the testing input data (.h5).')
parser.add_argument('--testing-label-path', type=str, default="",
                    help='The storage path of the testing label data (.h5).')
parser.add_argument('--testing-output-path', type=str, default="",
                    help='The storage path of the testing output data (.h5).')

parser.add_argument('--model-name', type=str, default='', help='The name of model.')
parser.add_argument('--result-record-path', type=str, default="",
                    help='The storage path of the training process record file.')