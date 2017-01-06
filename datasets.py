import config

import numpy as np
import gnumpy as gnp

import struct
import os
#import scipy.io


class GpuDataset():
    def __init__(self, train_data, test_data, n_used_for_validation, train_labels=None, test_labels=None,
                 shuffle=False, binarized=False):
        if shuffle:
            permutation = np.random.RandomState(123).permutation(train_data.shape[0])
            train_data = train_data[permutation]
            if train_labels is not None:
                train_labels = train_labels[permutation]

        self.data = {}

        self.data['train'] = gnp.garray(train_data[:-n_used_for_validation])
        self.data['validation'] = gnp.garray(train_data[-n_used_for_validation:])
        self.data['test'] = gnp.garray(test_data)

        if train_labels is not None:
            self.labels = {}
            self.labels['train'] = gnp.garray(train_labels[:-n_used_for_validation])
            self.labels['validation'] = gnp.garray(train_labels[-n_used_for_validation:])
            self.labels['test'] = gnp.garray(test_labels)
        else:
            self.labels = None

        self.binarized = binarized

    def get_minibatch_at_index(self, index, minibatch_size, subdataset='train', **kwargs):
        minibatch = self.data[subdataset][index*minibatch_size: (index+1)*minibatch_size]
        return gnp.rand(*minibatch.shape) < minibatch if self.binarized else minibatch

    def get_labels_at_index(self, index, minibatch_size, subdataset='train', **kwargs):
        return self.labels[subdataset][index*minibatch_size: (index+1)*minibatch_size] if self.labels is not None else None

    def get_train_logit_of_mean(self):
        mean = self.get_train_mean().asarray()
        clipped_mean = gnp.garray(np.clip(mean, 0.001, 0.999))
        return -gnp.log(1./clipped_mean-1.)

    def get_train_mean(self):
        return self.data['train'].mean(axis=0)[None, :]

    def get_n_examples(self, subdataset='train'):
        return self.data[subdataset].shape[0]

    def get_data_dim(self, subdataset='train'):
        return self.data[subdataset].shape[1]



def mnist(n_validation=400, load_labels=False, binarized=True):
    def load_mnist_images_np(imgs_filename):
        with open(imgs_filename, 'rb') as f:
            f.seek(4)
            nimages, rows, cols = struct.unpack('>iii', f.read(12))
            dim = rows*cols

            images = np.fromfile(f, dtype=np.dtype(np.ubyte))
            images = (images/255.0).astype('float32').reshape((nimages, dim))

        return images

    def load_mnist_labels_np(labels_filename):
        with open(labels_filename, 'rb') as f:
            f.seek(8)
            labels = np.fromfile(f, dtype=np.dtype(np.ubyte))
        return labels

    train_data = load_mnist_images_np(
        os.path.join(config.DATASETS_DIR, 'MNIST', 'train-images.idx3-ubyte'))
    test_data = load_mnist_images_np(
        os.path.join(config.DATASETS_DIR, 'MNIST', 't10k-images.idx3-ubyte'))
    if load_labels:
        train_labels = load_mnist_labels_np(os.path.join(config.DATASETS_DIR, 'MNIST', 'train-labels.idx1-ubyte'))
        test_labels = load_mnist_labels_np(os.path.join(config.DATASETS_DIR, 'MNIST', 't10k-labels.idx1-ubyte'))

        return GpuDataset(train_data, test_data, n_validation, shuffle=False,
                          train_labels=train_labels, test_labels=test_labels, binarized=binarized)
    else:
        return GpuDataset(train_data, test_data, n_validation, shuffle=False, binarized=binarized)


# def omniglot(n_validation=1345, binarized=True):
#     def reshape_data(data):
#         return data.reshape((-1, 28, 28)).reshape((-1, 28*28), order='fortran')
#     omni_raw = scipy.io.loadmat(
#         os.path.join(config.DATASETS_DIR, 'OMNIGLOT', 'chardata.mat'))
#
#     train_data = reshape_data(omni_raw['data'].T.astype('float32'))
#     test_data = reshape_data(omni_raw['testdata'].T.astype('float32'))
#
#     return GpuDataset(train_data, test_data, n_validation, shuffle=True, binarized=binarized)


def binarized_mnist_fixed_binarization():
    def lines_to_np_array(lines):
        return np.array([[int(i) for i in line.split()] for line in lines])
    with open(os.path.join(config.DATASETS_DIR, 'BinaryMNIST', 'binarized_mnist_train.amat')) as f:
        lines = f.readlines()
    train_data = lines_to_np_array(lines).astype('float32')
    with open(os.path.join(config.DATASETS_DIR, 'BinaryMNIST', 'binarized_mnist_valid.amat')) as f:
        lines = f.readlines()
    validation_data = lines_to_np_array(lines).astype('float32')
    with open(os.path.join(config.DATASETS_DIR, 'BinaryMNIST', 'binarized_mnist_test.amat')) as f:
        lines = f.readlines()
    test_data = lines_to_np_array(lines).astype('float32')

    return GpuDataset(np.concatenate([train_data, validation_data], axis=0), test_data,
                      n_used_for_validation=validation_data.shape[0], shuffle=False, binarized=False)
