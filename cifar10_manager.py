import file_manager
import download_service
import os
import glob
import numpy as np
from structure import Data
import random

_url_CIFAR10 = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
_unpacked_CIFAR10 = 'cifar-10-batches-py'
_max_train_samples = 50000
_max_test_samples = 10000
_img_width = 32
_img_height = 32
_img_channels = 3
_num_classes=10
_class_names=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']


def download_or_load_CIFAR10(target_dir):
    _, filename = os.path.split(_url_CIFAR10)
    target_file = os.path.join(target_dir, filename)
    target_folder = os.path.join(target_dir, _unpacked_CIFAR10)
    download_service.download_from_url_if_not_exists(_url_CIFAR10, target_dir)
    file_manager.unpack(target_file)
    train_images, train_labels = _load_data(target_folder, 'data_batch*', _max_train_samples)
    test_images, test_labels = _load_data(target_folder, 'test_batch*', _max_test_samples)
    return Data(train_images, train_labels, test_images, test_labels)


def _load_data(folder, file_pattern, max_samples):
    images = np.zeros(shape=[max_samples, _img_height, _img_height, _img_channels], dtype='uint8')
    labels = np.zeros(shape=[max_samples], dtype='uint8')
    index_start = 0
    for file in glob.glob(os.path.join(folder, file_pattern)):
        print('Processing file %s ' % file)
        dict = file_manager.unpickle(file)
        raw_images = dict[b'data']
        raw_labels = dict[b'labels']
        converted_images = _convert_images(raw_images)

        index_end = index_start + raw_images.shape[0]
        images[index_start:index_end, :, :, :] = converted_images
        labels[index_start:index_end] = raw_labels

        index_start = index_end
    return (images, labels)


def _convert_images(raw):
    images = raw.reshape([-1, _img_channels, _img_width, _img_height])
    images = images.transpose([0, 2, 3, 1])
    return images


def build_dictionary_with_images_per_class(images,labels,num_samples):
    dict={}
    for class_nr in range(0,_num_classes):
        img_positions=np.where(labels==class_nr)[0]
        selected_positions=random.sample(list(img_positions),num_samples)
        selected_images=images[selected_positions]
        dict[class_nr]=selected_images
    return dict

def get_class_names():
    return _class_names