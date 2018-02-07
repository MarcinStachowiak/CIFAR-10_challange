import numpy as np
import file_manager
import mahotas as mh

__author__ = "Marcin Stachowiak"
__version__ = "1.0"
__email__ = "marcin.stachowiak.ms@gmail.com"

def _calculate_transfer_values_features_for_image(model, image):
    return model.calculate(model.tensor_input_image_layer, model.tensor_transfer_layer, image)


def _calculate_transfer_values_features_for_images(model, images):
    images_count = len(images)
    output_values = []
    for i in range(images_count):
        print('Calculating transfer values for %d/%d image' % (i, images_count))
        output_values.append(_calculate_transfer_values_features_for_image(model, images[i]))
    return np.array(output_values)

def calculate_or_load_transfer_values_features_for_images(model,images, path_to_csv):
    if file_manager.check_if_file_exists(path_to_csv):
        print('Reading data from file %s' % path_to_csv)
        data = np.genfromtxt(path_to_csv, delimiter=',')
    else:
        data = _calculate_transfer_values_features_for_images(model,images)
        np.savetxt(path_to_csv, data, delimiter=',')
    return data


def calculate_or_load_texture_features_for_images(images, path_to_csv):
    if file_manager.check_if_file_exists(path_to_csv):
        print('Reading data from file %s' % path_to_csv)
        data = np.genfromtxt(path_to_csv, delimiter=',')
    else:
        data = _calculate_texture_features_for_images(images)
        np.savetxt(path_to_csv, data, delimiter=',')
    return data


def _calculate_texture_features_for_images(images):
    images_count = len(images)
    output_values = []
    for i in range(images_count):
        print('Calculating texture features for %d/%d image' % (i, images_count))
        output_values.append(_calculate_texture_features_for_image(images[i]))
    return np.array(output_values)


def _calculate_texture_features_for_image(image):
    vector_haralick = mh.features.haralick(image)
    vector_zernike = mh.features.zernike_moments(image, 32, degree=12)
    vector_lbp = mh.features.lbp(image, 4, 8)

    vector_result = np.concatenate(
        [vector_haralick[0, :], vector_haralick[1, :], vector_haralick[2, :], vector_haralick[3, :], vector_zernike,
         vector_lbp])
    return vector_result


class FearureImageExtractor:
    _models = []

    def __init__(self, traint_images, test_images):
        self._traint_images = traint_images
        self._test_images = test_images

    def perform_transfer_values_extraction(self, model):
        transfer_values_train_x = calculate_or_load_transfer_values_features_for_images(model, self._traint_images,
                                                                                'transfer_values_train_x_dump.csv')
        transfer_values_test_x = calculate_or_load_transfer_values_features_for_images(model, self._test_images,
                                                                               'transfer_values_test_x_dump.csv')
        return (transfer_values_train_x, transfer_values_test_x)

    def perform_texture_features_extraction(self):
        texture_train_x = calculate_or_load_texture_features_for_images(self._traint_images,
                                                                        'texture_train_x_dump.csv')
        texture_test_x = calculate_or_load_texture_features_for_images(self._test_images,
                                                                       'texture_test_x_dump.csv')
        return (texture_train_x, texture_test_x)
