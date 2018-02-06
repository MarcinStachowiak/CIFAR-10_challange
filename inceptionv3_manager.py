import os
import download_service
import file_manager
import tensorflow as tf
import numpy as np

_inceptionv3_url = "http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz"
_inceptionv3_model_file = 'classify_image_graph_def.pb'


def download_or_load_Inceptionv3(target_dir):
    _, filename = os.path.split(_inceptionv3_url)
    target_file = os.path.join(target_dir, filename)
    download_service.download_from_url_if_not_exists(_inceptionv3_url, target_dir)
    file_manager.unpack(target_file)
    model_path = os.path.join(target_dir, _inceptionv3_model_file)
    return Inception(model_path)


class Inception:
    tensor_input_image_layer = "DecodeJpeg:0"
    tensor_transfer_layer = "pool_3:0"

    def __init__(self, path):
        self.graph = tf.Graph()
        with self.graph.as_default():
            if not file_manager.check_if_file_exists(path):
                raise ValueError('Inception v3 model %s not extsts!' % path)
            with tf.gfile.FastGFile(path, 'rb') as file:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(file.read())
                tf.import_graph_def(graph_def, name='')
        self.session = tf.Session(graph=self.graph)

    def calculate(self, input_layer_name, output_layer_name, input_data):
        output_values = self.session.run(output_layer_name, feed_dict={input_layer_name: input_data})
        output_values = np.squeeze(output_values)
        return output_values


def _calculate_transfer_values_for_image(model, image):
    return model.calculate(model.tensor_input_image_layer, model.tensor_transfer_layer, image)


def calculate_transfer_values_for_images(model, images):
    images_count = len(images)
    output_values = []
    for i in range(images_count):
        print('Calculating transfer values for %d/%d image' % (i, images_count))
        output_values.append(_calculate_transfer_values_for_image(model, images[i]))
    return np.array(output_values)


def calculate_or_load_transfer_values_for_images(model, images, path_to_csv):
    if file_manager.check_if_file_exists(path_to_csv):
        data = np.genfromtxt(path_to_csv, delimiter=',')
    else:
        data = calculate_transfer_values_for_images(model, images)
        np.savetxt(path_to_csv, data, delimiter=',')
    return data
