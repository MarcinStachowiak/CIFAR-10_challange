import cifar10_manager
import image_service


data=cifar10_manager.download_or_load_CIFAR10('data')
dict=cifar10_manager.build_dictionary_with_images_per_class(data.train_x,data.train_y,10)
image_service.plot_images_per_class(dict,cifar10_manager.get_class_names())