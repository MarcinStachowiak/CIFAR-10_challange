import cifar10_manager
import dim_reduction_service
import image_service
import inceptionv3_manager
import metrics_service
from classification_service import EnsembleVotingModel
from feature_service import FearureImageExtractor

__author__ = "Marcin Stachowiak"
__version__ = "1.0"
__email__ = "marcin.stachowiak.ms@gmail.com"

transfer_values_features = True

data = cifar10_manager.download_or_load_CIFAR10('data')

dict = cifar10_manager.build_dictionary_with_images_per_class(data.train_x, data.train_y_cls, 10)
image_service.plot_images_per_class(dict, cifar10_manager.get_class_names())

feature_extractor = FearureImageExtractor(data.train_x, data.test_x)
if transfer_values_features:
    model = inceptionv3_manager.download_or_load_Inceptionv3('inception')
    (features_train_x, features_test_x) = feature_extractor.perform_transfer_values_extraction(model)
else:
    (features_train_x, features_test_x) = feature_extractor.perform_texture_features_extraction()

dim_reduction_service.reduce_dim_PCA(features_train_x, data.train_y_cls, cifar10_manager.get_class_names(),
                                     visualise=True)
dim_reduction_service.reduce_dim_TSNE(features_train_x, data.train_y_cls, cifar10_manager.get_class_names(),
                                      visualise=True)

voting_model = EnsembleVotingModel(features_train_x, data.train_y_cls) \
    .with_SVM_model()\
    .train()

predicted_cls = voting_model.predict(features_test_x)

metrics_service.print_full_metrics(data.test_y_cls,predicted_cls)
