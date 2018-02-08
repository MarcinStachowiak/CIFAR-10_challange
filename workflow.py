import cifar10_manager
import dim_reduction_service
import image_service
import inceptionv3_manager
import metrics_service
from classification_service import EnsembleVotingModel
from classification_service import NeuralNetworkModel
from feature_service import FearureImageExtractor

__author__ = "Marcin Stachowiak"
__version__ = "1.0"
__email__ = "marcin.stachowiak.ms@gmail.com"

# if True features will be calculated based on Inception v3 model and transfer learning
# if False features will be calbulated based on Haralick, Zernike and Linear Binary Patterns methods.
transfer_values_features = True

# if True the classification model will be based on SVM, LogicalRegression with boosting and voting
# if False the classification model will be based on Multilayer Perceptron (Neural Network)
use_ensemble=True

# Downloading or loading CIFAR 10 dataset.
data = cifar10_manager.download_or_load_CIFAR10('data')

# Plotting random images for each class.
dict = cifar10_manager.build_dictionary_with_images_per_class(data.train_x, data.train_y_cls, 10)
image_service.plot_images_per_class(dict, cifar10_manager.get_class_names())

# Feature extraction
feature_extractor = FearureImageExtractor(data.train_x, data.test_x)
if transfer_values_features:
    # Using transfer leatning and Inception v3 model
    model = inceptionv3_manager.download_or_load_Inceptionv3('inception')
    (features_train_x, features_test_x) = feature_extractor.perform_transfer_values_extraction(model)
else:
    # Using texture features: Haralick, Zernike and Linear Binary Patterns
    (features_train_x, features_test_x) = feature_extractor.perform_texture_features_extraction()

# Plotting features on a two-dimensional chart after applying the PCA and TSNE reduction methods
dim_reduction_service.reduce_dim_PCA(features_train_x, data.train_y_cls, cifar10_manager.get_class_names(),
                                     visualise=True)
dim_reduction_service.reduce_dim_TSNE(features_train_x, data.train_y_cls, cifar10_manager.get_class_names(),
                                      visualise=True)
# Classification
if use_ensemble:
    # Using ensemble: Boosting and Voting
    voting_model = EnsembleVotingModel(features_train_x, data.train_y_cls) \
        .with_SVM_model() \
        .with_RandomForest_AdaBoost_model(5)\
        .with_LogisticRegression_AdaBoost_model(5)\
        .train()
    predicted_cls = voting_model.predict(features_test_x)
else:
    # Using Multilayer Perception (Neural Network)
    model=NeuralNetworkModel(features_train_x,data.train_y).train()
    predicted_cls = model.predict(features_test_x)

# Metrics calculation
metrics_service.print_full_metrics(data.test_y_cls, predicted_cls)






