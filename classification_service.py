import numpy as np
import tensorflow as tf
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

__author__ = "Marcin Stachowiak"
__version__ = "1.0"
__email__ = "marcin.stachowiak.ms@gmail.com"


class EnsembleVotingModel:
    """
    Class represents model for ensemble voting algorithm.
    """
    _models = []

    def __init__(self, X, Y):
        self._X = X
        self._Y = Y

    def with_SVM_model(self, n_estimators=2):
        estimatior = SVC(kernel='rbf')
        self._models.append(('svm', estimatior))
        return self

    def with_RandomForest_model(self, n_estimators=2):
        estimatior = RandomForestClassifier(random_state=1)
        self._models.append(('rf', estimatior))
        return self

    def with_LogisticRegression(self, n_estimators=2):
        estimatior = LogisticRegression(random_state=1)
        self._models.append(('lr', estimatior))
        return self

    def with_SVM_AdaBoost_model(self, n_estimators=2):
        estimatior = AdaBoostClassifier(SVC(kernel='rbf', probability=True), n_estimators=n_estimators)
        self._models.append(('svm_ada', estimatior))
        return self

    def with_RandomForest_AdaBoost_model(self, n_estimators=2):
        estimatior = AdaBoostClassifier(RandomForestClassifier(random_state=1), n_estimators=n_estimators)
        self._models.append(('rf_ada', estimatior))
        return self

    def with_LogisticRegression_AdaBoost_model(self, n_estimators=2):
        estimatior = AdaBoostClassifier(LogisticRegression(random_state=1), n_estimators=n_estimators)
        self._models.append(('lr_ada', estimatior))
        return self

    def train(self):
        print('Building ensemble model')
        self.voting_model = VotingClassifier(estimators=self._models, voting='hard')
        print('Training ensemble model')
        self.voting_model.fit(self._X, self._Y)
        return self

    def predict(self, X):
        return self.voting_model.predict(X)

    def score(self, X, Y):
        return self.voting_model.score(X, Y)


class NeuralNetworkModel:
    """
       Class represents model for multilayer perceptron.
       """

    def __init__(self, X, Y, learning_rate=1e-4, num_steps=10000, batch_size=64, display_step=100, n_hidden_1=1024):
        self._X = X
        self._Y = Y
        self._learning_rate = learning_rate
        self._num_steps = num_steps
        self._batch_size = batch_size
        self._display_step = display_step
        self._n_hidden_1 = n_hidden_1
        self._num_input = X.shape[1]
        self._num_classes = Y.shape[1]

    def predict(self, X):
        return self._session.run(self._prediction, feed_dict={self._X_p: X})

    def score(self, X, Y):
        return self._session.run(self._accuracy, feed_dict={self._X_p: X, self._Y_p: Y})

    def _random_batch_data(self, data_x, data_y):
        num_images = len(data_x)
        idx = np.random.choice(num_images, size=self._batch_size, replace=False)
        x_batch = data_x[idx]
        y_batch = data_y[idx]

        return x_batch, y_batch

    def _build_graph(self):
        self._X_p = tf.placeholder("float", [None, self._num_input])
        self._Y_p = tf.placeholder("float", [None, self._num_classes])

        # Construct model
        logits = self._build_net_model(self._X_p)

        # Define loss and optimizer
        self._loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=self._Y_p))
        optimizer = tf.train.AdamOptimizer(learning_rate=self._learning_rate)
        self._train_op = optimizer.minimize(self._loss_op)

        # Evaluate model (with test logits, for dropout to be disabled)
        self._prediction = tf.argmax(logits, 1)
        correct_pred = tf.equal(self._prediction, tf.argmax(self._Y_p, 1))
        self._accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

    def _build_net_model(self, x):
        weights = {
            'h1': tf.Variable(tf.random_normal([self._num_input, self._n_hidden_1])),
            'out': tf.Variable(tf.random_normal([self._n_hidden_1, self._num_classes]))
        }
        biases = {
            'b1': tf.Variable(tf.random_normal([self._n_hidden_1])),
            'out': tf.Variable(tf.random_normal([self._num_classes]))
        }

        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
        return out_layer

    def train(self):
        self._build_graph()
        # Initialize the variables (i.e. assign their default value)
        init = tf.global_variables_initializer()

        self._session = tf.Session()
        self._session.run(init)

        for step in range(1, self._num_steps + 1):
            batch_x, batch_y = self._random_batch_data(self._X, self._Y)
            # Run optimization op (backprop)
            self._session.run(self._train_op, feed_dict={self._X_p: batch_x, self._Y_p: batch_y})
            if step % self._display_step == 0 or step == 1:
                # Calculate batch loss and accuracy
                loss, acc = self._session.run([self._loss_op, self._accuracy],
                                              feed_dict={self._X_p: batch_x, self._Y_p: batch_y})
                print("Step " + str(step) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss) + ", Training Accuracy= " + \
                      "{:.3f}".format(acc))
        print("Optimization Finished!")

        return self
