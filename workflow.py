import cifar10_manager
import image_service
import inceptionv3_manager
from sklearn.manifold import TSNE
import tensorflow as tf
import numpy as np
import prettytensor as pt

data=cifar10_manager.download_or_load_CIFAR10('data')

dict=cifar10_manager.build_dictionary_with_images_per_class(data.train_x,data.train_y,10)
image_service.plot_images_per_class(dict,cifar10_manager.get_class_names())

model=inceptionv3_manager.download_or_load_Inceptionv3('inception')
transfer_values_train_x=inceptionv3_manager.calculate_or_load_transfer_values_for_images(model,data.train_x,'train_x_dump.csv')
transfer_values_test_x=inceptionv3_manager.calculate_or_load_transfer_values_for_images(model,data.test_x,'test_x_dump.csv')

#TODO
#tsne = TSNE(n_components=3)
#transfer_values_train_x_reduced = tsne.fit_transform(transfer_values_train_x)

# Parameters
learning_rate = 1e-4
num_steps = 10000
batch_size = 64
display_step = 100

# Network Parameters
n_hidden_1 = 1024 # 1st layer number of neurons
n_hidden_2 = 1024 # 2nd layer number of neurons
num_input = transfer_values_train_x.shape[1]
num_classes = cifar10_manager._num_classes # MNIST total classes (0-9 digits)

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_classes])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([num_classes]))
}

# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Hidden fully connected layer with 256 neurons
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

# Construct model
# = neural_net(X)

# Define loss and optimizer
#oss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
 #   logits=logits, labels=Y))


x_pretty = pt.wrap(X)

with pt.defaults_scope(activation_fn=tf.nn.relu):
    logits, loss_op = x_pretty.\
        fully_connected(size=1024, name='layer_fc1').\
        softmax_classifier(num_classes=num_classes, labels=Y)

optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Evaluate model (with test logits, for dropout to be disabled)
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

def random_batch_train_data():
    num_images = len(transfer_values_train_x)

    idx = np.random.choice(num_images,
                           size=batch_size,
                           replace=False)

    x_batch = transfer_values_train_x[idx]
    y_batch = tf.one_hot(data.train_y[idx],10,1,0)

    return x_batch, y_batch

def batch_test_data():
    x_batch = transfer_values_test_x
    y_batch = tf.one_hot(data.test_y,10,1,0)
    return x_batch, y_batch

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    # Calculate accuracy for MNIST test images
    batch_x, batch_y = batch_test_data()
    y_values = sess.run(batch_y)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: batch_x,
                                      Y: y_values}))

    for step in range(1, num_steps+1):
        batch_x, batch_y = random_batch_train_data()

        y_values=sess.run(batch_y)
        # Run optimization op (backprop)
        sess.run(train_op, feed_dict={X: batch_x, Y: y_values})
        if step % display_step == 0 or step == 1:
            # Calculate batch loss and accuracy
            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: y_values})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Optimization Finished!")

    # Calculate accuracy for MNIST test images
    batch_x, batch_y = batch_test_data()
    y_values = sess.run(batch_y)
    print("Testing Accuracy:", \
        sess.run(accuracy, feed_dict={X: batch_x,
                                      Y: y_values}))