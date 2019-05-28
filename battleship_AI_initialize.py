
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import tensorflow as tf
if type(tf.contrib) != type(tf): tf.contrib._warning = None
tf.logging.set_verbosity(tf.logging.FATAL)
import pandas as pd
import numpy as np

import battleship_game
###############################################################

# Generate training data
'''player = battleship_game.Player()
features = []
labels = []

for i in range(2000):
    player.board = -np.ones((10,10))
    player.place_ships(random=True)
    board = player.board

    for j in range(100):
        mask = np.random.choice(2, size=(10,10), p=[0.5 + j/200, 0.5 - j/200])
        features.append(np.multiply(mask, board).flatten().astype(int))
        labels.append(board.flatten().astype(int))

featuresDF = pd.DataFrame(features)
labelsDF = pd.DataFrame(labels)

# shuffle training data
combinedDF = pd.concat([featuresDF, labelsDF], axis=1).sample(frac=1)
data_features = combinedDF.values[:, :100]
data_labels = combinedDF.values[:, 100:]
pd.DataFrame(featuresDF).to_csv('/home/brian/PycharmProjects/Battleship/battleship_features.csv')
pd.DataFrame(labelsDF).to_csv('/home/brian/PycharmProjects/Battleship/battleship_labels.csv')'''

# board == np.reshape(board.flatten(), (10, 10))

# Grab data

data_features = pd.read_csv('battleship_features.csv', index_col=0).values
data_labels = pd.read_csv('battleship_labels.csv', index_col=0).values

# Set hyperparameters
trainingRatio = 0.9  # Fraction of total data size
batchSize = 50
num_epochs = 100

learningRate = 1e-4
balanceWeight = 10

# Training/validation split
dataLen, labelsDim = data_labels.shape
batches_per_epoch = dataLen // batchSize
_, featuresDim = data_features.shape
training_len = int(dataLen * trainingRatio)
trainingLabels = data_labels[:training_len]
validationLabels = data_labels[training_len:]
trainingFeatures = data_features[:training_len]
validationFeatures = data_features[training_len:]

###############################################################

# Network structure
features = tf.placeholder(tf.float64, (None, featuresDim))
labels = tf.placeholder(tf.float64, (None, labelsDim))

hidden_layer = tf.contrib.layers.fully_connected(features, 100)
predictedLabels = tf.contrib.layers.fully_connected(hidden_layer, labelsDim, activation_fn=None)

# Cost function

BCE = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=(1+labels)/2,
                                                              logits=predictedLabels,
                                                              pos_weight=balanceWeight))
l2RegTerm = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
cost = BCE + l2RegTerm

# Optimizer
opt = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

###############################################################
warmStart = True
train = True

# Train / predict
with tf.Session() as sess:
    # Turn on model saver
    saver = tf.train.Saver()
    # Initialize parameters
    if warmStart:
        saver.restore(sess, "model/model.ckpt")
    else:
        sess.run(tf.global_variables_initializer())
    # Initialize optimization record
    costlist = []
    # Initialize mini-batch counter
    i = 0
    if train:
        while i < num_epochs * batches_per_epoch:
            j = i % batches_per_epoch
            # run optimizer on current batch
            _, cst = sess.run([opt, cost],
                              feed_dict={features: trainingFeatures[j * batchSize:(j + 1) * batchSize, :],
                                         labels: trainingLabels[j * batchSize:(j + 1) * batchSize, :]})
            # Print progress
            if i % 1000 == 0:
                validationCost = sess.run(cost,
                                          feed_dict={features: validationFeatures, labels: validationLabels})
                print('Mini-Batch {}: Training Cost: {}, Validation Cost: {}'.format(i, cst, validationCost))
                # costlist.append([cst, validationCost])
            i += 1
        # Save trained model
        save_path = saver.save(sess, "model/model.ckpt")

