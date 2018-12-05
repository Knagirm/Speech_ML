from __future__ import print_function

import tensorflow as tf
import math
import random
import numpy as np

import librosa

from model import myNeuralNet

import matplotlib.pyplot as plt

# Defining properties
dim_input = 900
dim_output = 1

# strings for file reading
#the data was exported as .npy since it was time-expensive to convert all the audio files everytime
train_signal_fname = 'speech/train_signal.npy'
train_lbls_fname = 'speech/train_lbls.npy'
valid_signal_fname = 'speech/valid_signal.npy'
valid_lbls_fname = 'speech/valid_lbls.npy'
test_fname = 'speech/test_signal.npy'


# set variables
hidden_size = 128
max_epochs = 16
learn_rate = 1e-4
batch_size = 32

# this just makes sure that all our following operations will be placed in the right graph.
tf.reset_default_graph()

# creates a session variable
session = tf.Session()

audioIn = tf.placeholder(tf.float32, [None, 45, 20])

labels = tf.placeholder(tf.float32, [None])
keep_prob = tf.placeholder(tf.float32)


# make the lstm cells, and wrap them in MultiRNNCell for multiple layers
lstm_cell_1 = tf.contrib.rnn.LSTMCell(hidden_size)
lstm_cell_1 = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_1, output_keep_prob=keep_prob)
lstm_cell_2 = tf.contrib.rnn.LSTMCell(hidden_size)
lstm_cell_2 = tf.contrib.rnn.DropoutWrapper(cell=lstm_cell_2, output_keep_prob=keep_prob)
multi_lstm_cells = tf.contrib.rnn.MultiRNNCell(cells=[lstm_cell_1, lstm_cell_2] , state_is_tuple=True)

# define the op that runs the LSTM, across time, on the data
outputtrial, final_state = tf.nn.dynamic_rnn(multi_lstm_cells, audioIn, dtype=tf.float32)

# a useful function that takes an input and what size we want the output 
# to be, and multiples the input by a weight matrix plus bias (also creating
# these variables)
def linear(input_, output_size, name, init_bias=0.0):
	shape = input_.get_shape().as_list()
	with tf.variable_scope(name):
		W = tf.get_variable("weight_matrix", [shape[-1], output_size], tf.float32, tf.random_normal_initializer(stddev=1.0 / math.sqrt(shape[-1])))
	if init_bias is None:
		return tf.matmul(input_, W)
	with tf.variable_scope(name):
		b = tf.get_variable("bias", [output_size], initializer=tf.constant_initializer(init_bias))
	return tf.matmul(input_, W) + b

# define that our final word logit is a linear function of the final state 
# of the LSTM
# final_state[-1][-1] represents the output after time T of the last LSTM layer
word = linear(final_state[-1][-1], 1, name="output")

word = tf.squeeze(word, [1])

# define cross entropy loss function
loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=word, labels=labels)
loss = tf.reduce_mean(loss)

# round our actual probabilities to compute error
prob = tf.nn.sigmoid(word)
prediction = tf.to_float(tf.greater_equal(prob, 0.5))
pred_err = tf.to_float(tf.not_equal(prediction, labels))
pred_err = tf.reduce_sum(pred_err)

# define our optimizer to minimize the loss
optimizer = tf.train.AdamOptimizer(learn_rate).minimize(loss)

# initialize any variables
tf.global_variables_initializer().run(session=session)



# load our data and separate it into audioIn and labels
train_signal = np.load(train_signal_fname)
train_lbls = np.load(train_lbls_fname).astype(int).reshape((-1,))
valid_signal = np.load(valid_signal_fname)
valid_lbls = np.load(valid_lbls_fname).astype(int).reshape((-1,))
test_signal = np.load(test_fname)

#reshaping all the inputs
train_signal = train_signal.reshape(-1,45,20)
valid_signal = valid_signal.reshape(-1,45,20)
test_signal = test_signal.reshape(-1,45,20)


# we'll train with batches of size 128.  This means that we run 
# our model on 128 examples and then do gradient descent based on the loss
# over those 128 examples.
num_steps = 100

error = []
val_error_arr = []
index = []
index1=[]

for epoch in range(max_epochs):
	for step in range(num_steps):
		# get data for a batch
		offset = (step * batch_size) % (len(train_lbls) - batch_size)

		batch_audioIn = train_signal[offset : (offset + batch_size)]
		batch_labels = train_lbls[offset : (offset + batch_size)]

		# put this data into a dictionary that we feed in when we run 
		# the graph.  this data fills in the placeholders we made in the graph.
		data = {audioIn: batch_audioIn, labels: batch_labels , keep_prob: 0.75}
		

		# run the 'optimizer', 'loss', and 'pred_err' operations in the graph
		_, loss_value_train, error_value_train = session.run(
		  [optimizer, loss, pred_err], feed_dict=data)
		
		if (step % 10 == 0):
			error.append(loss_value_train)
			index.append(epoch*num_steps+step)

		# print stuff every 50 steps to see how we are doing
		if (step % 50 == 0):
			print("Epoch no: ", epoch)
			print("Minibatch train loss at step", step, ":", loss_value_train)
			print("Minibatch train error: %.3f%%" % error_value_train)
			
			# get test evaluation
			val_loss = []
			val_error = []
			for batch_num in range(int(len(valid_lbls)/batch_size)):
				val_offset = (batch_num * batch_size) % (len(valid_lbls) - batch_size)

				val_batch_audioIn = valid_signal[val_offset : (val_offset + batch_size)]
				val_batch_labels = valid_lbls[val_offset : (val_offset + batch_size)]

				data_valid = {audioIn: val_batch_audioIn, labels: val_batch_labels, keep_prob: 1}
				loss_value_val, error_value_val = session.run([loss, pred_err], feed_dict=data_valid)
				val_loss.append(loss_value_val)
				val_error.append(error_value_val)
			
			index1.append(epoch*100+step)
			val_error_arr.append(np.mean(val_loss))
			print("Test loss: %.3f" % np.mean(val_loss))
			print("Test error: %.3f%%" % np.mean(val_error))


# plt.plot(index, error)
# plt.show()

# plt.plot(index1, val_error_arr)
# plt.show()

data_valid = {audioIn: test_signal, keep_prob: 1}
prediction_arr = session.run([prediction], feed_dict=data_valid)

# np.save('prediction-speech.npy',prediction_arr)