import sys
import random
import tensorflow as tf
import numpy as np

#############
### Utils ###
#############

# Network, input and training parameters
classes = 33			# number of action classes
batch_size = 256		# size of training/validation batch
feature_size = 8192		# size of the feature vector
n_steps = 10			# size of the temporal window
n_hidden1 = 1024		# size of the 1st hidden layer
n_hidden2 = 512			# size of the 2nd hidden layer
n_epochs = 3000			# duration of training
epoch_size = 1			# number of batches per epoch

# Load helper files
def get_file_list(preffix):
	files = []
	files_rgb = []
	for i in range(classes):
		files.append([])
		files_rgb.append([])
	for i in range(classes):
		filename = preffix
		filename += str(i)
		filename += ".txt"
		fp = open(filename, 'r')
		for line in fp:
			files[i].append(line[:-1])
			files_rgb[i].append(line[:-1].replace("flow", "rgb"))
		fp.close()
	return files, files_rgb

train_files, train_files_rgb = get_file_list("helper_files/train_")
val_files, val_files_rgb = get_file_list("helper_files/val_")

# Function to create a random batch of action samples
# Classes have the same probability
# Samples from a same class have the same probability
def get_new_batch(files, files_rgb):
	features = np.array([])
	labels = np.array([])
	for i in range(batch_size):
		c = random.randint(0, classes-1)
		img = random.randint(0, len(files[c])-1)
		names = files[c][img].split(";")
		names_rgb = files_rgb[c][img].split(";")
		fc7 = np.loadtxt(names[0])
		fc7_rgb = np.loadtxt(names_rgb[0])
		fc7 = np.concatenate((fc7, fc7_rgb))
		fc7 = fc7.reshape(1, fc7.shape[0])
		for j in range(1, 10):
			fc_cont = np.loadtxt(names[j])
			fc_cont_rgb = np.loadtxt(names_rgb[j])
			fc_cont = np.concatenate((fc_cont, fc_cont_rgb))
			fc_cont = fc_cont.reshape(1, fc_cont.shape[0])
			fc7 = np.concatenate((fc7, fc_cont), axis=0)
		fc7 = fc7.reshape((1, fc7.shape[0], fc7.shape[1]))
		if i == 0:
			features = fc7
			labels = np.array([[c]])
		else:
			features = np.concatenate((features, fc7))
			labels = np.concatenate((labels, np.array([[c]])))
	return features, labels

############################
### Network architecture ###
############################

# Input
xs = tf.placeholder(tf.float32, [None, n_steps, feature_size])
ys = tf.placeholder(tf.int64, [None, 1])
ys_one_hot = tf.one_hot(ys, classes)
input_dropout = tf.placeholder(tf.float32)
inner_dropout = tf.placeholder(tf.float32)
xs_drop = tf.nn.dropout(xs, input_dropout)

# 1st LSTM layer
xs_ = tf.unstack(xs_drop, n_steps, 1)
lstm_cell1 = tf.contrib.rnn.BasicLSTMCell(n_hidden1, forget_bias=1.0)
lstm1, _ = tf.contrib.rnn.static_rnn(lstm_cell1, xs_, dtype=tf.float32, scope="lstm1")
lstm1_ = tf.stack(lstm1, 1)
lstm1_drop = tf.nn.dropout(lstm1_, inner_dropout)
lstm1_drop_ = tf.unstack(lstm1_drop, n_steps, 1)

# 2nd LSTM layer
lstm_cell2 = tf.contrib.rnn.BasicLSTMCell(n_hidden2, forget_bias=1.0)
lstm2, _ = tf.contrib.rnn.static_rnn(lstm_cell2, lstm1_drop_, dtype=tf.float32, scope="lstm2")
lstm2_drop = tf.nn.dropout(lstm2[-1], inner_dropout)

# 1st fully connected layer
W_fc1 = tf.Variable(tf.truncated_normal([n_hidden2, classes], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[classes]))
fc1 = tf.matmul(lstm2_drop, W_fc1) + b_fc1

# Loss function
softmax_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ys_one_hot, logits=fc1)
loss = tf.reduce_mean(softmax_loss)

# Optimization
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Evaluation
_, top5 = tf.nn.top_k(fc1, 5)
result = tf.argmax(fc1, 1)
ground_truth = tf.reshape(ys, [-1])
correct_prediction = tf.equal(result, ground_truth)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#####################
### Training loop ###
#####################

init = tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=0)

with tf.Session() as sess:
	# Initialize parameters
	sess.run(init)

	### In case of interruption, load parameters from the last iteration (ex: 100)
	#saver.restore(sess, './model_stacked_lstm_rgb_100')
	### And update the loop to account for the previous iterations
	#for i in range(100,n_epochs):
	for i in range(n_epochs):
		# Run 1 epoch
		vloss = []
		acc = []
		for j in range(epoch_size):
			x_train, y_train = get_new_batch(train_files, train_files_rgb)
			ret = sess.run([train_op, loss, accuracy], feed_dict = {xs: x_train, ys: y_train, inner_dropout: 0.5, input_dropout: 0.2})
			vloss.append(ret[1])
			acc.append(ret[2])

		print 'TRAIN '+str(i+1)+':', np.mean(vloss), np.mean(acc)

		# Log training loss and accuracy for current epoch
		fp = open('log_stacked_lstm_rgb.txt', 'a')
		fp.write('TRAIN ' + str(i+1) + ' ' + str(np.mean(vloss)) + ' ' + str(np.mean(acc)) + '\n')
		fp.close()

		# Save network parameters
		if (i+1)%100 == 0:
			path = 'model_stacked_lstm_rgb_' + str(i+1)
			save_path = saver.save(sess, path)

		# Run validation
		if (i+1)%1 == 0:
			cont1 = 0
			cont5 = 0
			vloss = []
			for j in range(epoch_size):
				x_train, y_train = get_new_batch(val_files, val_files_rgb)
				ret_all = sess.run([top5, loss], feed_dict = {xs: x_train, ys: y_train, inner_dropout: 1.0, input_dropout: 1.0})
				ret = ret_all[0]
				vloss.append(ret[1])
				for k in range(batch_size):
					c = y_train[k][0]
					if c == ret[k][0]:
						cont1 += 1
						cont5 += 1
					elif c == ret[k][1] or c == ret[k][2] or c == ret[k][3] or c == ret[k][4]:
						cont5 += 1

			print 'VAL '+str(i+1)+':', np.mean(vloss), (100.*cont1)/(epoch_size*batch_size), (100.*cont5)/(epoch_size*batch_size)

			# Log Rank-1 and Rank-5
			fp = open('log_stacked_lstm_rgb.txt', 'a')
			fp.write('VAL ' + str(i+1) + ' ' + str(np.mean(vloss)) + ' ' + str((100.*cont1)/(epoch_size*batch_size)) + ' ' + str((100.*cont5)/(epoch_size*batch_size)) + '\n')
			fp.close()

