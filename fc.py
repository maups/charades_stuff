import sys
import random
import tensorflow as tf
import numpy as np

#############
### Utils ###
#############

# Network, input and training parameters
classes = 157			# number of action classes
batch_size = 256		# size of training/validation batch
feature_size = 40960	# size of the feature vector
n_hidden = 256			# size of the hidden layer
n_epochs = 1000			# duration of training
epoch_size = 100		# number of batches per epoch

# Load helper files
def get_file_list(preffix):
	files = []
	for i in range(classes):
		files.append([])
	for i in range(classes):
		filename = preffix
		filename += str(i)
		filename += ".txt"
		fp = open(filename, 'r')
		for line in fp:
			files[i].append(line[:-1])
		fp.close()
	return files

train_files = get_file_list("helper_files/train_")
val_files = get_file_list("helper_files/val_")

# Function to create a random batch of action samples
# Classes have the same probability
# Samples from a same class have the same probability
def get_new_batch(files):
	features = np.array([])
	labels = np.array([])
	for i in range(batch_size):
		c = random.randint(0, classes-1)
		img = random.randint(0, len(files[c])-1)
		names = files[c][img].split(";")
		fc7 = np.loadtxt(names[0])
		for j in range(1, 10):
			fc_cont = np.loadtxt(names[j])
			fc7 = np.concatenate((fc7, fc_cont))
		fc7 = fc7.reshape(1, fc7.shape[0])
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
xs = tf.placeholder(tf.float32, [None, feature_size])
ys = tf.placeholder(tf.int64, [None, 1])
ys_one_hot = tf.one_hot(ys, classes)
keep_prob = tf.placeholder(tf.float32)

# 1st fully connected layer
W_fc1 = tf.Variable(tf.truncated_normal([feature_size, n_hidden], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[n_hidden]))
fc1 = tf.matmul(xs, W_fc1) + b_fc1
fc1_drop = tf.nn.dropout(fc1, keep_prob)

# 2nd fully connected layer
W_fc2 = tf.Variable(tf.truncated_normal([n_hidden, classes], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[classes]))
fc2 = tf.matmul(fc1_drop, W_fc2) + b_fc2

# Loss function
softmax_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ys_one_hot, logits=fc2)
loss = tf.reduce_mean(softmax_loss)

# Optimization
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Evaluation
_, top5 = tf.nn.top_k(fc2, 5)
result = tf.argmax(fc2, 1)
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

	### In case of interruption, load parameters from the last iteration (ex: 29)
	#saver.restore(sess, './model_fc_29')
	### And update the loop to account for the previous iterations
	#for i in range(29,n_epochs):
	for i in range(n_epochs):
		print i

		# Run 1 epoch
		vloss = []
		acc = []
		for j in range(epoch_size):
			x_train, y_train = get_new_batch(train_files)
			ret = sess.run([train_op, loss, accuracy], feed_dict = {xs: x_train, ys: y_train, keep_prob: 0.5})
			vloss.append(ret[1])
			acc.append(ret[2])

		print 'TRAIN '+str(i+1)+':', np.mean(vloss), np.mean(acc)

		# Log training loss and accuracy for current epoch
		fp = open('log_fc.txt', 'a')
		fp.write('TRAIN ' + str(i+1) + ' ' + str(np.mean(vloss)) + ' ' + str(np.mean(acc)) + '\n')
		fp.close()

		# Save network parameters
		path = 'model_fc_' + str(i+1)
		save_path = saver.save(sess, path)

		# Run validation
		if (i+1)%5 == 0:
			cont1 = 0
			cont5 = 0
			for j in range(epoch_size):
				x_train, y_train = get_new_batch(val_files)
				ret = sess.run(top5, feed_dict = {xs: x_train, ys: y_train, keep_prob: 1.0})
				for k in range(batch_size):
					c = y_train[k][0]
					if c == ret[k][0]:
						cont1 += 1
						cont5 += 1
					elif c == ret[k][1] or c == ret[k][2] or c == ret[k][3] or c == ret[k][4]:
						cont5 += 1

			print 'VAL '+str(i+1)+':', (100.*cont1)/(epoch_size*batch_size), (100.*cont5)/(epoch_size*batch_size)

			# Log Rank-1 and Rank-5
			fp = open('log_fc.txt', 'a')
			fp.write('VAL ' + str(i+1) + ' ' + str((100.*cont1)/(epoch_size*batch_size)) + ' ' + str((100.*cont5)/(epoch_size*batch_size)) + '\n')
			fp.close()

