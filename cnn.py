import sys
import random
import tensorflow as tf
import numpy as np
from scipy.ndimage import imread

#############
### Utils ###
#############

# Network, input and training parameters
classes = 16			# number of action classes
batch_size = 256		# size of training/validation batch
image_size = 128		# size of the image (width = height)
image_channels = 3		# number of image channels
n_epochs = 3000			# duration of training
epoch_size = 1			# number of batches per epoch

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
		while True:
			c = random.randint(0, classes-1)
			img = random.randint(0, len(files[c])-1)
			name = files[c][img]
			buf = imread(name, mode='RGB')
			if buf.shape == (image_size, image_size, image_channels):
				break
		buf = buf.astype(np.float32)/255.0
		buf = buf.reshape(1, image_size, image_size, image_channels)
		if i == 0:
			features = buf
			labels = np.array([[c]])
		else:
			features = np.concatenate((features, buf))
			labels = np.concatenate((labels, np.array([[c]])))
	return features, labels

############################
### Network architecture ###
############################

# Input
xs = tf.placeholder(tf.float32, [None, image_size, image_size, image_channels])
ys = tf.placeholder(tf.int64, [None, 1])
ys_one_hot = tf.one_hot(ys, classes)
input_dropout = tf.placeholder(tf.float32)
inner_dropout = tf.placeholder(tf.float32)
xs_drop = tf.nn.dropout(xs, input_dropout)

# 1st convolutional layer
W_conv1 = tf.Variable(tf.truncated_normal([3, 3, image_channels, 128], stddev=0.1))
b_conv1 = tf.Variable(tf.constant(0.1, shape=[128]))
h_conv1 = tf.nn.relu(tf.nn.conv2d(xs_drop, W_conv1, strides=[1, 1, 1, 1], padding='SAME') + b_conv1)

# 2nd convolutional layer
W_conv2 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
b_conv2 = tf.Variable(tf.constant(0.1, shape=[128]))
h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, W_conv2, strides=[1, 1, 1, 1], padding='SAME') + b_conv2)

# 1st pooling layer
h_pool1 = tf.nn.max_pool(h_conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 3rd convolutional layer
W_conv3 = tf.Variable(tf.truncated_normal([3, 3, 128, 128], stddev=0.1))
b_conv3 = tf.Variable(tf.constant(0.1, shape=[128]))
h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool1, W_conv3, strides=[1, 1, 1, 1], padding='SAME') + b_conv3)

# 2nd pooling layer
h_pool2 = tf.nn.max_pool(h_conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 4th convolutional layer
W_conv4 = tf.Variable(tf.truncated_normal([3, 3, 128, 256], stddev=0.1))
b_conv4 = tf.Variable(tf.constant(0.1, shape=[256]))
h_conv4 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv4, strides=[1, 1, 1, 1], padding='SAME') + b_conv4)

# 3rd pooling layer
h_pool3 = tf.nn.max_pool(h_conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 5th convolutional layer
W_conv5 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1))
b_conv5 = tf.Variable(tf.constant(0.1, shape=[256]))
h_conv5 = tf.nn.relu(tf.nn.conv2d(h_pool3, W_conv5, strides=[1, 1, 1, 1], padding='SAME') + b_conv5)

# 4th pooling layer
h_pool4 = tf.nn.max_pool(h_conv5, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 6th convolutional layer
W_conv6 = tf.Variable(tf.truncated_normal([3, 3, 256, 256], stddev=0.1))
b_conv6 = tf.Variable(tf.constant(0.1, shape=[256]))
h_conv6 = tf.nn.relu(tf.nn.conv2d(h_pool4, W_conv6, strides=[1, 1, 1, 1], padding='SAME') + b_conv6)

# 5th pooling layer
h_pool5 = tf.nn.max_pool(h_conv6, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

# 1st fully connected layer
flattened_h_conv6 = tf.reshape(h_conv6, [-1, np.prod(h_conv6.shape[1:]).value])
flattened_h_pool5 = tf.reshape(h_pool5, [-1, np.prod(h_pool5.shape[1:]).value])
input_fc1 = tf.concat([flattened_h_conv6, flattened_h_pool5], 1)
W_fc1 = tf.Variable(tf.truncated_normal([input_fc1.shape[1].value, 512], stddev=0.1))
b_fc1 = tf.Variable(tf.constant(0.1, shape=[512]))
h_fc1 = tf.matmul(input_fc1, W_fc1) + b_fc1
drop_h_fc1 = tf.nn.dropout(h_fc1, inner_dropout)

# 2nd fully connected layer
W_fc2 = tf.Variable(tf.truncated_normal([512, classes], stddev=0.1))
b_fc2 = tf.Variable(tf.constant(0.1, shape=[classes]))
h_fc2 = tf.matmul(drop_h_fc1, W_fc2) + b_fc2

# Loss function
softmax_loss = tf.nn.softmax_cross_entropy_with_logits(labels=ys_one_hot, logits=h_fc2)
loss = tf.reduce_mean(softmax_loss)

# Optimization
train_op = tf.train.AdamOptimizer(1e-4).minimize(loss)

# Evaluation
_, top5 = tf.nn.top_k(h_fc2, 5)
result = tf.argmax(h_fc2, 1)
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

	### In case of interruption, load parameters from the last iteration (ex: 1000)
	#saver.restore(sess, './model_cnn_1000')
	### And update the loop to account for the previous iterations
	#for i in range(1000,n_epochs):
	for i in range(n_epochs):
		# Run 1 epoch
		vloss = []
		acc = []
		for j in range(epoch_size):
			x_train, y_train = get_new_batch(train_files)
			ret = sess.run([train_op, loss, accuracy], feed_dict = {xs: x_train, ys: y_train, inner_dropout: 0.5, input_dropout: 0.8})
			vloss.append(ret[1])
			acc.append(ret[2])

		print 'TRAIN '+str(i+1)+':', np.mean(vloss), np.mean(acc)

		# Log training loss and accuracy for current epoch
		fp = open('log_cnn.txt', 'a')
		fp.write('TRAIN ' + str(i+1) + ' ' + str(np.mean(vloss)) + ' ' + str(np.mean(acc)) + '\n')
		fp.close()

		# Save network parameters
		if (i+1)%100 == 0:
			path = 'model_cnn_' + str(i+1)
			save_path = saver.save(sess, path)

		# Run validation
		if (i+1)%1 == 0:
			cont1 = 0
			cont5 = 0
			vloss = []
			for j in range(epoch_size):
				x_train, y_train = get_new_batch(val_files)
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
			fp = open('log_cnn.txt', 'a')
			fp.write('VAL ' + str(i+1) + ' ' + str(np.mean(vloss)) + ' ' + str((100.*cont1)/(epoch_size*batch_size)) + ' ' + str((100.*cont5)/(epoch_size*batch_size)) + '\n')
			fp.close()

