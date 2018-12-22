""" Sparse AutoEncoder of DIABIMMUNE Project: Learn latent representation of DIABIMMUNE microbial profiles."""
## v1.6

from __future__ import division, print_function, absolute_import
import datetime
import sys
import os
from optparse import OptionParser
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.contrib.tensorboard.plugins import projector
from sklearn.utils import shuffle

## Hyperparameters
NumEpochs = 300
NumHidden_1 = 60
NumHidden_2 = 25
LearningRate = 0.001
BatchSize = 5
NumInput = -1
ActFunc = "relu"
rho = 0.01
beta = 3
alpha = 0.0001

def import_data(InputFile):
	df = pd.read_csv(InputFile, sep=",", index_col=0)
	features = list(df.index)
	samples = list(df)
	data = np.transpose(df.as_matrix())
	return data, features, samples
	
	
def generate_metadata(samples, MetadataFile):
	print("Generating Metadata...")
	age = {}
	country = {}
	delivery = {}
	allergy = {}
	
	with open(MetadataFile, 'r') as read_file:

		for line in read_file:
			row = line.split(",")
			id = row[3]
			age[id] = row[4]
			country[id] = row[5]
			delivery[id] = row[6]
			allergy[id] = row[10]

	with open(LogDir + "/metadata.tsv", 'w') as metadata_file:
		metadata_file.write("Age\tCountry\tDelivery\tAllergy\n")
		for s in samples:
			metadata_file.write(age[s] + "\t" + country[s] + "\t" + delivery[s] + "\t" + allergy[s])
			# print("sample = ", s)
			# print("age = ", age[s])
			# print("country = ", country[s])
			# print("country = ", delivery[s])
			# print("allergy = ", allergy[s])
	print("Done Generating Metadata")



def layer_output(x, w, b):
	if ActFunc == "relu":
		return tf.nn.relu(tf.add(tf.matmul(x, w), b))
	if ActFunc == "sigmoid":
		return tf.nn.sigmoid(tf.add(tf.matmul(x, w), b))
	if ActFunc == "tanh":
		return tf.nn.tanh(tf.add(tf.matmul(x, w), b))



## Build encoder_decoder
def EncoderDecoder(x, name="EncoderDecoder"):
	with tf.name_scope(name):
		with tf.variable_scope("Layer_1"):
			w = tf.Variable(tf.random_normal([num_features, NumHidden_1]), name = "W")
			b = tf.Variable(tf.random_normal([NumHidden_1]), name = "B")
			tf.summary.histogram("weights", w)
			tf.summary.histogram("biases", b)
			layer_1 = layer_output(x, w, b)
		with tf.variable_scope("Layer_2"):
			w = tf.Variable(tf.random_normal([NumHidden_1, NumHidden_2]), name = "W")
			b = tf.Variable(tf.random_normal([NumHidden_2]), name="B")
			tf.summary.histogram("weights", w)
			tf.summary.histogram("biases", b)
			layer_2 = layer_output(layer_1, w, b)
			tf.summary.tensor_summary = ("Latent", layer_2)
		with tf.variable_scope("Layer_3"):
			w = tf.Variable(tf.random_normal([NumHidden_2, NumHidden_1]), name="W")
			b = tf.Variable(tf.random_normal([NumHidden_1]), name="B")
			tf.summary.histogram("weights", w)
			tf.summary.histogram("biases", b)
			layer_3 = layer_output(layer_2, w, b)
		with tf.variable_scope("Layer_4"):
			w = tf.Variable(tf.random_normal([NumHidden_1, num_features]), name="W")
			b = tf.Variable(tf.random_normal([num_features]), name="B")
			tf.summary.histogram("weights", w)
			tf.summary.histogram("biases", b)
			layer_4 = layer_output(layer_3, w, b)
		return layer_2, layer_4




# Construct model
def autoencoder():
	X = tf.placeholder("float", [None, num_features], name="X")
	with tf.name_scope("Autoencoder"):
		encoder_op, decoder_op = EncoderDecoder(X)

	with tf.name_scope("Predictions"):
		y_pred = decoder_op

	with tf.name_scope("Targets"):
		y_true = X

	with tf.name_scope("Loss"):
		loss = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
		tf.summary.scalar("Loss", loss)

	return {'x': X, 'latent': encoder_op, 'y':X, 'pred':decoder_op, 'cost':loss}


def kl_divergence(rho, rho_hat):
	kl_1 = rho * tf.log(rho) - rho * tf.log(rho_hat) + (1 - rho) * tf.log(1 - rho) - (1 - rho) * tf.log(1 - rho_hat)
	return kl_1


def train(data):
	data = shuffle(data)
	numSamples = data.shape[0]
	split = round(0.1*numSamples)
	ValidData = data[0:split,:]
	TestData = data[split:2*split,:]
	TrainData = data[2*split:numSamples,:]
	dataset = tf.data.Dataset.from_tensor_slices(TrainData)
	dataset = dataset.batch(BatchSize)
	iterator = dataset.make_initializable_iterator()
	next_element = iterator.get_next()

	## Sparse autoencoder
	ae = autoencoder()
	rho_hat = tf.reduce_mean(ae['latent'])
	kl = kl_divergence(rho, rho_hat)
	sparse = beta * tf.reduce_sum(kl)

	with tf.name_scope("Train"):
		optimizer = tf.train.AdamOptimizer(LearningRate).minimize(ae['cost'] + sparse)

	init = tf.global_variables_initializer()
	sess = tf.Session()
	sess.run(init)
	merged_summary = tf.summary.merge_all()
	writer = tf.summary.FileWriter(LogDir)
	writer.add_graph(sess.graph)
	TestWriter = tf.summary.FileWriter(LogDir + "_Test")  # +"_Test")
	TestWriter.add_graph(sess.graph)


	# Training
	prev_loss = 500000
	epoch_idx = 0
	trainingLoss = list()
	validationLoss = list()
	#testingLoss = list()
	for i in range(1, NumEpochs):
		sess.run(iterator.initializer)
		loss = 0
		num_batch = 0
		saver = tf.train.Saver(max_to_keep = 5)
		while True:
			try:
				batch_x = sess.run(next_element)
				_, l = sess.run([optimizer, ae['cost']],  feed_dict={ae['x']: batch_x})
				loss += l
				num_batch += 1

			except tf.errors.OutOfRangeError:
				s = sess.run(merged_summary, feed_dict={ae['x']: TrainData})
				writer.add_summary(s, i)
				trainLoss = loss/num_batch
				print('Epoch = %i\tTraining Loss = %f' % (i, trainLoss))
				trainingLoss.append(trainLoss)
				break

		if i % 10 == 0:
			with tf.name_scope("TestingMeasurements"):
				s, validation_cost = sess.run([merged_summary, ae['cost']], feed_dict={ae['x']: ValidData})
				print('Epoch = %i\t***  Validation Loss = %f' % (i, validation_cost))
				validationLoss.append(validation_cost)

				if(validation_cost < prev_loss): # Save best model based on validation loss
					prev_loss = validation_cost
					saver.save(sess, os.path.join(LogDir, 'model.ckpt'), global_step=i)
					epoch_idx = i
				TestWriter.add_summary(s, i)



	## Apply best model to the test set
	bestmodel = os.path.join(LogDir, 'model.ckpt-'+str(epoch_idx))
	print("Best model occured at epoch = ", epoch_idx)
	print("Best Model path = ", bestmodel)
	saver.restore(sess, bestmodel)
	s, test_cost = sess.run([merged_summary, ae['cost']], feed_dict={ae['x']: TestData})
	embedded_data = sess.run(ae['latent'], feed_dict={ae['x']: data})
	np.savetxt(LogDir + "_trainingLoss.csv", trainingLoss, delimiter=",")
	np.savetxt(LogDir + "_validationLoss.csv", validationLoss, delimiter=",")
	csv = open(LogDir + "_testingLoss.csv", "w")
	csv.write(str(test_cost))


	### Visualize on Tensorboard
	with tf.device("/cpu:0"):
		embedding = tf.Variable(tf.stack(embedded_data, axis=0), trainable=False, name='embedding')

	sess.run(tf.global_variables_initializer())
	config = projector.ProjectorConfig()
	embed= config.embeddings.add()
	embed.tensor_name = 'embedding:0'
	embed.metadata_path = os.path.join(LogDir + '/metadata.tsv')
	projector.visualize_embeddings(writer, config)
	sess.close()
	return embedded_data




def main(_):
	optparser = OptionParser()
	optparser.add_option('-i', '--inputFile',
						 dest='input',
						 help='path to input file',
						 default=None)
	optparser.add_option('-m', '--metadata',
						 dest='meta',
						 help='path to metadata file',
						 default=None)
	optparser.add_option('-o', '--outputDir',
						 dest='out',
						 help='name of the output directory',
						 default="TEST",
						 type='string')

	(options, args) = optparser.parse_args()
	InputFile = None
	if options.input is not None:
		InputFile = options.input
	else:
		print('No input filename specified, system with exit\n')
		sys.exit('System will exit')

	MetadataFile = None
	if options.meta is not None:
		MetadataFile = options.meta
	else:
		print('No metadata file specified, system with exit\n')
		sys.exit('System will exit')

	OutDir = options.out


	global LogDir
	LogDir = os.getcwd() + "/logs/" + str(OutDir) + "_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
	if not os.path.exists(LogDir):
		os.makedirs(LogDir)
	print("InputFile = ", InputFile)
	print("MetadataFile = ", MetadataFile)
	print("OutputPrefix = ", OutDir)
	print("LogDir = ", LogDir)


	d,f,s = import_data(InputFile)
	global num_features
	global num_samples
	num_samples = len(s)
	num_features = len(f)
	print("# of Features = ", num_features)
	print("# of Samples = ", num_samples)
	print("shape of data = ", d.shape)


	generate_metadata(s, MetadataFile)
	embedded_data = train(d)
	np.savetxt(LogDir+"/diabimmune_latent_25_new.csv", embedded_data.transpose(), delimiter=",")



if __name__ == '__main__':
	tf.app.run(main=main)