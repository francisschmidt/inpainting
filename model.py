import numpy as np 
import theano 
import theano.tensor as T 
import lasagne
import lasagne.layers as layers

# Input info
input_n_channel = 3           
input_height = 64
input_width = 64

# Output info
output_n_channel = 3
output_height = 32
output_width = 32


def OneLayerMLP(batchsize, input_var=None):

	network = layers.InputLayer(shape=(None,input_n_channel, input_height, input_width), input_var=input_var)
	network = layers.DropoutLayer(network, p=0.2)
	network = layers.DenseLayer(network, num_units=15000, nonlinearity=lasagne.nonlinearities.rectify)
	network = layers.DropoutLayer(network, p=0.5)
	network = layers.DenseLayer(network, num_units=output_n_channel*output_height*output_width, nonlinearity=lasagne.nonlinearities.rectify)
	network = layers.ReshapeLayer(network, shape=(batchsize, output_n_channel, output_height, output_width))

	return network

def simpleConv(input_var=None, num_units=32):
	
	network = layers.InputLayer(shape=(None,input_n_channel, input_height, input_width), input_var=input_var)
	
	network = layers.Conv2DLayer(network, num_filters=num_units,filter_size=(9,9))
	network = layers.MaxPool2DLayer(network, pool_size=(2, 2))
	network = layers.Conv2DLayer(network, num_filters=num_units,filter_size=(9,9))
	network = layers.MaxPool2DLayer(network, pool_size=(2, 2))
	network = layers.Conv2DLayer(network, num_filters=1000,filter_size=(10,10))

	network = layers.DenseLayer(layers.DropoutLayer(network, p=0.2), num_units=1000)
	network = layers.DenseLayer(layers.DropoutLayer(network, p=0.5), num_units=1000)

	network = layers.ReshapeLayer(network, shape=(input_var.shape[0],1000,1,1))
	'''
	network = layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(4,4))
	network = layers.Upscale2DLayer(network, 2)
	network = layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(5,5))
	network = layers.Upscale2DLayer(network, 2)
	network = layers.TransposedConv2DLayer(network, num_filters=3, filter_size=(9,9))
	'''
	network = layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(8,8))
	network = layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(9,9))
	network = layers.TransposedConv2DLayer(network, num_filters=num_units, filter_size=(9,9))
	network = layers.TransposedConv2DLayer(network, num_filters=3, filter_size=(9, 9))
	
	return network