#!/usr/bin/env python
#-*- coding: utf8 -*-

import os
import sys
import json
from   optparse import OptionParser
import time
reload(sys)
sys.setdefaultencoding('utf-8')

import network3
from network3 import Network
from network3 import ConvPoolLayer, FullyConnectedLayer, SoftmaxLayer
from network3 import ReLU

# global variable
VERBOSE = 0

if __name__ == '__main__':
	parser = OptionParser()
	parser.add_option("--verbose", action="store_const", const=1, dest="verbose", help="verbose mode")
	(options, args) = parser.parse_args()

	if options.verbose == 1 : VERBOSE = 1

	training_data, validation_data, test_data = network3.load_data_shared()
	mini_batch_size = 10

	net = Network([
				ConvPoolLayer(image_shape=(mini_batch_size, 1, 28, 28), 
					filter_shape=(20, 1, 5, 5), 
					poolsize=(2, 2), 
					activation_fn=ReLU),
				ConvPoolLayer(image_shape=(mini_batch_size, 20, 12, 12), 
					filter_shape=(40, 20, 5, 5), 
					poolsize=(2, 2), 
					activation_fn=ReLU),
				FullyConnectedLayer(
					n_in=40*4*4, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
				FullyConnectedLayer(
					n_in=1000, n_out=1000, activation_fn=ReLU, p_dropout=0.5),
				SoftmaxLayer(n_in=1000, n_out=10, p_dropout=0.5)], 
				mini_batch_size)

	net.SGD(training_data, 40, mini_batch_size, 0.03, validation_data, test_data)

