"""This tutorial introduces restricted boltzmann machines (RBM) using Theano.

Boltzmann Machines (BMs) are a particular form of energy-based model which
contain hidden variables. Restricted Boltzmann Machines further restrict BMs
to those without visible-visible and hidden-hidden connections.
"""
from __future__ import print_function, division
import os
import sys
import timeit
from six.moves import cPickle as pickle

import numpy as np
import pandas as pd

import theano
import theano.tensor as T

from lib.deeplearning import autoencoder

os.chdir('/home/ekhongl/Codes/DL - Topic Modelling')

dat_x = np.genfromtxt('data/dtm_2000_20news.csv', dtype='float32', delimiter=',', skip_header = 1)
dat_y = dat_x[:,0]
dat_x = dat_x[:,1:]
vocab =  np.genfromtxt('data/dtm_2000_20news.csv', dtype=str, delimiter=',', max_rows = 1)[1:]
test_input = theano.shared(dat_x)

#model = autoencoder( architecture = [2756, 500, 500, 128], opt_epochs = [900,5,10], model_src = 'params/dbn_params')


#model.train(test_input, batch_size = 50, learning_rate = 1/20000, epochs = 60000, obj_fn = 'mean_sq', output_path = #'params/ae_params_meansq2')

# theano 2
#model.train(test_input, batch_size = 200, learning_rate = 1/20000, epochs = 60000, output_path = 'params/ae_params_noise')

# theano 3
#model.train(test_input, batch_size = 500, learning_rate = 1/20000, epochs = 60000, output_path = 'params/ae_params_noise_2')

#-----------------------------------------------------------------------------------------

model = autoencoder( architecture = [2000, 500, 500, 128], opt_epochs = [110,15,10], model_src = 'params_2000/dbn_params_pretrain')
# theano 1
#model.train(test_input, add_noise = 16, batch_size = 500, learning_rate = 1/20000, epochs = 60000, \
#            output_path = 'params_2000/ae_train_noise')

# theano 2
model.train(test_input,  batch_size = 200, learning_rate = 1/20000, epochs = 60000, \
            output_path = 'params_2000/ae_train_nonoise')