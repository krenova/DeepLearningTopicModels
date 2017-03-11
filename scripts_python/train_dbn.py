from __future__ import print_function, division
import os
import sys
import timeit
from six.moves import cPickle as pickle

import numpy as np
import pandas as pd

import theano
import theano.tensor as T

from lib.deeplearning import deepbeliefnet

os.chdir('~/Codes/DL - Topic Modelling')
            
dat_x = np.genfromtxt('data/dtm_2000_20news_6class.csv', dtype='float32', delimiter=',', skip_header = 1)
dat_y = dat_x[:,0]
dat_x = dat_x[:,1:]
vocab =  np.genfromtxt('data/dtm_2000_20news_6class.csv', dtype=str, delimiter=',', max_rows = 1)[1:]
x = theano.shared(dat_x)
y = T.cast(dat_y, dtype='int32')


#model = deepbeliefnet(architecture = [2756, 500, 500, 128], opt_epochs = [900,5,10], n_outs = 20, predefined_weights = 'params/dbn_params')

#model.train(x=x, y=y, training_epochs = 10000, batch_size = 70, output_path = 'params/dbn_params_trained_long')

# theano_dropout
#model.train(x=x, y=y, training_epochs = 10000, learning_rate = (1/70)/2, batch_size = 120,
#            drop_out = [0.2, .5, .5, .5], output_path = 'params/dbn_params_dropout')

# theano_dropout2
#model.train(x=x, y=y, training_epochs = 10000, learning_rate = (1/70)/2, batch_size = 120,
#            drop_out = [0.2, .3, .4, .5], output_path = 'params/dbn_params_dropout2')


 
model = deepbeliefnet(n_outs = 6, architecture = [2000, 500, 500, 128], 
                      opt_epochs = [110,15,10], predefined_weights = 'params_2000/dbn_params_pretrain')

#theano_6 class
model.train(x=x, y=y, training_epochs = 10000, learning_rate = 1/160, batch_size = 50,
            drop_out = [0.2, .5, .5, .5], output_path = 'params_2000/dbn_params_dropout')
