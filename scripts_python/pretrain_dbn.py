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
            
dat_x = np.genfromtxt('data/dtm_2000_20news.csv', dtype='float32', delimiter=',', skip_header = 1)
dat_y = dat_x[:,0]
dat_x = dat_x[:,1:]
vocab =  np.genfromtxt('data/dtm_20news.csv', dtype=str, delimiter=',', max_rows = 1)[1:]
x = theano.shared(dat_x)
y = T.cast(dat_y, dtype='int32')




model = deepbeliefnet(architecture = [2000, 500, 500, 128], n_outs = 20)


model.pretrain(input = x, pretraining_epochs= [1000,100,100], output_path = 'params/dbn_params_test')
