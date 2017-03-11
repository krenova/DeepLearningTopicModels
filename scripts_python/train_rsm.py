from __future__ import print_function

import timeit
from six.moves import cPickle as pickle

import numpy as np
import pandas as pd

import theano
import theano.tensor as T
import os
from lib.rbm import RSM

from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


os.chdir('~/Codes/DL - Topic Modelling')


def test_rbm(input, learning_rate=1/1600, 
             training_epochs=5000, batch_size=1600, 
             n_hidden=1500, output_folder = 'model_params'):
    """
    Demonstrate how to train and afterwards sample from it using Theano.

    This is demonstrated on MNIST.

    :param learning_rate: learning rate used for training the RBM

    :param training_epochs: number of epochs used for training

    :param dataset: path the the pickled dataset

    :param batch_size: size of a batch used to train the RBM

    """
    train_set_x = input

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
    
    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    rng = np.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    # initialize storage for the persistent chain (state = hidden
    # layer of chain)
    persistent_chain = theano.shared(np.zeros((batch_size, n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)
    
    # construct the RBM class
    rsm = RSM(input=x, n_visible=train_set_x.get_value(borrow=True).shape[1],
              n_hidden=n_hidden)#, numpy_rng=rng, theano_rng=theano_rng)

    # get the cost and the gradient corresponding to one step of CD-15
    cost, updates = rsm.get_cost_updates( lr=learning_rate,
                                          persistent=persistent_chain, 
                                          k=1 )

    #################################
    #     Training the RBM          #
    #################################
    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    
    # start-snippet-5
    # it is ok for a theano function to have no output
    # the purpose of train_rbm is solely to update the RBM parameters
    train_rbm = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        },
        name='train_rbm'
    )

    plotting_time = 0.
    start_time = timeit.default_timer()

    # go through training epochs
    lproxy = []
    for epoch in range(training_epochs):

        # go through the training set
        mean_cost = []
        for batch_index in range(n_train_batches):
            mean_cost += [train_rbm(batch_index)]
            print('Current iteration: Epoch=' + str(epoch) + ', ' + 'Batch=' + str(batch_index))
            
        # save the model parameters for each epoch
        print('Saving model...')
        epoch_pickle = output_folder + '/rsm_epoch_' + str(epoch) + '.pkl'
        path_epoch_pickle = os.path.join(os.getcwd(), epoch_pickle)
        pickle.dump( rsm.__getstate__(), open(path_epoch_pickle, 'wb'))
        print('...model saved.')
        
        # update current epoch and average cost
        lproxy += [np.mean(mean_cost)]
        print('Training epoch %d, likelihood proxy is ' % epoch, lproxy[epoch])

        
    end_time = timeit.default_timer()

    pretraining_time = (end_time - start_time) - plotting_time
    
    pd.DataFrame(data = {'likelihood_proxy' :lproxy} ). \
                to_csv(output_folder + '/likelihood_proxy.csv', index = False)
        
    print ('Training took %f minutes' % (pretraining_time / 60.))



dat_x = np.genfromtxt('data/dtm_20news.csv', dtype='float32', delimiter=',', skip_header = 1)
dat_y = dat_x[:,0]
dat_x = dat_x[:,1:]
vocab =  np.genfromtxt('data/dtm_20news.csv', dtype=str, delimiter=',', max_rows = 1)[1:]
test_input = theano.shared(dat_x)


test_input = theano.shared(dat_x)

test_rbm(input = test_input)








