from __future__ import print_function, division
import os
import sys
import timeit
from six.moves import cPickle as pickle

import numpy as np
import pandas as pd

import theano
import theano.tensor as T

from lib.mlp import HiddenLayer, LogisticRegression
from lib.rbm import RBM, RSM
from lib.dbn import DBN

os.chdir('/home/ekhongl/Codes/DL - Topic Modelling')

class InitializationError(object):
    '''this error is raised when the input definitions for the corresponding class is conflicting'''

########################################################################################################
## Deep Belief Net #####################################################################################
########################################################################################################
class deepbeliefnet(object):
            
    def __init__(self, architecture = [2000, 500, 500, 128], 
                 opt_epochs = [], predefined_weights = None, n_outs = 1):
    
        # ensure proper class initialization
        assert len(architecture) > 1 , "architecture definition must include both the hidden layers AND input layer"
        
        # numpy random generator
        numpy_rng = np.random.RandomState(123)
        
        # reconstruct the DBN class
        self.hidden_layers_sizes = architecture[1:]
        self.n_layers = len(self.hidden_layers_sizes)
        self.params = []        #params for the MLP
        self.params_rbm = []    #params for the RBMs
        self.n_ins = architecture[0]
        self.n_outs = n_outs
        print('... building the model')
        self.dbn = DBN( numpy_rng=numpy_rng, 
                        n_ins=self.n_ins,
                        n_outs=self.n_outs,
                        hidden_layers_sizes=self.hidden_layers_sizes )
        
        # loading pre-trained weights
        if predefined_weights is not None and len(opt_epochs) >0 :
            self.new_model = False
            
            # load saved model
            for i in range(self.n_layers):
                model_pkl = os.path.join(predefined_weights,
                            'dbn_layer' + str(i) + '_epoch_' + str(opt_epochs[i]) + '.pkl')
                self.dbn.rbm_layers[i].__setstate__(pickle.load(open(model_pkl, 'rb')))
            # extract the model parameters
            for i in range(self.n_layers):
                self.params_rbm.extend(self.dbn.rbm_layers[i].params)
            
            print('Pre-trained DBN model from "' + predefined_weights + '" loaded.')
        
        # loading fine-tuned weights
        elif predefined_weights is not None and len(opt_epochs) == 0:
            
            self.dbn.params = pickle.load(open(predefined_weights, 'rb'))
            for i in range(self.n_layers):
                self.dbn.sigmoid_layers[i].__setstate__ (self.dbn.params[(i*2):(i*2+2)])
            self.dbn.logLayer.__setstate__ (self.dbn.params[(self.n_layers*2):(self.n_layers*2+2)])
            
            print('Fine-tuned (or MLP) model from "' + predefined_weights + '" loaded.')
        
        # error in class initialization
        elif (predefined_weights is None and len(opt_epochs) > 0):
            
            raise InitializationError("Either 'opt_epochs' and predefined_weights' is empty or filled at the same time")
            
        # creating a new generative model
        else:
            self.new_model = True
            for i in range(self.n_layers):
                self.params_rbm.extend(self.dbn.rbm_layers[i].params)
    
        
                        
    def pretrain(self, input, pretraining_epochs=100, pretrain_lr=None, 
                        k=1, batch_size=800, output_path = 'params/dbn_params_test'):    
        
        train_set_x = input
        
        #---------------------------------------------------------------------------------------#
        # ensure class initialization matches input definition before function execution
        assert train_set_x.get_value(borrow=True).shape[1] == self.n_ins, \
                        "Input data dimensions must match initialized dimensions!"
        
        if pretrain_lr is None:
            pretrain_lr = [1/batch_size] * self.n_layers
        else:
            if type(pretrain_lr) is not list:
                pretrain_lr = [pretrain_lr] * self.n_layers
            elif len(pretrain_lr) != self.n_layers:
                pretrain_lr = [1/batch_size] * self.n_layers
                raise Warning('Warning: pretrain_lr len parameter not equal to number of hidden layers!')
                print('Reverting pretrain_lr to the default values (1/batch_size).')
        if type(pretraining_epochs) is not list:
            pretraining_epochs = [pretraining_epochs] * self.n_layers
        #---------------------------------------------------------------------------------------#
        
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0] // batch_size
        
        #########################
        # PRETRAINING THE MODEL #
        #########################
        if not os.path.isdir(output_path):
            os.makedirs(output_path)

        print('... getting the pretraining functions')
        pretraining_fns = self.dbn.pretraining_functions(train_set_x=train_set_x,
                                                        batch_size=batch_size,
                                                        k=k)
        
        print('... pre-training the model')
        start_time = timeit.default_timer()

        # Pre-train layer-wise
        for i in range(self.dbn.n_layers):

            # go through pretraining epochs
            lproxy = []
            for epoch in range(pretraining_epochs[i]):

                # go through the training set
                mean_cost = []
                for batch_index in range(n_train_batches):
                    mean_cost.append(pretraining_fns[i](index=batch_index, \
                                                        lr=pretrain_lr[i]))

                # calculating the epoch's mean proxy likelihood value
                lproxy += [np.mean(mean_cost)]
                print('Pre-training layer %i, epoch %d, cost ' % (i, epoch), end=' ')
                print(lproxy[epoch])

                # save the model parameters for each epoch
                epoch_pickle = output_path +'/dbn_layer' + str(i) + \
                               '_epoch_' + str(epoch) + '.pkl'
                #path_epoch_pickle = os.path.join( os.getcwd(), epoch_pickle)
                pickle.dump( self.dbn.rbm_layers[i].__getstate__(), \
                             open(epoch_pickle, 'wb')    )

            # save the proxy likelihood profile
            pd.DataFrame(data = {'likelihood_proxy' :lproxy} ). \
               to_csv( output_path +'/lproxy_layer_' + str(i) + '.csv', index = False)
            
            end_time = timeit.default_timer()
            print('The pretraining for layer ' + str(i) +
                  ' ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)
        
            
    def train(self, x, y, split_prop = [0.65,0.15,0.20], training_epochs = 100,
                    batch_size=800, learning_rate=None, drop_out = None,
                    output_path = 'params/dbn_params_trained'):
        
        if learning_rate is None: learning_rate = 1/batch_size
        
        # get the training, validation and testing function for the model
        print('... getting the finetuning functions')
        n_train_batches, train_fn, validate_model, test_model = self.dbn.build_finetune_functions(
            x=x, 
            y=y, 
            split_prop = split_prop, 
            batch_size=batch_size,
            learning_rate=learning_rate,
            drop_out = drop_out
        )

        print('... finetuning the model')
        
        #########################
        #  TRAINING THE MODEL   #
        #########################
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
            
        # look as this many examples regardless
        patience = 4 * n_train_batches

        # wait this much longer when a new best is found
        patience_increase = 2.

        # a relative improvement of this much is considered significant
        improvement_threshold = 0.995

        # go through this many minibatches before checking the network on
        # the validation set; in this case we check every epoch
        validation_frequency = min(n_train_batches, patience / 2)

        best_validation_loss = np.inf
        test_score = 0.
        start_time = timeit.default_timer()

        done_looping = False
        epoch = 0
        
        while (epoch < training_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in range(n_train_batches):

                train_fn(minibatch_index)
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:

                    validation_losses = validate_model()
                    this_validation_loss = np.mean(validation_losses, dtype='float64')
                    print('epoch %i, minibatch %i/%i, validation error %f %%' % (
                        epoch,
                        minibatch_index + 1,
                        n_train_batches,
                        this_validation_loss * 100.
                        )
                    )

                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:

                        # improve patience if loss improvement is good enough
                        if (this_validation_loss < best_validation_loss *
                                improvement_threshold):
                            patience = max(patience, iter * patience_increase)

                        # save best validation score and iteration number
                        best_validation_loss = this_validation_loss
                        best_iter = iter

                        # test it on the test set
                        test_losses = test_model()
                        test_score = np.mean(test_losses, dtype='float64')
                        print(('     epoch %i, minibatch %i/%i, test error of '
                               'best model %f %%') %
                              (epoch, minibatch_index + 1, n_train_batches,
                              test_score * 100.))
                        
                        #-----------------------------------------------------------------------------#
                        #----------- Saving the current best model -----------------------------------#
                        #-----------------------------------------------------------------------------#
                        print('Saving model...')
                        
                        tmp_params = []
                        for i in range(self.n_layers):
                            tmp_params.extend( self.dbn.sigmoid_layers[i].__getstate__ () )
                        tmp_params.extend( self.dbn.logLayer.__getstate__ () )
                        
                        pickle.dump( tmp_params, \
                                     open(output_path +'/trained_dbn.pkl', 'wb')    )
                        
                        del tmp_params
                        
                        print('...model saved.')
                        #-----------------------------------------------------------------------------#

                if patience <= iter:
                    done_looping = True
                    break

        end_time = timeit.default_timer()
        print(('Optimization complete with best validation score of %f %%, '
               'obtained at iteration %i, '
               'with test performance %f %%'
               ) % (best_validation_loss * 100., best_iter + 1, test_score * 100.))
        print('The fine tuning ran for %.2fm' % ((end_time - start_time) / 60.), file=sys.stderr)

    def score(self, input, batch_size = 2000):
        
        x = T.matrix('x')
        self.dbn.rbm_layers[0].input_rSum = x.sum(axis=1)
        train_set_x = input
        N_input_x = train_set_x.shape[0]

        # compute number of minibatches for scoring
        if train_set_x.get_value(borrow=True).shape[0] % batch_size != 0:
            N_splits = int( np.floor(train_set_x.get_value(borrow=True).shape[0] / batch_size) + 1 )
        else:
            N_splits = int( np.floor(train_set_x.get_value(borrow=True).shape[0] / batch_size) )

        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch

        # input_rSum must be specified for the RSM layer
        activation = x
        for i in range(self.n_layers):
            _, activation = self.dbn.rbm_layers[i].propup(activation)

        # it is ok for a theano function to have no output
        # the purpose of train_rbm is solely to update the RBM parameters
        score = theano.function(
            inputs = [index],
            outputs = activation,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size]
            }
        )

        return np.concatenate( [score(ii) for ii in range(N_splits)], axis=0 )

    def predict(self, input, batch_size = 2000, prob = False):
        
        return self.dbn.predict(input = input, batch_size = batch_size, prob = prob)
        
        
        
########################################################################################################
## Auto-Encoder ########################################################################################
########################################################################################################
class autoencoder(object):
        
    def __init__(self, architecture = [], opt_epochs = [], model_src = 'params/dbn_params', param_type = 'dbn'):
        
        # ensure model source directory is valid
        assert type(model_src) == str or model_src is not None, "dir to load model parameters not indicated"
        if len(opt_epochs)>0:
            assert len(architecture) == (len(opt_epochs)+1) , "len of network inputs must be 1 more than len of hidden layers"
        
        # reconstruct the DBN class
        self.params = []
        self.hidden_layers_sizes = architecture[1:]
        self.n_layers = len(self.hidden_layers_sizes)
        self.dbn = DBN( n_ins=architecture[0],
                        hidden_layers_sizes = self.hidden_layers_sizes )
        self.theano_rng = T.shared_randomstreams.RandomStreams(1234)
        
        if param_type == 'dbn':
            # load saved model
            print('Loading the pre-trained Deep Belief Net parameters...')
            for i in range(self.n_layers):
                model_pkl = os.path.join(model_src,
                            'dbn_layer' + str(i) + '_epoch_' + str(opt_epochs[i]) + '.pkl')
                self.dbn.rbm_layers[i].__setstate__(pickle.load(open(model_pkl, 'rb')))
                # extract the model parameters
                self.params.extend(self.dbn.rbm_layers[i].params)
            print('...model loaded.')

        
        else:
            print('Loading the trained auto-encoder parameters.')
            print('...please ensure that the auto-encoder params matches the defined architecture.')
            for i in range(self.n_layers):
                model_pkl = model_src +'/ae_layer_' + str(i) + '.pkl'
                self.dbn.rbm_layers[i].__setstate__(pickle.load(open(model_pkl, 'rb')))
                self.params.extend(self.dbn.rbm_layers[i].params)

    
    def get_corrupted_input(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input 
    
    def get_cost_updates(self, learning_rate, add_noise = False, obj_fn = 'cross_entropy'):
        """ This function computes the cost and the updates for one trainng
        step of the dA """
         
        # Encoding input data
        fwd_pass = self.x
        for i in range(self.n_layers):
            _, fwd_pass = self.dbn.rbm_layers[i].propup(fwd_pass)
           
        # Decoding encoded input data
        if add_noise:
            fwd_pass += self.noise
        
        # Decoding encoded input data
        for i in reversed(range(self.n_layers)):
            _, fwd_pass = self.dbn.rbm_layers[i].propdown(fwd_pass)
            
        if obj_fn == 'cross_entropy':
            # ------ Objective Function: multinomial cross entropy ------ #
            #L = - T.sum(self.x * T.log(fwd_pass), axis=1)
            x_normalized = self.x / self.dbn.rbm_layers[0].input_rSum[:,None]
            L = - T.sum(x_normalized * T.log(fwd_pass), axis=1)
        else:
            # ------ Objective Function: square error ------ #
            # rightfully, should follow by L = L / len(vocab), but linear scaling
            # does not affect search for minima and therefore omitted
            L = T.sum( T.pow( fwd_pass*self.dbn.rbm_layers[0].input_rSum[:,None] - self.x, 2), axis=1)
            
        # mean cost
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)
    
    
    def train(self, input, epochs= 1000, batch_size = 500, learning_rate = None, add_noise = None,
                    obj_fn = 'cross_entropy', output_path = 'params/ae_params'):
        
        N_input_x = input.get_value(borrow=True).shape[0]
        
        # compute number of minibatches for training
        if N_input_x % batch_size != 0:
            N_splits = int( np.floor(N_input_x / batch_size) + 1 )
        else:
            N_splits = int( np.floor(N_input_x / batch_size) )    
    
         # get the autoecoding training function
        print('... getting the finetuning functions')
        train_dae = self.dbn.auto_encoding(
            input=input, 
            batch_size = batch_size,
            learning_rate = learning_rate,
            add_noise = add_noise, 
            obj_fn = obj_fn
        )

        print('... finetuning the model')
        
        #########################
        #  TRAINING THE MODEL   #
        #########################
        if not os.path.isdir(output_path):
            os.makedirs(output_path)
            
        start_time = timeit.default_timer()
        
        # go through training epochs
        cost_profile = []
        for epoch in range(epochs):
            
            # go through trainng set
            c = []
            for batch_index in range(N_splits):
                c.append(train_dae(batch_index))
            
            # saving and printing iterations
            cost_profile += [np.mean(c, dtype='float64')]
            if epoch % 100 == 0:
                #-----------------------------------------------------------------------------#
                #----------- Saving the current best model -----------------------------------#
                #-----------------------------------------------------------------------------#
                print('Saving model...')
                # save the model parameters for all layers
                for i in range(self.n_layers):
                    pickle.dump( self.dbn.rbm_layers[i].__getstate__(), \
                                 open(output_path +'/ae_layer_' + str(i) + '.pkl', 'wb')  )
                print('...model saved')
                # save the proxy likelihood profile
                pd.DataFrame(data = {'cross_entropy' : cost_profile} ). \
                   to_csv( output_path +'/cost_profile.csv', index = False)
                print('Training epoch %d, cost ' % epoch, cost_profile[epoch])
                #-----------------------------------------------------------------------------#
        
        print('Saving model...')
                # save the model parameters for all layers
        for i in range(self.n_layers):
            pickle.dump( self.dbn.rbm_layers[i].__getstate__(), \
                        open(output_path +'/ae_layer_' + str(i) + '.pkl', 'wb')  )
        print('...model saved')
        # save the proxy likelihood profile
        pd.DataFrame(data = {'cross_entropy' : cost_profile} ). \
           to_csv( output_path +'/cost_profile.csv', index = False)
        print('Training epoch %d, cost ' % epoch, cost_profile[epoch])
        
        end_time = timeit.default_timer()

        training_time = (end_time - start_time)

        print(('Training ran for %.2fm' % (training_time / 60.)), file=sys.stderr)
    
    def score(self, input, batch_size = 2000):    
        train_set_x = input
        N_input_x = train_set_x.shape[0]

        # compute number of minibatches for scoring
        if train_set_x.get_value(borrow=True).shape[0] % batch_size != 0:
            N_splits = int( np.floor(train_set_x.get_value(borrow=True).shape[0] / batch_size) + 1 )
        else:
            N_splits = int( np.floor(train_set_x.get_value(borrow=True).shape[0] / batch_size) )
        
        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch

        # input_rSum must be specified for the RSM layer
        x = T.matrix('x')
        self.dbn.rbm_layers[0].input_rSum = x.sum(axis=1)
        activation = x
        for i in range(self.n_layers):
            _, activation = self.dbn.rbm_layers[i].propup(activation)

        # it is ok for a theano function to have no output
        # the purpose of train_rbm is solely to update the RBM parameters
        score = theano.function(
            inputs = [index],
            outputs = activation,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size]
            }
        )

        return np.concatenate( [score(ii) for ii in range(N_splits)], axis=0 )