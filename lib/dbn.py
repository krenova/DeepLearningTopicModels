"""
"""
from __future__ import print_function, division
import os
import sys
import timeit

import numpy

import theano
import theano.tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams

from lib.mlp import HiddenLayer, LogisticRegression
from lib.rbm import RBM, RSM


# start-snippet-1
class DBN(object):
    """Deep Belief Network

    A deep belief network is obtained by stacking several RBMs on top of each
    other. The hidden layer of the RBM at layer `i` becomes the input of the
    RBM at layer `i+1`. The first layer RBM gets as input the input of the
    network, and the hidden layer of the last RBM represents the output. When
    used for classification, the DBN is treated as a MLP, by adding a logistic
    regression layer on top.
    """

    def __init__(self, numpy_rng=None, theano_rng=None, n_ins=784,
                 hidden_layers_sizes=[500, 500], n_outs=10):
        """This class is made to support a variable number of layers.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: numpy random number generator used to draw initial
                    weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                           generated based on a seed drawn from `rng`

        :type n_ins: int
        :param n_ins: dimension of the input to the DBN

        :type hidden_layers_sizes: list of ints
        :param hidden_layers_sizes: intermediate layers size, must contain
                               at least one value

        :type n_outs: int
        :param n_outs: dimension of the output of the network
        """

        self.sigmoid_layers = []
        self.rbm_layers = []
        self.params = []
        self.params_rbm = []
        self.n_layers = len(hidden_layers_sizes)
        self.hidden_layers_sizes = hidden_layers_sizes

        assert self.n_layers > 0
        
        if not numpy_rng:
            numpy_rng = numpy.random.RandomState(123)
        if not theano_rng:
            self.theano_rng = T.shared_randomstreams.RandomStreams(1234)
        # allocate symbolic variables for the data

        # the data is presented as rasterized images
        self.x = T.matrix('x')

        # the labels are presented as 1D vector of [int] labels
        self.y = T.ivector('y')
        # end-snippet-1
        # The DBN is an MLP, for which all weights of intermediate
        # layers are shared with a different RBM.  We will first
        # construct the DBN as a deep multilayer perceptron, and when
        # constructing each sigmoidal layer we also construct an RBM
        # that shares weights with that layer. During pretraining we
        # will train these RBMs (which will lead to chainging the
        # weights of the MLP as well) During finetuning we will finish
        # training the DBN by doing stochastic gradient descent on the
        # MLP.

        for i in range(self.n_layers):
            # construct the sigmoidal layer

            # the size of the input is either the number of hidden
            # units of the layer below or the input size if we are on
            # the first layer
            if i == 0:
                input_size = n_ins
            else:
                input_size = hidden_layers_sizes[i - 1]

            # the input to this layer is either the activation of the
            # hidden layer below or the input of the DBN if you are on
            # the first layer
            if i == 0:
                layer_input = self.x
            else:
                layer_input = self.sigmoid_layers[-1].output
            print( 'Building layer: ' + str(i) )
            print( '   Input units: ' + str(input_size) )
            print( '  Output units: ' + str(hidden_layers_sizes[i]) )
            sigmoid_layer = HiddenLayer(rng=numpy_rng,
                                        input=layer_input,
                                        n_in=input_size,
                                        n_out=hidden_layers_sizes[i],
                                        activation=T.nnet.sigmoid)

            # add the layer to our list of layers
            self.sigmoid_layers.append(sigmoid_layer)

            # its arguably a philosophical question...  but we are
            # going to only declare that the parameters of the
            # sigmoid_layers are parameters of the DBN. The visible
            # biases in the RBM are parameters of those RBMs, but not
            # of the DBN.
            self.params.extend(sigmoid_layer.params)

            # Construct an RBM that share weights with this layer
            # First layer will be a RSM for inputs of count data
            # while the other hidden layers will be RBMs
            if i == 0: 
                rbm_layer = RSM(input=layer_input,
                                n_visible=input_size,
                                n_hidden=hidden_layers_sizes[i],
                                W=sigmoid_layer.W,
                                hbias=sigmoid_layer.b)
            else:
                rbm_layer = RBM(numpy_rng=numpy_rng,
                                theano_rng=self.theano_rng,
                                input=layer_input,
                                n_visible=input_size,
                                n_hidden=hidden_layers_sizes[i],
                                W=sigmoid_layer.W,
                                hbias=sigmoid_layer.b)
            self.rbm_layers.append(rbm_layer)
            self.params_rbm.extend(rbm_layer.params)

        # We now need to add a logistic layer on top of the MLP
        self.logLayer = LogisticRegression(
            input=self.sigmoid_layers[-1].output,
            n_in=hidden_layers_sizes[-1],
            n_out=n_outs)
        self.params.extend(self.logLayer.params)

        # compute the cost for second phase of training, defined as the
        # negative log likelihood of the logistic regression (output) layer
        self.finetune_cost = self.logLayer.negative_log_likelihood(self.y)

        # compute the gradients with respect to the model parameters
        # symbolic variable that points to the number of errors made on the
        # minibatch given by self.x and self.y
        self.errors = self.logLayer.errors(self.y)

    def pretraining_functions(self, train_set_x, batch_size, k):
        '''
        Generates a list of functions, for performing one step of
        gradient descent at a given layer. The function will require
        as input the minibatch index, and to train an RBM you just
        need to iterate, calling the corresponding function on all
        minibatch indexes.

        :type train_set_x: theano.tensor.TensorType
        :param train_set_x: Shared var. that contains all datapoints used
                            for training the RBM
        :type batch_size: int
        :param batch_size: size of a [mini]batch
        :param k: number of Gibbs steps to do in CD-k / PCD-k

        '''

        # index to a [mini]batch
        index = T.lscalar('index')  # index to a minibatch
        learning_rate = T.scalar('lr')  # learning rate to use

        # begining of a batch, given `index`
        batch_begin = index * batch_size
        # ending of a batch given `index`
        batch_end = batch_begin + batch_size

        pretrain_fns = []
        for rbm in self.rbm_layers:

            # get the cost and the updates list
            # using CD-k here (persisent=None) for training each RBM.
            # TODO: change cost function to reconstruction error
            persistent_chain = theano.shared(numpy.zeros((batch_size, rbm.n_hidden),
                                                 dtype=theano.config.floatX),
                                     borrow=True)
            
            cost, updates = rbm.get_cost_updates(learning_rate,
                                                 persistent=persistent_chain, k=k)

            # compile the theano function
            fn = theano.function(
                inputs=[index, theano.In(learning_rate, value=0.1)],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[batch_begin:batch_end]
                }
            )
            # append `fn` to the list of functions
            pretrain_fns.append(fn)

        return pretrain_fns
    
    def auto_encoding(self, input, 
                      batch_size =500, learning_rate = None, 
                      add_noise = None, obj_fn = 'cross_entropy'):
        
        if learning_rate is None:
            learning_rate = 1/batch_size
            
        # Encoding input data
        train_set_x = input
        N_input_x = train_set_x.shape[0]
        if add_noise:
            assert type(add_noise) == float or type(add_noise) == int, "'add_noise' must be either None, float or int"
            noise = T.matrix('noise')
            train_noise = self.theano_rng.normal(
                size=(N_input_x.eval(), self.hidden_layers_sizes[-1]),
                avg=0, 
                std=add_noise, 
                ndim=None
            )
        fwd_pass = self.x
        
        for i in range(self.n_layers):
            _, fwd_pass = self.rbm_layers[i].propup(fwd_pass)
           
        # Decoding encoded input data
        if add_noise:
            fwd_pass += noise
        
        # Decoding encoded input data
        for i in reversed(range(self.n_layers)):
            _, fwd_pass = self.rbm_layers[i].propdown(fwd_pass)
            
        if obj_fn == 'cross_entropy':
            # ------ Objective Function: multinomial cross entropy ------ #
            #L = - T.sum(self.x * T.log(fwd_pass), axis=1)
            x_normalized = self.x / self.rbm_layers[0].input_rSum[:,None]
            L = - T.sum(x_normalized * T.log(fwd_pass), axis=1)
        else:
            # ------ Objective Function: square error ------ #
            L = T.sum( T.pow( fwd_pass - (self.x/self.dbn.rbm_layers[0].input_rSum[:,None]), 2), axis=1)
            
        # mean cost
        cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params_rbm)
        
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params_rbm, gparams)
        ]
        
        index = T.lscalar('index')
        if add_noise:
            train_dae = theano.function(
                inputs=[index],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[index * batch_size: (index + 1) * batch_size],
                    noise: train_noise[index * batch_size: (index + 1) * batch_size]
                }
            )
        else:
            train_dae = theano.function(
                inputs=[index],
                outputs=cost,
                updates=updates,
                givens={
                    self.x: train_set_x[index * batch_size: (index + 1) * batch_size]
                }
            )
            
        return train_dae
    
    

    def apply_dropout(self, input, corruption_level):
        return self.theano_rng.binomial(size=input.shape, n=1,
                                        p=1 - corruption_level,
                                        dtype=theano.config.floatX) * input 
    
    def build_finetune_functions(self, x, y,
                                 batch_size, learning_rate,
                                 drop_out = None,
                                 split_prop = [0.65,0.15,0.20]):
        '''Generates a function `train` that implements one step of
        finetuning, a function `validate` that computes the error on a
        batch from the validation set, and a function `test` that
        computes the error on a batch from the testing set

        :type datasets: list of pairs of theano.tensor.TensorType
        :param datasets: It is a list that contain all the datasets;
                        the has to contain three pairs, `train`,
                        `valid`, `test` in this order, where each pair
                        is formed of two Theano variables, one for the
                        datapoints, the other for the labels
        :type batch_size: int
        :param batch_size: size of a minibatch
        :type learning_rate: float
        :param learning_rate: learning rate used during finetune stage

        '''
        assert y.shape[0].eval() == x.shape[0].eval(), "independent and target length do not match!"
        assert len(split_prop) == 3 and type(split_prop) == list, \
            "'split_prop' cannot have more or less than 3 inputs and must in list format"
        
        N = y.shape[0].eval()
        split_prop = numpy.array(split_prop)
        split_prop = split_prop / split_prop.sum()
        idx_rand = numpy.random.randint(N, size=N)
        idx_train = idx_rand[:int(N*split_prop[0])]
        idx_valid = idx_rand[len(idx_train):(len(idx_train)+int(N*split_prop[1]))]
        idx_test  = idx_rand[(len(idx_train)+len(idx_valid)):]
                                  
        (train_set_x, train_set_y) = (x[idx_train,], y[idx_train])
        (valid_set_x, valid_set_y) = (x[idx_valid,], y[idx_valid])
        (test_set_x, test_set_y)   = (x[idx_test,],  y[idx_test])
        
        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.shape[0].eval()
        n_train_batches //= batch_size
        n_valid_batches = valid_set_x.shape[0].eval()
        n_valid_batches //= batch_size
        n_test_batches = test_set_x.shape[0].eval()
        n_test_batches //= batch_size

        index = T.lscalar('index')  # index to a [mini]batch

        if drop_out:
            
            assert type(drop_out) == list, "'drop_out' variable must be none or or a list of proportions"
            assert len(drop_out) == (self.n_layers+1), "len of 'drop_out' list must equal number of layers"
            
            x_dropout = T.matrix('x_dropout')
            fwd_pass = self.x
            
            for i in range(self.n_layers):
                self.sigmoid_layers[i].input = self.apply_dropout(fwd_pass, drop_out[i])
                fwd_pass = self.sigmoid_layers[i].output
            
            self.logLayer.input = self.apply_dropout(fwd_pass, drop_out[self.n_layers])
            finetune_cost_dropout = self.logLayer.negative_log_likelihood(self.y)
            
            # compute the gradients with respect to the model parameters
            gparams = T.grad(finetune_cost_dropout, self.params)
            
            # compute list of fine-tuning updates
            updates = []
            for param, gparam in zip(self.params, gparams):
                updates.append((param, param - gparam * learning_rate))

            train_fn = theano.function(
                inputs=[index],
                outputs=finetune_cost_dropout,
                updates=updates,
                givens={
                    self.x: train_set_x[
                        index * batch_size: (index + 1) * batch_size
                    ],
                    self.y: train_set_y[
                        index * batch_size: (index + 1) * batch_size
                    ]
                }
            )
            
        else:
            
            # compute the gradients with respect to the model parameters
            gparams = T.grad(self.finetune_cost, self.params)

            # compute list of fine-tuning updates
            updates = []
            for param, gparam in zip(self.params, gparams):
                updates.append((param, param - gparam * learning_rate))

            train_fn = theano.function(
                inputs=[index],
                outputs=self.finetune_cost,
                updates=updates,
                givens={
                    self.x: train_set_x[
                        index * batch_size: (index + 1) * batch_size
                    ],
                    self.y: train_set_y[
                        index * batch_size: (index + 1) * batch_size
                    ]
                }
            )        
        
        test_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: test_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: test_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )

        valid_score_i = theano.function(
            [index],
            self.errors,
            givens={
                self.x: valid_set_x[
                    index * batch_size: (index + 1) * batch_size
                ],
                self.y: valid_set_y[
                    index * batch_size: (index + 1) * batch_size
                ]
            }
        )
        
        # Create a function that scans the entire validation set
        def valid_score():
            return [valid_score_i(i) for i in range(n_valid_batches)]

        # Create a function that scans the entire test set
        def test_score():
            return [test_score_i(i) for i in range(n_test_batches)]

        return n_train_batches, train_fn, valid_score, test_score

    
    
    def predict(self, input, batch_size = 2000, prob = False):
        
        train_set_x = input
        N_input_x = train_set_x.shape[0]
        
        # compute number of minibatches for scoring
        if train_set_x.get_value(borrow=True).shape[0] % batch_size != 0:
            N_splits = int( numpy.floor(train_set_x.get_value(borrow=True).shape[0] / batch_size) + 1 )
        else:
            N_splits = int( numpy.floor(train_set_x.get_value(borrow=True).shape[0] / batch_size) )

        # allocate symbolic variables for the data
        index = T.lscalar()    # index to a [mini]batch
        
        if prob:
            output = theano.function(
                 inputs = [index],
                 outputs = self.logLayer.p_y_given_x,
                 givens={
                    self.x: train_set_x[index * batch_size: (index + 1) * batch_size]
                 }
            )  
        else:
            output = theano.function(
                 inputs = [index],
                 outputs = self.logLayer.y_pred,
                 givens={
                    self.x: train_set_x[index * batch_size: (index + 1) * batch_size]
                 }
            )  
            
        return numpy.concatenate( [output(ii) for ii in range(N_splits)], axis=0 )
