# Deep Learning Topic Modelling
This repo is a collection of neural network tools, built on top of the [Theano](http://deeplearning.net/software/theano/) framework with the primary objective of performing Topic Modelling. Topic modelling is commonly approached using the [Latent Dirichlet Allocation](https://www.cs.princeton.edu/~blei/papers/BleiNgJordan2003.pdf) (LDA) or [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) (LSA) algorithms but more recently, with the advent of modelling count data using [Restricted Boltzmann Machines](https://en.wikipedia.org/wiki/Restricted_Boltzmann_machine) (RBMs), also known as the [Replicated Softmax Model](https://papers.nips.cc/paper/3856-replicated-softmax-an-undirected-topic-model.pdf) (RSM), Deep Neural Network models were soon adapted to perform Topic Modelling with results empirically shown to be in better agreement with human's semantic interpretations (see [[1](http://www.utstat.toronto.edu/~rsalakhu/papers/topics.pdf)]). 

The overview of the model construction comprises of 3 phases.

![addfdfdf](http://i.imgur.com/pVs5Rvb.png)_Image taken from [[1](http://www.utstat.toronto.edu/~rsalakhu/papers/topics.pdf)]_
1. The first is to design the Network architecture using a RSM to model the input data followed by stacking as many layers of RBMs as deemed reasonable to model the outputs of the RSM. The stacking of RBMs (and RSM) leads what is called a Deep Generative Model or a more specifically in this case, a [Deep Belief Network](http://deeplearning.net/tutorial/DBN.html) (DBN). Like single layered RSMs or RBMs, this multi-layered network is bidirectional. It is able to generate encoded outputs from input data and more distinctly, generate 'input' data using encoded data. However, unlike single layered networks, multilayered networks are more likely to be able to generate input data with more similarity to the training data due to their ability to capture structure high-dimensions.


2. Once the network's architecture is defined, pre-training then follows. Pre-training has empircally been shown to improve the accuracy (or other measures) of neural network models and one of the main hypothesis to justify this phenomena is that pre-training helps configure the network to start off at a more optimal point compared to a random initialization.


3. After pre-training, the DBN is unrolled to produce an [Auto-Encoder](http://deeplearning.net/tutorial/dA.html). Auto-Encoders take in input data and reduce them into their lower dimensional representations before reconstructing them to be as close as possible to their input form. This is effectively a form of data compression but more importantly, it also means that the lower dimensional representations hold sufficient information about its higher dimensional input data for reconstruction to be feasible. Once training, or more appropriately fine-tuning in this case, is completed, only the segment of the Auto-Encoder that produces the lower dimensional output is retained.


As these lower dimensional representations of the input data are easier to work with, algorithms that can be used to establish similarities between data points could be applied to the compressed data, to indirectly estimate similarities between the input data. For text data broken down into counts of words in documents, this dimension reduction technique can be used as an alternative method of information retrieval or topic modelling. 


## Codes
Much of codes are a modification and addition of codes to the libraries provided by the developers of Theano at http://deeplearning.net/tutorial/. While Theano may now have been slightly overshadowed by its more prominent counterpart, [TensorFlow](https://www.tensorflow.org/), the tutorials and codes at deeplearning.net still provides a good avenue for anyone who wants to get a deeper introduction to deep learning and the mechanics of it. Moreover, given the undeniable inspiration that TensorFlow had from Theano, once Theano is mastered, the transition from Theano to TensorFlow should be almost seamless.

The main codes are found in the **lib** folder, where we have:

|no.| codes| description |
|:-:|:-----:|:----|
|1  | [rbm.py](https://github.com/krenova/DeepLearningTopicModels/blob/master/lib/rbm.py) | contains the RBM and RSM classes |
|2  | [mlp.py](https://github.com/krenova/DeepLearningTopicModels/blob/master/lib/mlp.py) | contains the sigmoid and logistic regression classes |
|3  | [dbn.py](https://github.com/krenova/DeepLearningTopicModels/blob/master/lib/dbn.py) | the DBN class to construct the netowrk functions for pre-training and fine tuning|
|4  | [deeplearning.py](https://github.com/krenova/DeepLearningTopicModels/blob/master/lib/deeplearning.py)| wrapper around the DBN class|


## Examples
Examples of using the tools in this repo are written in jupyter notebooks. The data source for the example can be sourced from 
http://qwone.com/~jason/20Newsgroups/20news-18828.tar.gz.

|no.| codes| description |
|:-:|:-----:|:----|
|1  | [data_proc.ipynb](https://github.com/krenova/DeepLearningTopicModels/blob/master/notebooks/data_proc.ipynb) | notebook to process the raw data (please change the data dir name accordingly) |
|2  | [train_dbn.ipynb](https://github.com/krenova/DeepLearningTopicModels/blob/master/notebooks/train_dbn.ipynb) | demonstrates how to pre-train the DBN and subsequently turn it into a Multilayer Perceptron for document classification |
|3  | [train_sae.ipynb](https://github.com/krenova/DeepLearningTopicModels/blob/master/notebooks/train_sae.ipynb) | training the pre-trained model from train_dbn.ipynb as an Auto-Encoder |
|4  | [topic_modelling.ipynb](https://github.com/krenova/DeepLearningTopicModels/blob/master/notebooks/topic_modelling.ipynb)| (using R here) clustering the lower dimensional output of the Auto-Encoder|



## Reading References
1. http://www.utstat.toronto.edu/~rsalakhu/papers/topics.pdf
2. http://deeplearning.net/tutorial/rbm.html
3. http://deeplearning.net/tutorial/DBN.html
4. http://deeplearning.net/tutorial/dA.html
5. http://deeplearning.net/tutorial/SdA.html