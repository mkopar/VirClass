import hashlib
import random
import time
import cPickle
import numpy as np
import sys
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from load import load_data
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.signal.downsample import DownsampleFactorMax
from load_ncbi import get_gids

theano.config.floatX='float32'

srng = RandomStreams()

def floatX(X):
    """
    Convert X to float32 or float64, depending on how we configure theano.
    """
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    """
    Initialize model parameters.
    """
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    """
    Rectifier.
    """
    return T.maximum(X, 0.)

def softmax(X):
    """
    Numerically stable softmax.
    """
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    """
    Way of injecting noise into our network - help us with overfitting and local extrema.
    Randomly drop values and scale rest.
    """
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
    """
    Learn different networks better and learning them quicker.
    """
    grads = T.grad(cost=cost, wrt=params)
    updates = []
    for p, g in zip(params, grads):
        acc = theano.shared(p.get_value() * 0.)
        acc_new = rho * acc + (1 - rho) * g ** 2
        gradient_scaling = T.sqrt(acc_new + epsilon)
        g = g / gradient_scaling
        updates.append((acc, acc_new))
        updates.append((p, p - lr * g))
    return updates

def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden, w_o):
    """
        Perform convolution and everything belonging to it.
        Code is split into 4 "blocks" - 3 blocks of computation and last block where we have fully connected layer.

        In each block we perform convolution, followed by rectify activation function. After that we perform max pool
        and add some noise with dropout. This is repeated for layers 2 and 3.
        Last layer is fully connected layer, which connects all the filters to 500 hidden nodes. These nodes are then
        connected to the output nodes.
    """
    # block of computation
    # border_mode='valid': apply filter wherever it completely overlaps with the input
    l1a = rectify(conv2d(X, w, border_mode='valid', subsample=(1, conv1_stride))) # stride along one (horizontal) dimension only
    l1 = max_pool_2d(l1a, (1, downscale1), st=(1, stride1)) # (1,1)=(vertical, horizontal) downscale, st=(1, step): move to every stride1 column and perform max_pooling there
    l1 = dropout(l1, p_drop_conv)

    # repeat block
    l2a = rectify(conv2d(l1, w2, subsample=(1, 1))) # stride along horizontal
    l2 = max_pool_2d(l2a, (1, downscale2), st=(1, stride2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3, subsample=(1, 1))) # stride along horizontal
    l3b = max_pool_2d(l3a, (1, downscale3), st=(1, stride3))
    l3 = T.flatten(l3b, outdim=2) # convert from 4tensor to normal matrix
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx

def save_model(filename, model):
    """
    Get name of the file where you want to save the model. We expect here directory to be provided in filename, otherwise
    it will be saved in current directory.
    :param filename: directory / filename
    :param model: model to be saved
    :return: None
    """
    print "saving model..."
    f = open(filename, 'wb')
    cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print "model saved..."

def load_dataset(filename, debug=False):
    """
    Function for loading dataset before initializing neural network and evaluating the model.
    If you get/build dataset in fasta format beforehand, provide filename in argument when calling build.py. We expect
    provided filename is located in media directory.
    If filename is empty/not provided, then specify all the needed params for expected data loading. Filename is build from
    md5 from sorted genome IDs, depth param, sample param, read_size param, onehot param and seed param. File is saved
    in fasta format and zipped with gzip.
    :param filename: filename, given from
    :return: train and test datasets as well as number of classes
    """
    transmission_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    test = 0.2
    depth = 4
    sample = 0.2
    read_size = 100
    onehot = True
    # taxonomy_el_count = 20 and seed = 0 for debug only
    if debug:
        seed = 0
        taxonomy_el_count = 20
    else:
        seed = random.randint(0, sys.maxint)
        taxonomy_el_count = -1
    if not filename:
        filename = "%s_%d_%.3f_%d_%d_%d_%d%s" % (hashlib.md5(str(sorted(get_gids()))).hexdigest(), depth, sample, read_size, onehot,
                                                 seed, taxonomy_el_count, ".fasta.gz")
    trX, teX, trY, teY, trteX, trteY, num_of_classes = load_data(filename=filename, test=test, depth=depth,
                                                                 read_size=read_size, transmission_dict=transmission_dict,
                                                                 sample=sample, seed=seed, taxonomy_el_count=taxonomy_el_count)
    return trX, teX, trY, teY, trteX, trteY, num_of_classes

def init_net(num_of_classes, input_len):
    """
    Major initialize of the neural net is in this method. You can adjust convolutional window size for each layer,
    number of filters for each layer and all the cascade parameters for every layer. We also initialize and define weights
    for neural net.
    :param num_of_classes: number of classes
    :param input_len: read (sequence chunk) length
    :return: weights in param variable, X and Y matrices, cost function, update function and maxima prediction
    """
    cwin1=4*6  # multiples of 4 because of data representation
    cwin2=3
    cwin3=2

    num_filters_1=32 / 2  # how many different filters to learn at each layer
    num_filters_2=48 / 2
    num_filters_3=64 / 2
    # size of convolution windows, for each layer different values can be used
    w = init_weights((num_filters_1, 1, 1, cwin1)) # first convolution, 32 filters, stack size 1, 1 rows, cwin1 columns
    w2 = init_weights((num_filters_2, num_filters_1, 1, cwin2)) # second convolution, 64 filters, stack size 32 (one stack for each filter from previous layer), 1 row, cwin2 columns
    w3 = init_weights((num_filters_3, num_filters_2, 1, cwin3)) # third convolution, 128 filters, stack size 64 (one stack for each filter from previous layes), 1 row, cwin3 columns

    print "#### CONVOLUTION PARAMETERS ####"
    print "cwin1 %d" % cwin1
    print "cwin2 %d" % cwin2
    print "cwin3 %d" % cwin3
    print "num_filters_1 %d" % num_filters_1
    print "num_filters_2 %d" % num_filters_2
    print "num_filters_3 %d" % num_filters_3

    # convolution: filters are moved by one position at a time, see parameter subsample=(1, 1)
    #
    # max pooling:
    #   scaling the input before applying the maxpool filter and
    #   displacement (stride) when sliding the max pool filters

    # l1 conv:
    es = input_len
    es = (es - cwin1 + 1)
    es = es / conv1_stride
    # l1 max_pool:
    es = DownsampleFactorMax.out_shape((1, es), (1, downscale1), st=(1, stride1))[1] # downscale for first layer
    print "l1 es:", es

    # l2 conv:
    es = (es - cwin2 + 1)
    # l2 max_pool:
    es = DownsampleFactorMax.out_shape((1, es), (1, downscale2), st=(1, stride2))[1] # downscale for second layer
    print "l2 es:", es

    # l3 conv:
    es = (es - cwin3 + 1)
    # l3 max_pool:
    es = DownsampleFactorMax.out_shape((1, es), (1, downscale3), st=(1, stride3))[1] # downscale for third layer
    print "l3 es:", es

    # downscaling is performed so that we correctly set number of filters in last layer

    w4 = init_weights((num_filters_3 * es, 500))  # fully conected last layer, connects the outputs of 128 filters to 500 (arbitrary) hidden nodes, which are then connected to the output nodes
    w_o = init_weights((500, num_of_classes))  # number of exptected classes

    # matrix types
    X = T.ftensor4()
    Y = T.fmatrix()

    noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5, w_o)
    l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0., w_o)
    y_x = T.argmax(py_x, axis=1)  # maxima predictions

    cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y)) # classification matrix to optimize - maximize the value that is actually there and minimize the others
    params = [w, w2, w3, w4, w_o]
    updates = RMSprop(cost, params, lr=0.001) # update function

    return params, X, Y, cost, updates, y_x

print "start:", time.strftime('%X %x %Z')
start = time.gmtime(0)

# TODO - parse filename argument
filename = ""

trX, teX, trY, teY, trteX, trteY, num_of_classes = load_dataset(filename, debug=True)

print(trX.shape)
input_len = trX.shape[1] # save input length for further use
trX = trX.reshape(-1, 1, 1, input_len)
teX = teX.reshape(-1, 1, 1, input_len)

# params for model and cascade initialization
conv1_stride=4
stride1=2
downscale1=3
stride2=2
downscale2=2
stride3=2
downscale3=1

params, X, Y, cost, updates, y_x = init_net(num_of_classes, input_len)

# compile train and predict function
train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

epsilon = 0.005  # if evaluation score does not improve for 0,5% every 5 tries, then stop evaluating - we get best model
best_score = -1
count_best = 10
iter = 100
for i in range(iter):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    # evaluate model on train_test data, not on test data!
    curr_score = (np.mean(np.argmax(trteY, axis=1) == predict(trteX)))
    print "%.5f" % curr_score
    if count_best == 0:
        break
    elif curr_score > (best_score + epsilon):
        best_score = curr_score
        count_best = 5  # reset counter
    else:
        count_best -= 1

# save best model to models directory
save_model("models/best_params-%d.pkl" % int(time.time()), params)

print "stop:", time.strftime('%X %x %Z')
print "elapsed: ", (time.gmtime(0) - start)