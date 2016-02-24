import time
import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import seq_load
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d
from theano.tensor.signal.downsample import DownsampleFactorMax

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
    Way of injecting noise into our network - help us with overfitting.
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

# convolution: filters are moved by one position at a time, see parameter subsample=(1, 1)
#
# max pooling:
#   scaling the input before applying the maxpool filter and
#   displacement (stride) when sliding the max pool filters
conv1_stride=4

stride1=2
downscale1=3

stride2=2
downscale2=2

stride3=2
downscale3=1

def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    """
        Here we perform convolution and everything belonging to it.
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

    # repeat process
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

print "start:", time.strftime('%X %x %Z')
trX, teX, trY, teY, num_of_classes = seq_load(onehot=True, ntrain=100000, ntest=20000) # load data

print(trX.shape)
input_len = trX.shape[1] # save input length for further use
trX = trX.reshape(-1, 1, 1, input_len)
teX = teX.reshape(-1, 1, 1, input_len)

# matrix types
X = T.ftensor4()
Y = T.fmatrix()

# size of convolution windows, for each layer different values can be used
cwin1=4*6  # multiples of 4 because of data representation
cwin2=5
cwin3=3

num_filters_1=32 # how many different filters to learn at each layer
num_filters_2=8
num_filters_3=4
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

w4 = init_weights((num_filters_3 * es, 500))  # fully conected last layer, connects the outputs of n filters to 500 (arbitrary) hidden nodes, which are then connected to the output nodes
w_o = init_weights((500, num_of_classes))

noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5) # noise during training
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.) # no noise for prediction
y_x = T.argmax(py_x, axis=1) # maxima predictions

cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y)) # classification matrix to optimize - maximize the value that is actually there and minimize the others
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost, params, lr=0.001) # update function

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True) # compile train function
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True) # compile predict function

# testing
for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == predict(teX))

print "stop:", time.strftime('%X %x %Z')