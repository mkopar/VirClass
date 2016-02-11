import math
import sys
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
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
    print "rectify", X.shape
    return T.maximum(X, 0.)

def softmax(X):
    e_x = T.exp(X - X.max(axis=1).dimshuffle(0, 'x'))
    return e_x / e_x.sum(axis=1).dimshuffle(0, 'x')

def dropout(X, p=0.):
    if p > 0:
        retain_prob = 1 - p
        X *= srng.binomial(X.shape, p=retain_prob, dtype=theano.config.floatX)
        X /= retain_prob
    return X

def RMSprop(cost, params, lr=0.001, rho=0.9, epsilon=1e-6):
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
conv_stride=4

stride1=2
downscale1=3  # mogoce na vrednost 2

stride2=2
downscale2=2

stride3=2
downscale3=1

def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    # TODO spremeni max_pool argumente
    l1a = rectify(conv2d(X, w, border_mode='valid', subsample=(1, conv_stride))) # stride along one (horizontal) dimension only
    l1 = max_pool_2d(l1a, (1, downscale1), st=(1, stride1)) # (1,1)=(vertical, horizontal) downscale, st=(1, step): move to every stride1 column and perform max_pooling there
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2, subsample=(1, 1))) # stride along horizontal
    # l2 = max_pool_2d(l2a, (2, 2))
    l2 = max_pool_2d(l2a, (1, downscale2), st=(1, stride2))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3, subsample=(1, 1))) # stride along horizontal
    # l3b = max_pool_2d(l3a, (2, 2))
    l3b = max_pool_2d(l3a, (1, downscale3), st=(1, stride3))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx

trX, teX, trY, teY, num_of_classes = seq_load(onehot=True, seed=7970223320302509880)
# trX, teX, trY, teY, num_of_classes = seq_load(onehot=True)

print(trX.shape)
# TODO spremeni argumente tukaj
# trX = trX.reshape(-1, 1, 1, 100)
# teX = teX.reshape(-1, 1, 1, 100)
trX = trX.reshape(-1, 1, 1, trX.shape[1])
teX = teX.reshape(-1, 1, 1, trX.shape[1])

X = T.ftensor4()
Y = T.fmatrix()

# size of convolution windows, for each layer different values can be used
cwin1=6  # 5 ali 6 nukleotidov da dobimo vzorce ki so dovolj redki da so uporabni
cwin2=5
cwin3=3

num_filters_1=32 # how many different filters to lear at each layer
num_filters_2=48
num_filters_3=64
w = init_weights((num_filters_1, 1, 1, cwin1)) # first convolution, 32 filters, stack size 1, 1 rows, cwin1 columns
w2 = init_weights((num_filters_2, num_filters_1, 1, cwin2)) # second convolution, 64 filters, stack size 32 (one stack for each filter from previous layer), 1 row, cwin2 columns
w3 = init_weights((num_filters_3, num_filters_2, 1, cwin3)) # third convolution, 128 filters, stack size 64 (one stack for each filter from previous layes), 1 row, cwin3 columns


# TODO popravi es (najbrz mora bit 400) in nato se ostale formule

# expected
es = 100
# es = trX.shape[1]
# l1 conv:
es = (es - cwin1 + 1)
# es = int(math.ceil((es - cwin1 + 1) / 4))  # ?? mogoce +4?
# l1 max_pool:
es = DownsampleFactorMax.out_shape((1, es), (1, downscale1), st=(1, stride1))[1]

# l2 conv:
es = (es - cwin2 + 1)
# l2 max_pool:
es = DownsampleFactorMax.out_shape((1, es), (1, downscale2), st=(1, stride2))[1]

# l3 conv:
es = (es - cwin3 + 1)
# l3 max_pool:
es = DownsampleFactorMax.out_shape((1, es), (1, downscale3), st=(1, stride3))[1]

w4 = init_weights((num_filters_3 * es, 500))  # fully conected last layer, connects the outputs of 128 filters to 500 (arbitrary) hidden nodes, which are then connected to the output nodes
w_o = init_weights((500, num_of_classes))  # stevilo koncnih razredov

noise_l1, noise_l2, noise_l3, noise_l4, noise_py_x = model(X, w, w2, w3, w4, 0.2, 0.5)
l1, l2, l3, l4, py_x = model(X, w, w2, w3, w4, 0., 0.)
y_x = T.argmax(py_x, axis=1)


cost = T.mean(T.nnet.categorical_crossentropy(noise_py_x, Y))
params = [w, w2, w3, w4, w_o]
updates = RMSprop(cost, params, lr=0.001)

train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

for i in range(100):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print np.mean(np.argmax(teY, axis=1) == predict(teX))