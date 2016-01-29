import theano
from theano import tensor as T
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
import numpy as np
from load import seq_load
from theano.tensor.nnet.conv import conv2d
from theano.tensor.signal.downsample import max_pool_2d

srng = RandomStreams()

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def rectify(X):
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

def model(X, w, w2, w3, w4, p_drop_conv, p_drop_hidden):
    # TODO spremeni max_pool argumente
    l1a = rectify(conv2d(X, w, border_mode='full'))
    # l1 = max_pool_2d(l1a, (2, 2))  # 0,2 bi blo za nas primer (al 0,5?)
    l1 = max_pool_2d(l1a, (1, 3))
    l1 = dropout(l1, p_drop_conv)

    l2a = rectify(conv2d(l1, w2))
    # l2 = max_pool_2d(l2a, (2, 2))
    l2 = max_pool_2d(l2a, (1, 3))
    l2 = dropout(l2, p_drop_conv)

    l3a = rectify(conv2d(l2, w3))
    # l3b = max_pool_2d(l3a, (2, 2))
    l3b = max_pool_2d(l3a, (1, 3))
    l3 = T.flatten(l3b, outdim=2)
    l3 = dropout(l3, p_drop_conv)

    l4 = rectify(T.dot(l3, w4))
    l4 = dropout(l4, p_drop_hidden)

    pyx = softmax(T.dot(l4, w_o))
    return l1, l2, l3, l4, pyx

num_of_classes = 104
trX, teX, trY, teY = seq_load(number_of_classes=num_of_classes, onehot=True)

trX = trX.reshape(-1, 1, 1, 100)  # TODO spremeni argumente tukaj
teX = teX.reshape(-1, 1, 1, 100)

X = T.ftensor4()
Y = T.fmatrix()

# conv weights (n_kernels, n_channels, kernel_w, kernel_h) --> tukaj bi popravil zadnji dve na 0,3 ali 1,3 ker imam samo vektor
# TODO nastavi ustrezne utezi
# w = init_weights((32, 1, 3, 3))
# w2 = init_weights((64, 32, 3, 3))
# w3 = init_weights((128, 64, 3, 3))
# w4 = init_weights((128 * 3 * 3, 625))
# w_o = init_weights((625, 10))  # 10 ker imamo 10 koncnih vrednosti
w = init_weights((16, 1, 1, 5))
w2 = init_weights((32, 16, 1, 5))
w3 = init_weights((64, 32, 1, 5))
w4 = init_weights((64 * 1 * 10, 350))  # to ne bo delal
w_o = init_weights((350, num_of_classes))  # namesto 10 damo stevilo koncnih razredov

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