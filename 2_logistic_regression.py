import cPickle
import theano
from theano import tensor as T
import numpy as np
from load_mnist import mnist

def floatX(X):
    return np.asarray(X, dtype=theano.config.floatX)

def init_weights(shape):
    return theano.shared(floatX(np.random.randn(*shape) * 0.01))

def model(X, w):
    return T.nnet.softmax(T.dot(X, w))

def save_model(filename, model):
    print "saving model..."
    f = open(filename, 'wb')
    cPickle.dump(model, f, protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()
    print "model saved..."

# loadamo pa filename v train
def load_model(filename):
    f = open(filename, 'rb')
    loaded_obj = cPickle.load(f)
    f.close()
    return loaded_obj

trX, teX, trY, teY = mnist(onehot=True)

X = T.fmatrix()
Y = T.fmatrix()

w = init_weights((784, 10))

py_x = model(X, w)
y_pred = T.argmax(py_x, axis=1)

cost = T.mean(T.nnet.categorical_crossentropy(py_x, Y))
gradient = T.grad(cost=cost, wrt=w)
update = [[w, w - gradient * 0.05]]

train = theano.function(inputs=[X, Y], outputs=cost, updates=update, allow_input_downcast=True)
predict = theano.function(inputs=[X], outputs=y_pred, allow_input_downcast=True)

for i in range(20):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print i, np.mean(np.argmax(teY, axis=1) == predict(teX))

save_model("models/w.pkl", w)

for i in range(20):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train(trX[start:end], trY[start:end])
    print i, np.mean(np.argmax(teY, axis=1) == predict(teX))

print
print "saved model"
w1 = load_model("models/w.pkl")
py_x1 = model(X, w1)
cost1 = T.mean(T.nnet.categorical_crossentropy(py_x1, Y))
gradient1 = T.grad(cost=cost1, wrt=w1)
update1 = [[w1, w1 - gradient1 * 0.05]]
y_pred1 = T.argmax(py_x1, axis=1)

train1 = theano.function(inputs=[X, Y], outputs=cost1, updates=update1, allow_input_downcast=True)
predict1 = theano.function(inputs=[X], outputs=y_pred1, allow_input_downcast=True)

for i in range(20):
    for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
        cost = train1(trX[start:end], trY[start:end])
    print i, np.mean(np.argmax(teY, axis=1) == predict1(teX))
