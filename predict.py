from collections import Counter
from cnet_3 import model, load_dataset
from cnet_3 import load_model
from theano import tensor as T
import theano
import numpy as np

__author__ = 'Matej'

# load data
trX, teX, trY, teY, num_of_classes = load_dataset()
# trX, trY, num_of_classes not used

# initialize matrices
X = T.ftensor4()
Y = T.fmatrix()

conv1_stride=4

stride1=2
downscale1=3
stride2=2
downscale2=2
stride3=2
downscale3=1

params = load_model("models/params.pkl")
l1, l2, l3, l4, py_x = model(X=X, w=params[0], w2=params[1], w3=params[2], w4=params[3], p_drop_conv=0., p_drop_hidden=0., w_o=params[4])
y_x = T.argmax(py_x, axis=1)
y_x = py_x

predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

print predict(teX)