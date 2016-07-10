import argparse
import cPickle
import sys
from build import model
from load import load_dataset
from theano import tensor as T
import theano
import numpy as np

__author__ = 'Matej'

def load_model(filename):
    """
    Load the model from file and return the model. We expect here directory to be provided in filename, otherwise
    it will look for filename in current directory.
    :param filename: directory/filename
    :return: model
    """
    try:
        f = open(filename, 'rb')
    except EOFError:
        print "File " + filename + " does not exist!"
        sys.exit(0)
    else:
        with f:
            loaded_obj = cPickle.load(f)
    return loaded_obj

parser = argparse.ArgumentParser()
parser.add_argument("teX", help="Provide filename for test dataset you want to use (reads). It should have been in 'media/'"
                                "directory and filename should end with '-teX.fasta.gz'", type=str)
parser.add_argument("teX", help="Provide filename for test dataset you want to use (classes). It should have been in 'media/'"
                                "directory and filename should end with '-teY.fasta.gz'", type=str)
# parser.add_argument("-l", "--length", help="Input length - how big chunks you want to be sequences sliced to.", default=100, type=int)
results = parser.parse_args()
teX = load_dataset(results.teX)
teY = load_dataset(results.teY)

# initialize matrices
X = T.ftensor4()
Y = T.fmatrix()

#### THIS SETTINGS MUST BE SAME AS IN BUILD.PY FILE ####
conv1_stride=4
stride1=2
downscale1=3
stride2=2
downscale2=2
stride3=2
downscale3=1
#### THIS SETTINGS MUST BE SAME AS IN BUILD.PY FILE ####

# TODO add parsing filename of the best model

# models filename is now of format best_params-[timestamp].pkl
params = load_model("models/params.pkl")
l1, l2, l3, l4, py_x = model(X=X, w=params[0], w2=params[1], w3=params[2], w4=params[3], p_drop_conv=0., p_drop_hidden=0., w_o=params[4])
y_x = T.argmax(py_x, axis=1) # maxima predictions
y_x = py_x

# compile only predict function
predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

# TODO - magic with predicted values :)

print predict(teX)