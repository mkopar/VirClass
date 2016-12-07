"""
Module predict.
"""

import argparse
import sys
import operator
import numpy as np
import _pickle as cPickle
from theano import tensor as T, function
from .build import model
from .load import load_dataset

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
        print("File " + filename + " does not exist!")
        sys.exit(0)
    else:
        with f:
            loaded_obj = cPickle.load(f)
    return loaded_obj

parser2 = argparse.ArgumentParser()
parser2.add_argument("teX", help="Provide filename for test dataset you want to use (reads). It should have been in "
                                 "'media/' directory and filename should end with '-teX.fasta.gz'", type=str)
parser2.add_argument("best_model", help="Provide filename for the best model. Filename must include directory. Must be "
                                        "of ormat 'best_model_with_params-[timestamp].pkl'.", type=str)
parser2.add_argument("-teY", help="Provide filename for test dataset you want to use (classes). It should have been in "
                                  "'media/' directory and filename should end with '-teY.fasta.gz'", type=str)
results = parser2.parse_args()
teX = np.asarray(load_dataset(results.teX))
teX = teX.reshape(-1, 1, 1, teX.shape[1])
teY = load_dataset(results.teY)
best_model = results.best_model

# teX_filename = "media/2114bef791b6111f12575439a7bbed73_4_0.200_100_1_0_20-teX.fasta.gz"
# teY_filename = "media/2114bef791b6111f12575439a7bbed73_4_0.200_100_1_0_20-teY.fasta.gz"
# model_filename = "models/best_model_with_params-1468304923-improving-eval.pkl"
# teX = np.asarray(load_dataset(teX_filename))
# teX = teX.reshape(-1, 1, 1, teX.shape[1])
# teY = np.asarray(load_dataset(teY_filename))
# best_model = model_filename

# initialize matrices
X = T.ftensor4()
Y = T.fmatrix()

# models filename is of format best_params-[timestamp].pkl
params = load_model(best_model)
# params = [w, w2, w3, w4, w_o, train_class_sizes, conv_params]

conv_params = params[6]
# conv_params = (conv1_stride, stride1, downscale1, stride2, downscale2, stride3, downscale3)
# conv1_stride = 4
conv1_stride = conv_params[0]
# stride1 = 2
stride1 = conv_params[1]
# downscale1 = 3
downscale1 = conv_params[2]
# stride2 = 2
stride2 = conv_params[3]
# downscale2 = 2
downscale2 = conv_params[4]
# stride3 = 2
stride3 = conv_params[5]
# downscale3 = 1
downscale3 = conv_params[6]

l1, l2, l3, l4, py_x = model(X=X, w=params[0], w2=params[1], w3=params[2], w4=params[3], p_drop_conv=0.,
                             p_drop_hidden=0., w_o=params[4], conv_params=conv_params)
# y_x = T.argmax(py_x, axis=1) # maxima predictions
y_x = py_x

# compile only predict function
predict = function(inputs=[X], outputs=y_x, allow_input_downcast=True)

# load class sizes
class_sizes = params[5]

final_results = predict(teX)
sum_results = np.sum(final_results, axis=0)

# detect which classes are present in teX
assert len(class_sizes.values()) == sum_results.shape[0]
weighted_vector = [float(a) / b for a, b in zip(sum_results, class_sizes.values())]
norm_vector = [x / sum(weighted_vector) for x in weighted_vector]
norm_dict = dict(zip(range(len(norm_vector)), norm_vector))
sorted_norm_dict = sorted(norm_dict.items(), key=operator.itemgetter(1), reverse=True)
print("raw predicted values: ", sum_results)
print()
print("weighted and normed predicted values: ", weighted_vector)
print()
print("sorted probabilities: ", sorted_norm_dict)
print()
print("expected classes: ", np.unique(teY))
