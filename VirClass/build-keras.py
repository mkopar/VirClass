"""Build neural net architecture with keras library and save best model."""

import argparse
import hashlib
import random
import time
from keras.models import Sequential
from keras.layers import Dense, MaxPooling1D, Activation, Convolution1D, Flatten, Dropout
from VirClass.VirClass.load_ncbi import get_gids
from VirClass.VirClass.load import load_data


def load_data_sets_from_file(data_set_filename, debug_mode=False, input_length=100):
    """
    Load datasets from given filename.

    Function for loading data set before initializing neural network and evaluating the model.
    If you get/build data set in fasta format beforehand, provide filename in argument when calling build.py. We expect
    provided filename is located in media directory.
    If filename is empty/not provided, then specify all the needed params for expected data loading. Filename is build
    from md5 from sorted genome IDs, depth param, sample param, read_size param, one_hot param and seed param. File is
    saved in fasta format and zipped with gzip.
    :param data_set_filename: data set filename
    :param debug_mode: if the flag for debug is present, run in debug mode (controlled seed, smaller taxonomy)
    :param input_length: input length
    :return: train and test data sets as well as number of classes
    """
    transmission_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    test = 0.2
    depth = 4
    sample = 0.2
    # read_size = 100
    one_hot = True
    # taxonomy_el_count = 20 and seed = 0 for debug only
    if debug_mode:
        seed = 0
        taxonomy_el_count = 20
    else:
        seed = random.randint(0, 4294967295)
        taxonomy_el_count = -1
    if not data_set_filename:
        data_set_filename = "%s_%d_%.3f_%d_%d_%d_%d%s" % (hashlib.md5(str(sorted(get_gids())).encode('utf-8')).hexdigest(), depth,
                                                          sample, input_length, one_hot, seed, taxonomy_el_count,
                                                          ".fasta.gz")

    return load_data(filename=data_set_filename, test=test, depth=depth, read_size=input_length,
                     trans_dict=transmission_dict, sample=sample, seed=seed, taxonomy_el_count=taxonomy_el_count)


def init_keras(train_input_shape, p_drop_conv, p_drop_hidden, number_of_classes, convolution_params=None):
    """
    Initialize keras neural net.

    Perform convolution and everything belonging to it.
    Code is split into 4 "blocks" - 3 blocks of computation and last block where we have fully connected layer.

    In each block we perform convolution, followed by rectify activation function. After that we perform max pool
    and add some noise with dropout. This is repeated for layers 2 and 3.
    Last layer is fully connected layer, which connects all the filters to 500 hidden nodes. These nodes are then
    connected to the output nodes.
    """
    if convolution_params is None:
        raise AssertionError("USER ERROR - convolution parameters are empty!")
        # convolution1_stride = convolution_params[0]
        # stride1 = convolution_params[1]
        # downscale1 = convolution_params[2]
        # stride2 = convolution_params[3]
        # downscale2 = convolution_params[4]
        # stride3 = convolution_params[5]
        # downscale3 = convolution_params[6]
    # block of computation
    # border_mode='valid': apply filter wherever it completely overlaps with the input
    temp_model = Sequential()

    for i, (conv_stride, stride, downscale, filters) in enumerate(conv_params):
        if i == 0:
            temp_model.add(Convolution1D(filters, conv_stride, border_mode='valid',
                                         input_shape=train_input_shape, subsample_length=stride))
        else:
            temp_model.add(Convolution1D(filters, conv_stride, border_mode='valid', subsample_length=stride))
        temp_model.add(Activation("relu"))  # rectified linear unit
        temp_model.add(MaxPooling1D(pool_length=downscale, stride=stride, border_mode='valid'))
        temp_model.add(Dropout(p=p_drop_conv))
    # temp_model.add(Convolution1D(stride1, convolution1_stride, border_mode='valid', input_shape=train_input_shape))
    # temp_model.add(Activation("relu"))  # rectified linear unit
    # temp_model.add(MaxPooling1D(pool_length=downscale1, stride=stride1, border_mode='valid'))
    # temp_model.add(Dropout(p=p_drop_conv))
    #
    # temp_model.add(Convolution1D(stride2, 1, border_mode='valid'))
    # temp_model.add(Activation("relu"))  # rectified linear unit
    # temp_model.add(MaxPooling1D(pool_length=downscale2, stride=stride2, border_mode='valid'))
    # temp_model.add(Dropout(p=p_drop_conv))
    #
    # temp_model.add(Convolution1D(stride3, 1, border_mode='valid'))
    # temp_model.add(Activation("relu"))  # rectified linear unit
    # temp_model.add(MaxPooling1D(pool_length=downscale3, stride=stride3, border_mode='valid'))
    # temp_model.add(Dropout(p=p_drop_conv))

    temp_model.add(Flatten())
    temp_model.add(Activation("relu"))
    temp_model.add(Dropout(p=p_drop_hidden))

    temp_model.add(Dense(number_of_classes))
    temp_model.add(Activation("softmax"))
    return temp_model


if __name__ == "__main__":
    print("start:", time.strftime('%X %x %Z'))
    # start = int(time.gmtime(0))

    # arguments - filename, debug, input length
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="Provide filename for dataset you want to use. It MUST be in 'media/' "
                                                 "folder in current directory. If None is given then dataset is built "
                                                 "from NCBI database.", type=str, default="")
    parser.add_argument("-d", "--debug", action="store_true", help="If you want the enable debug mode, call program "
                                                                   "with this flag.", default=False)
    parser.add_argument("-l", "--length", help="Input length - how big chunks you want to be sequences sliced to.",
                        default=100, type=int)
    results = parser.parse_args()
    filename = results.filename
    debug = results.debug
    read_size = results.length
    debug = True

    trX, teX, trY, teY, trteX, trteY, num_of_classes, train_class_sizes = \
        load_data_sets_from_file(filename, debug_mode=debug, input_length=read_size)

    print("trX shape:", trX.shape)
    input_len = trX.shape[1]
    # trX = trX.reshape(-1, 1, input_len)
    # teX = teX.reshape(-1, 1, input_len)
    # trteX = trteX.reshape(-1, 1, input_len)

    # params for model and cascade initialization
    # conv1_stride = 4
    # stride_1 = 2
    # downscale_1 = 3
    # stride_2 = 2
    # downscale_2 = 2
    # stride_3 = 2
    # downscale_3 = 1
    # conv params: ((conv1_stride, stride1, downscale1, filters1), (conv2_stride, stride2, downscale2, filters2), ...)
    conv_params = ((4, 2, 3, 32), (1, 2, 2, 48), (1, 2, 1, 64))
    # make dynamic
    # conv_params = [conv1_stride, (stride_1, downscale_1), (stride_2, downscale_2), (stride_3, downscale_3)]

    # X = K.ftensor4()
    # Y = K.fmatrix()

    input_shape = trX.shape

    model = init_keras(input_shape, 0.2, 0.5, num_of_classes, conv_params)
    # params, X, Y, cost, updates, y_x = init_net(num_of_classes, input_len, conv_params)

    # compile train and predict function
    # train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    # predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

    trX = trX.reshape(1, -1, input_len)
    teX = teX.reshape(1, -1, input_len)
    trteX = trteX.reshape(1, -1, input_len)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(trX, trY, batch_size=128, nb_epoch=len(trX) / 128, verbose=1)
    # error here - wrong input shape
    model.evaluate(trteX, trteY, 128, verbose=1)

    model.save("model.h5")

    # model.predict(teX, 32, verbose=1)

    # epsilon = 0.005
    #  if evaluation score does not improve for 0,5% every 5 tries, then stop evaluating - we get best model
    # best_score = -1
    # count_best = 10
    # iter = 100
    # for i in range(iter):
    #     for start, end in zip(range(0, len(trX), 128), range(128, len(trX), 128)):
    #         cost = train(trX[start:end], trY[start:end])
    #     # evaluate model on train_test data, not on test data!
    #     curr_score = (np.mean(np.argmax(trteY, axis=1) == predict(trteX)))
    #     print "%.5f" % curr_score
    #     if count_best == 0:
    #         break
    #     elif curr_score > (best_score + epsilon):
    #         best_score = curr_score
    #         count_best = 10  # reset counter
    #     else:
    #         count_best -= 1

    # save best model to models directory; add also parameters for net initialization and train class sizes
    # params.append(train_class_sizes)
    # params.append(conv_params)
    # save_model("models/best_model_with_params-%d.pkl" % int(time.time()), params)

    print("stop:", time.strftime('%X %x %Z'))
    # print "elapsed: ", (int(time.gmtime(0)) - start)
