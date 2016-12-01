import argparse
import hashlib
import random
import time
from keras.models import Sequential
from keras.layers import Dense, MaxPooling1D, Activation, Convolution1D, Flatten, Dropout
from load_ncbi import get_gids
from load import load_data


def load_datasets_from_file(filename, debug=False, read_size=100):
    """
    Function for loading dataset before initializing neural network and evaluating the model.
    If you get/build dataset in fasta format beforehand, provide filename in argument when calling build.py. We expect
    provided filename is located in media directory.
    If filename is empty/not provided, then specify all the needed params for expected data loading. Filename is build from
    md5 from sorted genome IDs, depth param, sample param, read_size param, onehot param and seed param. File is saved
    in fasta format and zipped with gzip.
    :param filename: filename, given from
    :param debug: if the flag for debug is present, run in debug mode (controlled seed, smaller taxonomy)
    :param read_size: input length
    :return: train and test datasets as well as number of classes
    """
    transmission_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    test = 0.2
    depth = 4
    sample = 0.2
    #read_size = 100
    onehot = True
    # taxonomy_el_count = 20 and seed = 0 for debug only
    if debug:
        seed = 0
        taxonomy_el_count = 20
    else:
        seed = random.randint(0, 4294967295)
        taxonomy_el_count = -1
    if not filename:
        filename = "%s_%d_%.3f_%d_%d_%d_%d%s" % (hashlib.md5(str(sorted(get_gids()))).hexdigest(), depth, sample, read_size, onehot,
                                                 seed, taxonomy_el_count, ".fasta.gz")
    trX, teX, trY, teY, trteX, trteY, \
    num_of_classes, train_class_sizes = load_data(filename=filename, test=test, depth=depth, read_size=read_size,
                                                  transmission_dict=transmission_dict, sample=sample, seed=seed,
                                                  taxonomy_el_count=taxonomy_el_count)
    return trX, teX, trY, teY, trteX, trteY, num_of_classes, train_class_sizes


def init_keras(input_shape, p_drop_conv, p_drop_hidden, number_of_classes, conv_params=None):
    """
        Perform convolution and everything belonging to it.
        Code is split into 4 "blocks" - 3 blocks of computation and last block where we have fully connected layer.

        In each block we perform convolution, followed by rectify activation function. After that we perform max pool
        and add some noise with dropout. This is repeated for layers 2 and 3.
        Last layer is fully connected layer, which connects all the filters to 500 hidden nodes. These nodes are then
        connected to the output nodes.
    """
    if conv_params is not None:
        conv1_stride = conv_params[0]
        stride1 = conv_params[1]
        downscale1 = conv_params[2]
        stride2 = conv_params[3]
        downscale2 = conv_params[4]
        stride3 = conv_params[5]
        downscale3 = conv_params[6]
    # block of computation
    # border_mode='valid': apply filter wherever it completely overlaps with the input
    model = Sequential()
    # layer 1
    model.add(Convolution1D(stride1, conv1_stride, border_mode='valid', input_shape=input_shape))
    model.add(Activation("relu"))  # rectified linear unit
    model.add(MaxPooling1D(pool_length=downscale1, stride=stride1, border_mode='valid'))
    model.add(Dropout(p=p_drop_conv))

    # l1a = rectify(conv2d(X, w, border_mode='valid', subsample=(1, conv1_stride))) # stride along one (horizontal) dimension only
    # l1 = max_pool_2d(l1a, (1, downscale1), st=(1, stride1)) # (1,1)=(vertical, horizontal) downscale, st=(1, step): move to every stride1 column and perform max_pooling there
    # l1 = dropout(l1, p_drop_conv)

    # repeat block
    # l2a = rectify(conv2d(l1, w2, subsample=(1, 1))) # stride along horizontal
    # l2 = max_pool_2d(l2a, (1, downscale2), st=(1, stride2))
    # l2 = dropout(l2, p_drop_conv)

    model.add(Convolution1D(stride2, 1, border_mode='valid'))
    model.add(Activation("relu"))  # rectified linear unit
    model.add(MaxPooling1D(pool_length=downscale2, stride=stride2, border_mode='valid'))
    model.add(Dropout(p=p_drop_conv))

    # l3a = rectify(conv2d(l2, w3, subsample=(1, 1))) # stride along horizontal
    # l3b = max_pool_2d(l3a, (1, downscale3), st=(1, stride3))
    # l3 = T.flatten(l3b, outdim=2) # convert from 4tensor to normal matrix
    # l3 = dropout(l3, p_drop_conv)

    model.add(Convolution1D(stride3, 1, border_mode='valid'))
    model.add(Activation("relu"))  # rectified linear unit
    model.add(MaxPooling1D(pool_length=downscale3, stride=stride3, border_mode='valid'))
    model.add(Dropout(p=p_drop_conv))
    model.add(Flatten())

    # l4 = rectify(T.dot(l3, w4))
    # l4 = dropout(l4, p_drop_hidden)

    # model.add() T.dot?
    model.add(Activation("relu"))
    model.add(Dropout(p=p_drop_hidden))

    # pyx = softmax(T.dot(l4, w_o))
    model.add(Dense(number_of_classes))
    model.add(Activation("softmax"))
    return model


if __name__ == "__main__":
    print "start:", time.strftime('%X %x %Z')
    # start = int(time.gmtime(0))

    # arguments - filename, debug, input length
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--filename", help="Provide filename for dataset you want to use. "
                                                 "It MUST be in 'media/' folder in current directory. If None is given then dataset"
                                                 "is built from NCBI database.", type=str, default="")
    parser.add_argument("-d", "--debug", action="store_true", help="If you want the enable debug mode, call program with this flag.", default=False)
    parser.add_argument("-l", "--length", help="Input length - how big chunks you want to be sequences sliced to.", default=100, type=int)
    results = parser.parse_args()
    filename = results.filename
    debug = results.debug
    read_size = results.length
    debug = True

    trX, teX, trY, teY, trteX, trteY, num_of_classes, train_class_sizes = load_datasets_from_file(filename, debug=debug, read_size=read_size)

    print(trX.shape)
    input_len = trX.shape[1]
    # trX = trX.reshape(-1, 1, 1, input_len)
    # teX = teX.reshape(-1, 1, 1, input_len)
    # trteX = trteX.reshape(-1, 1, 1, input_len)

    # params for model and cascade initialization
    conv1_stride = 4
    stride1 = 2
    downscale1 = 3
    stride2 = 2
    downscale2 = 2
    stride3 = 2
    downscale3 = 1
    conv_params = (conv1_stride, stride1, downscale1, stride2, downscale2, stride3, downscale3)

    #X = K.ftensor4()
    #Y = K.fmatrix()

    input_shape = trX.shape

    model = init_keras(input_shape, 0.2, 0.5, num_of_classes, conv_params)
    # params, X, Y, cost, updates, y_x = init_net(num_of_classes, input_len, conv_params)

    # compile train and predict function
    # train = theano.function(inputs=[X, Y], outputs=cost, updates=updates, allow_input_downcast=True)
    # predict = theano.function(inputs=[X], outputs=y_x, allow_input_downcast=True)

    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(trX, trY, batch_size=128, nb_epoch=len(trX)/128, verbose=1)
    # error here - wrong input shape
    model.evaluate(trteX, trteY, 128, verbose=1)

    model.save("model.h5")

    # model.predict(teX, 32, verbose=1)

    # epsilon = 0.005  # if evaluation score does not improve for 0,5% every 5 tries, then stop evaluating - we get best model
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

    print "stop:", time.strftime('%X %x %Z')
    #print "elapsed: ", (int(time.gmtime(0)) - start)