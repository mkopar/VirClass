import os
import pickle
import random
import math
import numpy as np
import sys
from load_sequences import run
from load_sequences import get_rec
import matplotlib.pyplot as plt
import pylab as P


def one_hot(x, n):
    """
    Get true classes (Y) and number of classes and return Y matrix in binary representation.
    :param x: true classes (Y)
    :param n: number of classes
    :return: Y matrix in binary representation
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def seq_to_bits(vec):
    """
    Get sequence and transform it into number representation.
    :param vec: sequence
    :return: number representation
    """
    bits_vector = []
    for i, c in enumerate(vec):
        if c == "A":
            bits_vector += [1, 0, 0, 0]
        elif c == "T":
            bits_vector += [0, 1, 0, 0]
        elif c == "C":
            bits_vector += [0, 0, 1, 0]
        elif c == "G":
            bits_vector += [0, 0, 0, 1]
        else:
            bits_vector += [1, 1, 1, 1]
    return bits_vector

    # for i, c in enumerate(vec):
    #     if c == "A":
    #         bits_vector.append(0)
    #     elif c == "T":
    #         bits_vector.append(0.33)
    #     elif c == "C":
    #         bits_vector.append(0.66)
    #     elif c == "G":
    #         bits_vector.append(1)
    #     else:
    #         bits_vector.append(0.5)
    # return bits_vector


def load_seqs(ids):
    """
    Get list of genome ids and return list of genome sequences.
    :param ids: list of genome ids
    :return: list of genome sequences
    """
    return [get_rec(x).seq._data for x in ids]


def histogram(values, name):
    """Draw histogram for given values and save it with given name.
    :param values: values to show in histogram
    :param name: name for saving figure
    """
    print max(values)
    bins = np.arange(0, 5000000, 250000)
    P.hist(values, bins, histtype='bar', rwidth=0.8, log=True)
    plt.tight_layout()
    ylims = P.ylim()
    P.ylim((0.1, ylims[1]))
    plt.savefig(name)
    plt.clf()


def seq_load(ntrain=50000, ntest=10000, onehot=True, seed=random.randint(0, sys.maxint), thresh=0.1):
    """
    In this method we want to simulate sequencing. We create samples in length of seq_len (in our case 100).

    We load data from cached files data-ids.pkl.gz and labels.pkl.gz. When we first run it, those files won't exist
    so script executes imported method 'run' from file load_sequences.py and generates data (genome ids) and
    labels (corresponding class ids). We cache those two variables for future executions of script.

    When we have data (genome ids) and labels we load sequences for every genome id.

    We are caching train and test datasets for every seed. If train and test files does not exist, we build it.
    First we calculate the threshold with selected formula:
            thresh * examples_per_class * seq_len
    All classes, which sum of genome lengths is smaller than threshold are skipped.

    When we get labels which are big enough, we start building train and test datasets.
    We randomly choose virus from a class. After that we choose randomly sample virus sequence.
    Then we transform every read into numerical values with function seq_to_bits.
    Nucleotides are being transformed in the following pattern:

            A = [1, 0, 0, 0]
            T = [0, 1, 0, 0]
            C = [0, 0, 1, 0]
            G = [0, 0, 0, 1]
            _ = [1, 1, 1, 1]

    Datasets are being built until we reach desired dataset size. That means that some classes might be smaller than
    others for 1 example. Save built datasets. Filename is in following format:

            train-data-seed_1234-onehot_1-threshold_30.pkl.gz

    This means that file represents train data for seed 1234. Onehot flag is true and our threshold (parameter thresh)
    is set to 0.3.

    Check if onehot flag is true and appropriately change class vector (Y).

    Return train and test datasets in numpy arrays.

    :param ntrain: train size
    :param ntest: test size
    :param onehot: binary representation of true classes
    :param seed: random seed
    :param thresh: threshold
    :return: train and test datasets as numpy arrays
    """

    # check if thresh parameter is valid
    if thresh < 0.0 or thresh > 1.0:
        raise ValueError("Parameter thresh must be set between 0.0 and 1.0.")


    seq_len = 100

    random.seed(seed)

    dir = "media"
    if not os.path.isdir(dir):
        os.makedirs(dir)

    try:
        print "loading data and labels..."
        data = pickle.load(open(dir + "/data-ids.pkl.gz", "rb"))
        labels = pickle.load(open(dir + "/labels.pkl.gz", "rb"))
    except IOError:
        print "data and labels not found...\ngenerating data and labels..."
        data, labels = run()
        print "saving data and labels..."
        pickle.dump(data, open(dir + "/data-ids.pkl.gz", "wb"), -1)
        pickle.dump(labels, open(dir + "/labels.pkl.gz", "wb"), -1)

    print "getting sequences..."
    data = load_seqs(data)

    try:
        print "loading train and test data for seed %d..." % seed
        trX = pickle.load(open(dir + "/tr%d-data-seed_%d-onehot_%d-threshold_%d.pkl.gz" % (ntrain, seed, int(onehot), thresh * 100), "rb"))
        trY = pickle.load(open(dir + "/tr%d-labels-seed_%d-onehot_%d-threshold_%d.pkl.gz" % (ntrain, seed, int(onehot), thresh * 100), "rb"))
        teX = pickle.load(open(dir + "/te%d-data-seed_%d-onehot_%d-threshold_%d.pkl.gz" % (ntest, seed, int(onehot), thresh * 100), "rb"))
        teY = pickle.load(open(dir + "/te%d-labels-seed_%d-onehot_%d-threshold_%d.pkl.gz" % (ntest, seed, int(onehot), thresh * 100), "rb"))
    except IOError:  # , FileNotFoundError:
        print "train and test data not found for seed %d..." % seed
        print "generating train and test data..."
        number_of_classes = len(set(labels))
        # train_examples_per_class = int(ceil(ntrain / float(number_of_classes)))
        # test_examples_per_class = int(ceil(ntest / float(number_of_classes)))
        train_examples_per_class = int(math.ceil(ntrain / float(number_of_classes)))
        test_examples_per_class = int(math.ceil(ntest / float(number_of_classes)))
        examples_per_class = train_examples_per_class + test_examples_per_class

        trX = []
        trY = []
        teX = []
        teY = []

        labels_to_process = []
        examples_in_class = []
        labels_lengths = []
        smaller = []

        for label in set(labels):
            first = labels.index(label)
            last = len(labels) - labels[::-1].index(label)
            print "number of examples in class: %d" % (last - first)

            sum_lengths = sum(len(s) for s in data[first:last])
            print "sum lengths of genomes: %d" % sum_lengths

            examples_in_class.append((last-first, sum_lengths))
            labels_lengths.append((sum_lengths, last-first))

            threshold = thresh * examples_per_class * seq_len
            if sum_lengths > thresh * (examples_per_class * seq_len):
                labels_to_process.append((label, first, last))
            else:
                smaller.append(label)

        print "labels which sum of genome lengths are smaller than %d:" % threshold, smaller

        number_of_classes = len(labels_to_process)
        train_examples_per_class = int(math.ceil(ntrain / float(number_of_classes)))
        test_examples_per_class = int(math.ceil(ntest / float(number_of_classes)))
        examples_per_class = train_examples_per_class + test_examples_per_class

        print "number of classes: %d, examples per class: %d" % (number_of_classes, examples_per_class)

        # histogram(labels_lengths, "class_genome_lengths.png")
        # print sorted(examples_in_class)
        # print sorted(labels_lengths)
        # return 0

        temp_count = 0

        prvi = False

        while temp_count < (ntrain + ntest):
            for label, first, last in labels_to_process:

                vir_idx     = random.choice(range(first, last))  # randomly choose virus from class
                sample_idx  = random.choice(range(0, (len(data[vir_idx])) - seq_len - 1))  # randomly sample virus

                if temp_count < ntrain:
                    trX.append(seq_to_bits(data[vir_idx][sample_idx:sample_idx + seq_len]))
                    trY.append(label)
                else:
                    if not prvi:
                        print label
                        prvi = True
                    if temp_count - ntrain >= ntest:
                        break
                    teX.append(seq_to_bits(data[vir_idx][sample_idx:sample_idx + seq_len]))
                    teY.append(label)

                temp_count += 1

        """
            Example:
                ntrain  = 100000
                ntest   = 20000

                after threshold filter, there are 95 classes left

                train_examples_per_class = 100000 / 95 = 1052.63 = 1053
                test_examples_per_class  = 20000  / 95 = 210.53 = 211

                After random sampling, there will be only first n classes represented with 1053 examples per class,
                others will have 1052.
                So if we ceil numbers, we need to check it with first class, if we don't ceil it, we need to check it
                with last class in trY.

                As for the test data, there will be a set of data where we have 210 examples per class, others
                will have 211. Sampling mechanism starts filling test data set when we completely fill train data set.
                Test examples starts from class n and stops when the test data set is filled out.
        """

        assert train_examples_per_class == trY.count(labels_to_process[0][0])
        assert train_examples_per_class - 1 == trY.count(labels_to_process[-1][0])
        assert test_examples_per_class == teY.count(labels_to_process[-1][0])

        print "saving train and test data for seed %d..." % seed
        pickle.dump(trX, open(dir + "/tr%d-data-seed_%d-onehot_%d-threshold_%d.pkl.gz" % (ntrain, seed, int(onehot), thresh * 100), "wb"), -1)
        pickle.dump(trY, open(dir + "/tr%d-labels-seed_%d-onehot_%d-threshold_%d.pkl.gz" % (ntrain, seed, int(onehot), thresh * 100), "wb"), -1)
        pickle.dump(teX, open(dir + "/te%d-data-seed_%d-onehot_%d-threshold_%d.pkl.gz" % (ntest, seed, int(onehot), thresh * 100), "wb"), -1)
        pickle.dump(teY, open(dir + "/te%d-labels-seed_%d-onehot_%d-threshold_%d.pkl.gz" % (ntest, seed, int(onehot), thresh * 100), "wb"), -1)
        print "saving done"

    number_of_classes = len(set(labels))

    print len(trX)
    print len(teX)

    if onehot:
        trY = one_hot(trY, number_of_classes)
        teY = one_hot(teY, number_of_classes)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return np.asarray(trX), np.asarray(teX), np.asarray(trY), np.asarray(teY), number_of_classes