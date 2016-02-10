import gzip
from math import ceil
import pickle
import random
import numpy as np
import os
import sys
from load_sequences import run
from load_sequences import get_rec

datasets_dir = 'media/datasets/'


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


def seq_to_bits(vec):
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
    return [get_rec(x).seq._data for x in ids]  # nalozi sekvence


def seq_load(ntrain=50000, ntest=10000, onehot=True, seed=random.randint(0, sys.maxint)):

    seq_len = 100

    # nastavi seed
    random.seed(seed)

    # preveri ce obstaja datoteka s podatki o taxonomy (samo id-ji in labeli), ce ne obstaja pozeni run in shrani

    dir = "media"
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
        trX = pickle.load(open(dir + "/train-data-seed_%d.pkl.gz" % seed, "rb"))
        trY = pickle.load(open(dir + "/train-labels-seed_%d.pkl.gz" % seed, "rb"))
        teX = pickle.load(open(dir + "/t10k-data-seed_%d.pkl.gz" % seed, "rb"))
        teY = pickle.load(open(dir + "/t10k-labels-seed_%d.pkl.gz" % seed, "rb"))
    except IOError:  # , FileNotFoundError:
        print "train and test data not found for seed %d..." % seed
        print "generating train and test data..."
        number_of_classes = len(set(labels))
        # train_examples_per_class = int(ceil(ntrain / float(number_of_classes)))
        # test_examples_per_class = int(ceil(ntest / float(number_of_classes)))
        # popravi da se bo v trX vstavljalo zaporedoma za vsak label
        train_examples_per_class = ntrain / number_of_classes
        test_examples_per_class = ntest / number_of_classes
        examples_per_class = train_examples_per_class + test_examples_per_class

        trX = []
        trY = []
        teX = []
        teY = []

        temp_count = 0

        while temp_count < (ntrain + ntest):
            for label in set(labels):
                first = labels.index(label)
                last = len(labels) - labels[::-1].index(label)
                print "number of examples in class: %d" % (last - first)

                sum_lengths = sum(len(s) for s in data[first:last])
                print "sum lengths of genomes: %d" % sum_lengths

                vir_idx     = random.choice(range(first, last))  # nakljucno izberi virus iz razreda
                sample_idx  = random.choice(range(0, (len(data[vir_idx])) - seq_len - 1))  # nakljucno vzorci virus

                if temp_count < ntrain:
                    trX.append(seq_to_bits(data[vir_idx][sample_idx:sample_idx + seq_len]))
                    trY.append(label)
                else:
                    teX.append(seq_to_bits(data[vir_idx][sample_idx:sample_idx + seq_len]))
                    teY.append(label)

                temp_count += 1

        print "saving train and test data for seed %d..." % seed
        pickle.dump(trX, open(dir + "/train-data-seed_%d.pkl.gz" % seed, "wb"), -1)
        pickle.dump(trY, open(dir + "/train-labels-seed_%d.pkl.gz" % seed, "wb"), -1)
        pickle.dump(teX, open(dir + "/t10k-data-seed_%d.pkl.gz" % seed, "wb"), -1)
        pickle.dump(teY, open(dir + "/t10k-labels-seed_%d.pkl.gz" % seed, "wb"), -1)
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

    # data = np.load('media/data1-100.npy')
    # labels = np.load('media/labels1-100.npy')
    #
    # tr_idx = []
    # te_idx = []
    #
    # train_examples_per_class = ntrain / number_of_classes
    # test_examples_per_class = ntest / number_of_classes
    # examples_per_class = train_examples_per_class + test_examples_per_class
    #
    # labels_list = np.ndarray.tolist(labels)
    # for label in set(labels):
    #     if labels_list.count(label) < examples_per_class:
    #         print "skipping label #%d, size:%d" % (label, labels_list.count(label))
    #         continue
    #
    #     first = labels_list.index(label)
    #     last = len(labels_list) - labels_list[::-1].index(label)
    #
    #     temp_tr = np.random.choice(range(first, last), train_examples_per_class, replace=False).tolist()
    #     tr_idx += temp_tr
    #
    #     test_set = list(set(range(first, last)) - set(temp_tr))
    #     te_idx += np.random.choice(test_set, test_examples_per_class, replace=False).tolist()
    #
    # trX = data[tr_idx].astype(float)
    # trY = labels[tr_idx]
    # teX = data[te_idx].astype(float)
    # teY = labels[te_idx]

    # trX = trX[:ntrain]
    # trY = trY[:ntrain]
    #
    # teX = teX[:ntest]
    # teY = teY[:ntest]

    # if onehot:
    #     trY = one_hot(trY, number_of_classes)
    #     teY = one_hot(teY, number_of_classes)
    # else:
    #     trY = np.asarray(trY)
    #     teY = np.asarray(teY)
    #
    # return trX, teX, trY, teY


def mnist(ntrain=60000, ntest=10000, onehot=True):
    data_dir = os.path.join(datasets_dir, 'mnist/')
    fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trX = loaded[16:].reshape((60000, 28 * 28)).astype(float)

    fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    trY = loaded[8:].reshape((60000))

    fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teX = loaded[16:].reshape((10000, 28 * 28)).astype(float)

    fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    loaded = np.fromfile(file=fd, dtype=np.uint8)
    teY = loaded[8:].reshape((10000))

    trX = trX / 255.
    teX = teX / 255.

    trX = trX[:ntrain]
    trY = trY[:ntrain]

    teX = teX[:ntest]
    teY = teY[:ntest]

    if onehot:
        trY = one_hot(trY, 10)
        teY = one_hot(teY, 10)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return trX, teX, trY, teY


### not used ###

def get_labels(filename):
    l = []
    n_l = []
    i = 0
    with gzip.open(filename, 'rb') as f:
        for line in f:
            if line not in l:
                l.append(line)
                i += 1
            n_l.append(i)
    return n_l


def read_file(filename):
    temp = []
    with gzip.open(filename, 'rb') as f:
        for line in f:
            try:
                temp.append(np.array(map(float, str.split(line, '\t'))))
            except Exception as e:
                print e
                return
    return np.array(temp)


def load_obj(dir, name):
    with open(dir + '/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)