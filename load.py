import gzip
import numpy as np
import os

datasets_dir = 'media/datasets/'


def one_hot(x, n):
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


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


def get_data(filename):
    temp = []
    with gzip.open(filename, 'rb') as f:
        for line in f:
            try:
                temp.append(np.array(map(float, str.split(line, '\t'))))
            except Exception as e:
                c = str.split(str(e), ": ")[-1]
                temp.append(np.array(map(float, str.split(line.replace(c, "-1.0"), '\t'))))
    return np.array(temp)


def seq_load(ntrain=50000, ntest=10000):
    data = np.load('media/data1-100.npy')
    labels = np.load('media/labels1-100.npy')
    number_of_classes = 104

    tr_idx = []
    te_idx = []
    train_examples_per_class = ntrain / number_of_classes
    test_examples_per_class = ntest / number_of_classes
    examples_per_class = train_examples_per_class + test_examples_per_class

    labels_list = np.ndarray.tolist(labels)
    for label in set(labels):
        if labels_list.count(label) < examples_per_class:
            print "skipping label #%d, size:%d" % (label, labels_list.count(label))
            continue

        first = labels_list.index(label)
        last = len(labels_list) - labels_list[::-1].index(label)

        tr_idx += np.random.choice(range(first, last), train_examples_per_class, replace=False).tolist()
        te_idx += np.random.choice(range(first, last), test_examples_per_class, replace=False).tolist()

    trX = data[tr_idx].astype(float)
    trY = labels[tr_idx]
    teX = data[te_idx].astype(float)
    teY = data[te_idx]

    return trX, teX, trY, teY


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