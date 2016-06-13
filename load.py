from collections import defaultdict
import csv
import gzip
import os
import pickle
import random
import math
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqIO import FastaIO
from Bio.SeqRecord import SeqRecord
import numpy as np
import sys
from sklearn import cross_validation
from load_ncbi import run, load_seqs_from_ncbi, get_rec, check_hash
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


def seq_to_bits(vec, unique_nucleotides=None, transmission_dict=None):
    """
    Get sequence and transform it into number representation. Given parameters set some rules in representation.
    :param vec: sequence to transform
    :param unique_nucleotides: number of unique nucleotides in loaded files
    :param transmission_dict: transmission dictionary - if None, build it here (every nucleotide represents one bit)
    :return: number representation of vec
    """

    if transmission_dict is None:
        if unique_nucleotides is None:
            print "Problems - number of unique nucleotides and transmission dictionary not present. Exiting now."
            sys.exit(0)
        transmission_dict = {}
        for el in unique_nucleotides:
            transmission_dict[el] = [1 if x == el else 0 for x in unique_nucleotides]

    bits_vector = []
    for i, c in enumerate(vec):
        if c in transmission_dict.keys():
            bits_vector += transmission_dict[c]
        else:
            bits_vector += [1 for _ in transmission_dict.keys()]
        # if c == "A":
        #     bits_vector += [1, 0, 0, 0]
        # elif c == "T":
        #     bits_vector += [0, 1, 0, 0]
        # elif c == "C":
        #     bits_vector += [0, 0, 1, 0]
        # elif c == "G":
        #     bits_vector += [0, 0, 0, 1]
        # else:
        #     bits_vector += [1, 1, 1, 1]
    return bits_vector


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


def load_from_file_fasta(filename, depth=4):
    """
    Load data from filename. Default value for depth is 4 - this is depth in classification.

        For example, if we have:
            Viruses;ssRNA viruses;ssRNA negative-strand viruses;Mononegavirales;Rhabdoviridae;Ephemerovirus

        we want to strip it to
            Viruses;ssRNA viruses;ssRNA negative-strand viruses;Mononegavirales

        which is exactly of depth 4.

    Files are saved in FASTA format.
    Each read will start with genome ID, continued with classification in header and whole genome sequence.

    Example:

        >10086561 Viruses;ssRNA viruses;ssRNA negative-strand viruses;Mononegavirales
        ACGAGAAAAAACAAAAAAACTAATTGATATTGATACCCAATTAGTGTTTCAACAGGTCTC...
        >1007626122 Viruses;ssRNA viruses;ssRNA positive-strand viruses, no DNA stage;Caliciviridae
        GTGAATGAAGATGGCGTCTAACGACGCTAGTGTTGCCAACAGCAACAGCAAAACCATTGC...


    :param filename: filename of fasta file
    :param depth: default value 4, how deep into classification we want to train
    :return: data and tax - data represents sequences, taxonomy represents classification (for each genome ID)
    """

    data = defaultdict(list)
    tax = {}

    try:
        assert os.path.isfile(filename)
        with gzip.open(filename, "r") as file:
            # read data
            print "reading..."
            for seq_record in SeqIO.parse(file, "fasta"):
                oid = seq_record.id
                classification = seq_record.description.split(oid)[1].strip()
                seq = str(seq_record.seq)
                data[oid] = seq
                tax[oid] = classification
    except AssertionError:
        data, tax = load_seqs_from_ncbi(seq_len=-1, skip_read=0, overlap=0)
        # save data
        with gzip.open(filename, "w") as file:
            print "writing..."
            for oid, seq in data.iteritems():
                tax[oid] = ';'.join(tax[oid].split(";")[:depth])
                # prepare row
                row = SeqRecord(Seq(seq), id=str(oid), description=tax[oid])
                SeqIO.write(row, file, "fasta")

    return data, tax


def load_data(filename, test=0.2, transmission_dict=None, depth=4, sample=0.2, read_size=100, onehot=True, seed=random.randint(0, sys.maxint)):

    assert test < 1.0 and sample < 1.0
    dir = "media/"

    data, labels = load_from_file_fasta(dir + filename, depth=depth)

    temp_l = []
    label_num = -1
    tax = {}
    for id, l in labels.iteritems():
        if l not in temp_l:
            temp_l.append(l)
            label_num += 1
        tax[id] = label_num

    trX = []
    trY = []
    teX = []
    teY = []

    # keys must be same
    assert data.keys() == labels.keys()
    oids = [x for x in labels.keys()]
    number_of_classes = len(data.keys())

    ss = cross_validation.LabelShuffleSplit(oids, n_iter=1, test_size=test, random_state=seed)
    for train_index, test_index in ss:
        # we split ids to train and test
        tr_ids = list(oids[i] for i in train_index)
        te_ids = list(oids[i] for i in test_index)

        # intersection of train and test must be empty set
        assert(set(tr_ids).intersection(set(te_ids)) == set())

        # use only "sample" percent of data?

        for tr_id in tr_ids:
            seq = data[tr_id]
            while seq:
                if len(seq) < read_size:
                    break
                trX.append(seq_to_bits(seq[:read_size], transmission_dict=transmission_dict))
                trY.append(tax[tr_id])
                # don't use whole sequence, only every second, third etc (depending on sample percent) - use ceil to avoid decimals
                seq = seq[int(math.ceil(read_size / sample)):]

        for te_id in te_ids:
            seq = data[te_id]
            while seq:
                if len(seq) < read_size:
                    break
                teX.append(seq_to_bits(seq[:read_size], transmission_dict=transmission_dict))
                teY.append(tax[te_id])
                # don't use whole sequence, only every second, third etc (depending on sample percent) - use ceil to avoid decimals
                seq = seq[int(math.ceil(read_size / sample)):]

    if onehot:
        trY = one_hot(trY, number_of_classes)
        teY = one_hot(teY, number_of_classes)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)

    return np.asarray(trX), np.asarray(teX), np.asarray(trY), np.asarray(teY), number_of_classes


#### DEPRECATED - used for csv ####

def load_seqs(ids):
    """
    Get list of genome ids and return list of genome sequences.
    :param ids: list of genome ids
    :return: list of genome sequences
    """
    return [get_rec(x).seq._data for x in ids]


def seq_load(ntrain=50000, ntest=10000, onehot=True, seed=random.randint(0, sys.maxint), thresh=0.1):
    """
    In this method we want to simulate sequencing. We create samples in length of seq_len (in our case 100).

    We load data from cached files data-ids.pkl.gz and labels.pkl.gz. When we first run it, those files won't exist
    so script executes imported method 'run' from file load_ncbi.py and generates data (genome ids) and
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

            examples_in_class.append((last - first, sum_lengths))
            labels_lengths.append((sum_lengths, last - first))

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

        while temp_count < (ntrain + ntest):
            for label, first, last in labels_to_process:

                vir_idx = random.choice(range(first, last))  # randomly choose virus from class
                sample_idx = random.choice(range(0, (len(data[vir_idx])) - seq_len - 1))  # randomly sample virus

                if temp_count < ntrain:
                    trX.append(seq_to_bits(data[vir_idx][sample_idx:sample_idx + seq_len]))
                    trY.append(label)
                else:
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
        pickle.dump(trX, open(
            dir + "/tr%d-data-seed_%d-onehot_%d-threshold_%d.pkl.gz" % (ntrain, seed, int(onehot), thresh * 100), "wb"),
                    -1)
        pickle.dump(trY, open(
            dir + "/tr%d-labels-seed_%d-onehot_%d-threshold_%d.pkl.gz" % (ntrain, seed, int(onehot), thresh * 100),
            "wb"), -1)
        pickle.dump(teX, open(
            dir + "/te%d-data-seed_%d-onehot_%d-threshold_%d.pkl.gz" % (ntest, seed, int(onehot), thresh * 100), "wb"),
                    -1)
        pickle.dump(teY, open(
            dir + "/te%d-labels-seed_%d-onehot_%d-threshold_%d.pkl.gz" % (ntest, seed, int(onehot), thresh * 100),
            "wb"), -1)
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


def load_from_file(filename):
    """
    DEPRECATED
    Open tab delimited file with gzip. If file does not exists, create a new one.
    This is a csv format of data we want for input.
    Each row consists as follow:

        GID\tSEQUENCE(length 100)\tCLASSIFICATION

    GID is genome ID, SEQUENCE is our read (of length 100) and CLASSIFICATION is taxonomy for specific genome.

    :param filename:
    :return:
    """

    data = defaultdict(list)
    tax = {}
    unique_nucleotides = ''

    try:
        with gzip.open(filename, 'rb') as file:
            reader = csv.reader(file, delimiter='\t', quotechar='\'')
            for oid, seq, classification in reader:
                oid = int(oid)
                unique_nucleotides += ''.join(set(seq))
                unique_nucleotides = ''.join(set(unique_nucleotides))
                #data.append((oid, seq))
                data[oid].append(seq)
                tax[oid] = classification
    except IOError:
        data, tax = load_seqs_from_ncbi(seq_len=100, skip_read=0, overlap=50)
        # save data
        with gzip.open(filename, 'wb') as file:
            csv_file = csv.writer(file, delimiter='\t', quotechar='\'')
            for oid, seq in data.iteritems():
                oid = int(oid)
                unique_nucleotides += list({''.join([x for x in seq])})
                taxonomy_part = tax[oid]
                # prepare row
                row = [oid, seq, taxonomy_part]
                csv_file.writerow(row)

    return data, tax, unique_nucleotides


if __name__ == "__main__":
    data, tax = load_from_file_fasta("test-0.20-0-4-0.20-100-onehot.fasta.gz", depth=4)

    format_str = "{:11}\t{:80}\t{:7}"

    print format_str.format("genome_id", "classification", "sequence_length")
    for key, val in tax.iteritems():
        print format_str.format(key, val, len(data[key]))
