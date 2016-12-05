from collections import defaultdict
import csv
import gzip
import os
import pickle
import random
import math
from Bio import SeqIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
import numpy as np
import sys
from sklearn import cross_validation
from load_ncbi import run, load_seqs_from_ncbi, get_rec
import matplotlib.pyplot as plt
import pylab as P


def one_hot(x, n):
    """
    Get true classes (Y) and number of classes and return Y matrix in binary representation.
    From vector x, we get matrix of sizes [x.length, n].
        Example: We have 5 different classes (0, 1, 2, 3, 4) and vector x = [0, 0, 1, 2, 2, 3, 4].
                 From that we would get matrix of sizes [7, 5] and the matrix would look like:

                        [1, 0, 0, 0, 0],
                        [1, 0, 0, 0, 0],
                        [0, 1, 0, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 1, 0, 0],
                        [0, 0, 0, 1, 0],
                        [0, 0, 0, 0, 1]

    :param x: vector true classes (Y)
    :param n: number of classes
    :return: Y matrix in binary representation
    """
    if np.max(x) > n:
        raise AssertionError("USER ERROR - cannot create numpy array; number of classes must be bigger than max number of list")
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
    :param unique_nucleotides: number of unique nucleotides in loaded files - mandatory if transmission_dict is None
    :param transmission_dict: transmission dictionary - if None, build it here (every nucleotide represents one bit)
                                     e.g. if our unique nucleotides are "ATCGYM", then corresponding dictionary will be
                                          {"A": [1, 0, 0, 0, 0, 0]
                                           "T": [0, 1, 0, 0, 0, 0]
                                           "C": [0, 0, 1, 0, 0, 0]
                                           "G": [0, 0, 0, 1, 0, 0]
                                           "Y": [0, 0, 0, 0, 1, 0]
                                           "M": [0, 0, 0, 0, 0, 1]}

                              otherwise set rules beforehand and put it in dictionary;
                              e.g. if you want your prediction to be based on "ATCG" nucleotides and every other
                              nucleotide won't have much influence, then create:
                                      {"A": [1, 0, 0, 0],
                                       "T": [0, 1, 0, 0],
                                       "C": [0, 0, 1, 0],
                                       "G": [0, 0, 0, 1]}
                              with such dictionary you would set up rules for "ATCG" and every other nucleotide
                              will get value [1, 1, 1, 1] so it won't have much influence on prediction.
    :return: number representation of vec
    """

    if transmission_dict is None:
        if unique_nucleotides is None:
            raise AssertionError("USER ERROR - number of unique nucleotides and transmission dictionary not present.")
        transmission_dict = {}
        for el in unique_nucleotides:
            transmission_dict[el] = [1 if x == el else 0 for x in unique_nucleotides]
    else:
        if len(transmission_dict.keys()) != len(transmission_dict.itervalues().next()):
            print "WARNING: number of keys in transmission dictionary and length of either value aren't same!"

    bits_vector = []
    for i, c in enumerate(vec):
        if c in transmission_dict.keys():
            bits_vector += transmission_dict[c]
        else:
            bits_vector += [1 for _ in transmission_dict.keys()]
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


def load_from_file_fasta(filename, depth=4, taxonomy_el_count=-1):
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
    :param taxonomy_el_count: how many elements we want in taxonomy; -1 means whole taxonomy
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
        data, tax = load_seqs_from_ncbi(seq_len=-1, skip_read=0, overlap=0, taxonomy_el_count=taxonomy_el_count)
        # save data
        with gzip.open(filename, "w") as file:
            print "writing..."
            for oid, seq in data.iteritems():
                tax[oid] = ';'.join(tax[oid].split(";")[:depth])
                # prepare row
                row = SeqRecord(Seq(seq), id=str(oid), description=tax[oid])
                SeqIO.write(row, file, "fasta")

    return data, tax


def dataset_from_id(data, tax, ids, read_size, sample, transmission_dict):
    """
    Build dataset from given IDs list and other params.
    :param data: whole dataset dictionary
    :param tax: whole taxonomy dictionary
    :param ids: IDs for building dataset of them
    :param read_size: size of reads you want to generate
    :param sample: how many data you want to skip - 20% means that every fifth read will be included
    :param transmission_dict: dictionary of transmission
    :return: build dataset with numeric sequences and classes
    """
    tempX = []
    tempY = []
    for te_id in ids:
        seq = data[te_id]
        while seq:
            if len(seq) < read_size:
                break
            tempX.append(seq_to_bits(seq[:read_size], transmission_dict=transmission_dict))
            tempY.append(tax[te_id])
            # don't use whole sequence, only every second, third etc (depending on sample percent) - use ceil to avoid decimals
            seq = seq[int(math.ceil(read_size / sample)):]
    return tempX, tempY


def load_dataset(filename):
    """
    Return object from filename. Filename must include directory.
    :param filename: dir + filename
    :return: object
    """
    with gzip.open(filename, "rb") as f:
        return pickle.load(f)


def save_dataset(filename, obj):
    """
    Save dataset to given filename. Filename must include directory.
    :param filename: dir + filename
    :param obj: object you want to save
    :return: None
    """
    with gzip.open(filename, "wb") as f:
        pickle.dump(obj, f)
    print "Successfully saved as: " + filename


def load_data(filename, test=0.2, transmission_dict=None, depth=4, sample=0.2, read_size=100, onehot=True, seed=random.randint(0, sys.maxint), taxonomy_el_count=-1):
    """
    Main function for loading data. We expect, that fasta files with data are in media directory - if the file with
    given filename does not exist, we build a new one from NCBI database.
    Then we build numeric taxonomy representation.
    With sklearn function LabelShuffleSplit we split whole dataset into train and test set. When built, test and train
    datasets are saved for later use.
    We want to evaluate our model only with train data, so we split train data into two parts
    (with LabelShuffleSplit function) by 80-20. Files tr_ (trX and trY) are actually trtr_ (train from train - so we
    are going to train our model ONLY on this data). With files trte_ we are going to evaluate our model for saving,
    and with te_ files we are going to actually predict final classes.
    The params helps you set some rules for building the data you want.
    :param filename: dataset filename (which MUST be in media directory, otherwise new will be built)
    :param test: test size in percentage of whole dataset
    :param transmission_dict: dictionary for transforming nucleotides to bits (seq_to_bits function)
    :param depth: taxonomy tree depth
    :param sample: sampling size - 20% means that every fifth read is included into dataset
    :param read_size: chunk size
    :param onehot: binary representation of true classes
    :param seed: random seed for replicating experiments
    :param taxonomy_el_count: how many elements we want in taxonomy; -1 means whole taxonomy
    :return: train and test datasets as numpy arrays
    """

    assert test < 1.0 and sample < 1.0
    dir = "media/"

    # load data from fasta file - we need it here because of num_of_classes - we only can get this from labels/data dict
    data, labels = load_from_file_fasta(dir + filename, depth=depth, taxonomy_el_count=taxonomy_el_count)

    # numeric representation of classes and calculating sum of sequence lengths for each class
    temp_l = []
    label_num = -1
    tax = {}
    class_size = defaultdict(int)
    for id, l in labels.iteritems():
        if l not in temp_l:
            temp_l.append(l)
            label_num += 1
        tax[id] = label_num
        print label_num, len(data[id])
        class_size[label_num] += len(data[id])

    for _class, sum in class_size.iteritems():
        class_size[_class] = sum / tax.values().count(_class)

    try:
        # we save files as something.fasta.gz, so we try to open those files - if they don't exist, generate new ones
        trX = load_dataset(dir + filename[:filename.index(".fasta.gz")] + "-trX.fasta.gz")
        trY = load_dataset(dir + filename[:filename.index(".fasta.gz")] + "-trY.fasta.gz")
        teX = load_dataset(dir + filename[:filename.index(".fasta.gz")] + "-teX.fasta.gz")
        teY = load_dataset(dir + filename[:filename.index(".fasta.gz")] + "-teY.fasta.gz")
        trteX = load_dataset(dir + filename[:filename.index(".fasta.gz")] + "-trteX.fasta.gz")
        trteY = load_dataset(dir + filename[:filename.index(".fasta.gz")] + "-trteY.fasta.gz")
    except IOError:
        # data, labels = load_from_file_fasta(dir + filename, depth=depth, taxonomy_el_count=taxonomy_el_count)

        # taxonomy annotations to numeric representation (from 0 to num_of_classes)
        # temp_l = []
        # label_num = -1
        # tax = {}
        # for id, l in labels.iteritems():
        #     if l not in temp_l:
        #         temp_l.append(l)
        #         label_num += 1
        #     tax[id] = label_num

        trX = []
        trY = []
        teX = []
        teY = []
        trteX = []
        trteY = []

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

            # get train test IDs for evaluating model
            ss_tr = cross_validation.LabelShuffleSplit(tr_ids, n_iter=1, test_size=0.2, random_state=seed)
            for train_train_index, train_test_index in ss_tr:
                trtr_ids = list(tr_ids[i] for i in train_train_index)
                trte_ids = list(tr_ids[i] for i in train_test_index)

            # use only "sample" percent of data
            for tr_id in tr_ids:
                seq = data[tr_id]
                while seq:
                    if len(seq) < read_size:
                        break
                    if tr_id in trte_ids:
                        trteX.append(seq_to_bits(seq[:read_size], transmission_dict=transmission_dict))
                        trteY.append(tax[tr_id])
                    elif tr_id in trtr_ids:
                        trX.append(seq_to_bits(seq[:read_size], transmission_dict=transmission_dict))
                        trY.append(tax[tr_id])
                    else:
                        print tr_id + " is not in train_index list!"
                        sys.exit(0)
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

            tmpteX, tmpteY = dataset_from_id(data, tax, te_ids, read_size, sample, transmission_dict)
            tmptrX, tmptrY = dataset_from_id(data, tax, trtr_ids, read_size, sample, transmission_dict)
            tmptrteX, tmptrteY = dataset_from_id(data, tax, trte_ids, read_size, sample, transmission_dict)

            assert teX == tmpteX and teY == tmpteY
            assert trX == tmptrX and trY == tmptrY
            assert trteX == tmptrteX and trteY == tmptrteY

            # set filenames for saving like filename-trX, filename-teX...
            save_dataset(dir + filename[:filename.index(".fasta.gz")] + "-trX.fasta.gz", trX)
            save_dataset(dir + filename[:filename.index(".fasta.gz")] + "-trY.fasta.gz", trY)
            save_dataset(dir + filename[:filename.index(".fasta.gz")] + "-teX.fasta.gz", teX)
            save_dataset(dir + filename[:filename.index(".fasta.gz")] + "-teY.fasta.gz", teY)
            save_dataset(dir + filename[:filename.index(".fasta.gz")] + "-trteX.fasta.gz", trteX)
            save_dataset(dir + filename[:filename.index(".fasta.gz")] + "-trteY.fasta.gz", trteY)

    number_of_classes = len(class_size.keys())

    if onehot:
        trY = one_hot(trY, number_of_classes)
        teY = one_hot(teY, number_of_classes)
        trteY = one_hot(trteY, number_of_classes)
    else:
        trY = np.asarray(trY)
        teY = np.asarray(teY)
        trteY = np.asarray(trteY)

    return np.asarray(trX), np.asarray(teX), np.asarray(trY), np.asarray(teY), np.asarray(trteX), np.asarray(trteY), \
           number_of_classes, class_size


#### DEPRECATED - used for csv ####

def load_seqs(ids):
    """
    Get list of genome ids and return list of genome sequences.
    :param ids: list of genome ids
    :return: list of genome sequences
    """
    return [get_rec(x).seq._data for x in ids]


def seq_load(ntrain=50000, ntest=10000, onehot=True, seed=random.randint(0, sys.maxint), thresh=0.1, transmission_dict=None, save=False):
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
        if save:
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
                    trX.append(seq_to_bits(data[vir_idx][sample_idx:sample_idx + seq_len], transmission_dict=transmission_dict))
                    trY.append(label)
                else:
                    if temp_count - ntrain >= ntest:
                        break
                    teX.append(seq_to_bits(data[vir_idx][sample_idx:sample_idx + seq_len], transmission_dict=transmission_dict))
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

        if save:
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
    transmission_dict = {'A': [1, 0, 0, 0], 'T': [0, 1, 0, 0], 'C': [0, 0, 1, 0], 'G': [0, 0, 0, 1]}
    data, tax = load_from_file_fasta("blabla-test.fasta.gz", depth=4, taxonomy_el_count=10)

    format_str = "{:11}\t{:80}\t{:7}"

    print format_str.format("genome_id", "classification", "sequence_length")
    for key, val in tax.iteritems():
        print format_str.format(key, val, len(data[key]))
