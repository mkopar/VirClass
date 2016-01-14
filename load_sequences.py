from itertools import product
import numpy as np
import matplotlib as mpl
from sklearn import preprocessing

mpl.use('Agg')
import matplotlib.pyplot as plt
import pylab as P

__author__ = 'Matej'

import os
import pickle
from Bio import Entrez
from Bio import SeqIO
from collections import defaultdict

dir = "../Diploma/cache"
if not os.path.isdir(dir):
    os.makedirs(dir)

# all virus sequences
# term = "Viruses[Organism] NOT srcdb_refseq[PROP] NOT cellular organisms[ORGN] AND nuccore genome samespecies[Filter] NOT nuccore genome[filter] NOT gbdiv syn[prop]"
# only reference (refSEQ) virues sequences
# see distinction between the two, here:
# http://www.ncbi.nlm.nih.gov/genomes/GenomesHome.cgi?taxid=10239&hopt=faq

term = "Viruses[Organism] AND srcdb_refseq[PROP] AND complete_genome"
handle = Entrez.esearch(db="nucleotide", term=term, retmax=100000)
record = Entrez.read(handle)
id_list = sorted(set(record["IdList"]))
print(record["Count"], len(record["IdList"]), len(id_list))


def get_rec(rec_ID):
    """Get record for given ID."""
    try:
        rec = pickle.load(open(dir + "/%s.pkl.gz" % rec_ID, "rb"))
    except IOError:  # , FileNotFoundError:
        print("downloading sequence id:", rec_ID)
        handle = Entrez.efetch(db="nucleotide", rettype="gb", id=rec_ID, email="a@uni-lj.si")
        rec = SeqIO.read(handle, "gb")
        handle.close()
        pickle.dump(rec, open(dir + "/%s.pkl.gz" % rec_ID, "wb"), -1)
        print("genome size:", len(rec.seq), rec.seq[:20] + "...")
        print("Taxonomy:", rec.annotations['taxonomy'])
        for a, t in rec.annotations.items():
            print("  %s: %s" % (a, str(t)[:15]))
        print()
    return rec


def rec_dd():
    return defaultdict(rec_dd)


def update_taxonomy(taxonomy, tax_path, seq_record):
    """Create dictionary with taxonomy name and IDs of sequences which belongs to specific taoxnomy."""
    if len(tax_path) == 0:
        return taxonomy

    tax = tax_path[0].lower()
    if tax in taxonomy:
        taxonomy[tax]["data"].append(seq_record.annotations["gi"])
        update_taxonomy(taxonomy[tax], tax_path[1:], seq_record)
    else:
        taxonomy[tax] = dict({"data": list({seq_record.annotations["gi"]})})
        temp = update_taxonomy(taxonomy[tax], tax_path[1:], seq_record)
        if len(temp) > 1:  # 1 = data, 2 = data + key
            taxonomy = temp
    return taxonomy


def check_taxonomy_filter(taxonomy_name, to_filter):
    in_to_filter = False
    for temp_tax in taxonomy_name:
        temp_tax = temp_tax.lower().split()
        for temp_tax_el in temp_tax:
            if temp_tax_el in to_filter:
                in_to_filter = True
                print "filtering taxonomy in ", rec.annotations["taxonomy"]
    return in_to_filter


def print_nice(taxonomy, level=0):
    for i in sorted(taxonomy.keys()):
        if i == "data":
            if len(taxonomy) == 1:
                return
            else:
                continue
        else:
            print level * "\t", i.replace("->", "", 1), len(taxonomy[i]["data"])
            print_nice(taxonomy[i], level + 1)


def filter_taxonomy(taxonomy, filter_list):
    """Remove bacteria, unclassified, ... nodes."""

    if type(taxonomy) is list:
        return taxonomy

    for i in [x for x in taxonomy.keys() if x != "data"]:

        # filter
        if i.split(" ")[0] in filter_list:
            ignored.append(("filter " + i, len(taxonomy[i]["data"])))
            taxonomy.pop(i)
            continue

        filter_taxonomy(taxonomy[i], filter_list)

    return taxonomy


def count_list_nodes(taxonomy):
    count = 0
    keys = [x for x in taxonomy.keys() if x != "data"]
    for i in keys:
        if set(taxonomy[i]) == set(list({"data"})):
            if i == keys[-1]:
                count += 1
                return count
            else:
                count += 1
        else:
            count += count_list_nodes(taxonomy[i])
    return count


def count_examples(taxonomy):
    count = 0
    keys = [x for x in taxonomy.keys() if x != "data"]
    for i in keys:
        if set(taxonomy[i]) == set(list({"data"})):
            if i == keys[-1]:
                count += len(taxonomy[i]["data"])
                return count
            else:
                count += len(taxonomy[i]["data"])
        else:
            count += count_examples(taxonomy[i])
    return count


def get_list_nodes(taxonomy, parent):
    # preverjeno na roke in dela
    list_nodes = list()
    keys = [x for x in taxonomy.keys() if x != "data"]
    for i in keys:
        if set(taxonomy[i]) == set(list({"data"})):
            #list_nodes.append((i, parent, taxonomy[i]))
            if i == keys[-1]:
                list_nodes.append((i, parent, taxonomy[i]))
                return list_nodes
            else:
                list_nodes.append((i, parent, taxonomy[i]))
        else:
            list_nodes += get_list_nodes(taxonomy[i], parent + "->" + i)
    return list_nodes


def get_all_nodes(taxonomy, parent=""):
    """
    :param taxonomy:
    :return: all nodes (including list nodes)
    """
    all_nodes = list()
    keys = [x for x in taxonomy.keys() if x != "data"]
    for i in keys:
        # if we want all non-list nodes, than this stays, otherwise comment this
        # if len([x for x in taxonomy[i].keys() if x != "data"]) == 0:
        # continue
        if i == "rest":
            all_nodes.append(parent + "->" + i)
        else:
            all_nodes.append(i)
        all_nodes += get_all_nodes(taxonomy[i], i)
    return all_nodes


def get_path_row(node, path_attributes):
    path = node.split("->")[1:]  # because parent starts like that "->viruses->dsdna..." and first el is always empty
    # if rest in path, we need to prepare different path list
    if "rest" in path:
        # rest could be only list node - if not, raise an error
        if "rest" != path[-1]:
            ValueError("rest is not list node. List node: ", path[-1])

        # rest is list node, so we need to merge last two elements of path into last
        path[-1] = path[-2] + "->" + path[-1]

    #vector2 = MultiLabelBinarizer(classes=path_attributes).fit_transform([path, path_attributes])[0]
    # mlb is not working if i pass classes parameter with length 178

    #vector = np.zeros(len(path_attributes), dtype=np.float32)

    #c = 0
    vector = np.array([1 if attr in path else 0
                       for attr in path_attributes],
                      dtype=np.float32)
    #for attr in path_attributes:
    #    if attr in path:
    #        vector[c] = 1
    #    c += 1
    if sum(vector) != len(path):
        ValueError("problems in get_path_row...")

    return vector


def draw_length_histogram(list_nodes, in_a_row):
    list_lengths = list()
    for list_el in list_nodes:
        for list_id in list_el[2]["data"]:
            list_temp = get_rec(list_id).seq
            list_lengths.append(len(list_temp))
    print sorted(set(list_lengths))
    print max(list_lengths)
    bins = np.arange(0, 1400000, 100000)
    n, bins, patches = P.hist(list_lengths, bins, histtype='bar', rwidth=0.8, log=True)
    plt.axvline(200000, color='r', linestyle='dashed', linewidth=1)
    plt.xlabel("lengths")
    plt.tight_layout()
    ylims = P.ylim()
    P.ylim((0.1, ylims[1]))

    plt.savefig("lenghts_histogram"+str(in_a_row)+".png")


if __name__ == "__main__":
    # call: python get_viral_sequence.py>log.out 2>log.err

    # pobrisi vse ki nimajo naslednika niti brata

    taxonomy = rec_dd()
    i = 0
    for genome_id in id_list:
        try:
            rec = get_rec(genome_id)
            update_taxonomy(taxonomy, rec.annotations["taxonomy"], rec)
        except Exception as e:
            print("problems...")
            print e

    print_nice(taxonomy)

    print "no of examples after taxonomy was built: %d" % count_examples(taxonomy)
    print "no of list nodes after taxonomy was built: %d" % count_list_nodes(taxonomy)

    print "filtering bacteria, unclassified, unassigned..."
    filter_taxonomy(taxonomy, list({"bacteria", "unclassified", "unassigned"}))