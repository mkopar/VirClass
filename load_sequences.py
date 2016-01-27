__author__ = 'Matej'

import numpy as np
import os
import pickle
from Bio import Entrez
from Bio import SeqIO
from collections import defaultdict
import gzip

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


def get_gene(rec):
    sequence = ""
    for f in rec.features:
        if f.type == "gene":
            start = f.location.nofuzzy_start
            end = f.location.nofuzzy_end
            if f.location.strand == 1:
                sequence += rec.seq[start:end]
            else:
                # to nisem zihr
                sequence += rec.seq[start:end].complement()

    return str(sequence)


def update_taxonomy(taxonomy, tax_path, seq_record):
    """Create dictionary with taxonomy name and IDs of sequences which belongs to specific taoxnomy."""
    if len(tax_path) == 0:
        return taxonomy

    tax = tax_path[0].lower()
    if tax in taxonomy:
        taxonomy[tax]["data"].append(seq_record.annotations["gi"])
        # taxonomy[tax]["data"].append(get_gene(rec))
        update_taxonomy(taxonomy[tax], tax_path[1:], seq_record)
    else:
        taxonomy[tax] = dict({"data": list({seq_record.annotations["gi"]})})
        # taxonomy[tax] = dict({"data": list({get_gene(rec)})})
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
                print "filtered ", rec.annotations["taxonomy"]
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
            #ignored.append(("filter " + i, len(taxonomy[i]["data"])))
            taxonomy.pop(i)
            continue

        filter_taxonomy(taxonomy[i], filter_list)

    return taxonomy


def simplify(taxonomy):
    # check for recurse exit
    if type(taxonomy) is defaultdict or type(taxonomy) is dict:
        for i in [x for x in taxonomy.keys() if x != "data"]:
            # check the list nodes

            if set(taxonomy[i]) == set(list({"data"})):
                # if parent has only one list node, remove it
                #if len([x for x in taxonomy.keys() if x != "data"]) == 1:
                taxonomy.pop(i)
                continue

            else:
                simplify(taxonomy[i])

    else:
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


def get_list_nodes(taxonomy, parent=""):
    # preverjeno na roke in dela
    list_nodes = list()
    keys = [x for x in taxonomy.keys() if x != "data"]
    for i in keys:
        if set(taxonomy[i]) == set(list({"data"})):
            #list_nodes.append((i, parent, taxonomy[i]))
            if i == keys[-1]:
                # list_nodes.append((i, parent, taxonomy[i]))
                list_nodes.append((i, parent))
                return list_nodes
            else:
                # list_nodes.append((i, parent, taxonomy[i]))
                list_nodes.append((i, parent))
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


train_data = []
label = []
class_size = []


def build_data(taxonomy, seq_len=100):
    for node in [x for x in taxonomy.keys() if x != "data"]:
        if set(taxonomy[node]) == set(list({"data"})):
            sum_100 = 0
            for gid in taxonomy[node]["data"]:
                temp_rec = get_rec(gid)
                temp_seq = temp_rec.seq._data
                print "%s: %d for %d sequences" % (node, len(temp_seq) / seq_len, seq_len)

                sum_100 += (len(temp_seq) / seq_len)
                j = 0
                while (j+1) * seq_len < len(temp_seq):
                    vector = list(temp_seq[j * seq_len: (j + 1) * seq_len])
                    for n, e in enumerate(vector):
                        if e == "A":
                            vector[n] = 0.0
                        elif e == "T":
                            vector[n] = 0.33
                        elif e == "C":
                            vector[n] = 0.66
                        elif e == "G":
                            vector[n] = 1.0
                        else:
                            vector[n] = 0.5
                            # ce je vrednost nepoznana naj bo enakmoerno oddaljena med 0 in 1
                            # (ko bomo imel bite bi tukaj dal 0.25 na vse 4 vrednosti)
                    train_data.append(vector)
                    label.append(node)
                    j += 1
            print "number of 100 long sequences for %s: %d" % (node, sum_100)
            class_size.append((sum_100, node))

        else:
            build_data(taxonomy[node], seq_len)

if __name__ == "__main__":
    # call: python get_viral_sequence.py>log.out 2>log.err

    taxonomy = rec_dd()
    count = 0
    for genome_id in id_list:
        try:
            rec = get_rec(genome_id)
            in_filter = check_taxonomy_filter(rec.annotations["taxonomy"], list({"bacteria", "unclassified", "unassigned"}))
            if not in_filter:
                update_taxonomy(taxonomy, rec.annotations["taxonomy"], rec)

            # if count == 20:
            #     break
            # count += 1
        except Exception as e:
            print("problems...")
            print e

    print "no of examples after taxonomy was built: %d" % count_examples(taxonomy)
    print "no of list nodes after taxonomy was built: %d" % count_list_nodes(taxonomy)

    print_nice(taxonomy)

    simplify(taxonomy)

    print_nice(taxonomy)

    print get_list_nodes(taxonomy)

    build_data(taxonomy, 100)

    print len(train_data)
    print len(label)

    print sorted(class_size)

    np.save('media/data1-100', train_data)
    # with gzip.open('media/data1-100.gz', 'wb') as file:
    #     file.writelines('\t'.join(str(j) for j in i) + '\n' for i in train_data)

    label_n = []
    temp_l = []
    label_number = 0
    for l in label:
        if l not in temp_l:
            temp_l.append(l)
            label_number += 1
        label_n.append(label_number)

    np.save('media/labels1-100', label_n)
    # with gzip.open('media/labels1-100.gz', 'wb') as file:
    #     file.writelines(i + '\n' for i in label)