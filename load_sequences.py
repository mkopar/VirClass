__author__ = 'Matej'

import os
import pickle
from Bio import Entrez
from Bio import SeqIO
from collections import defaultdict

dir = "../Diploma/cache"
if not os.path.isdir(dir):
    dir = "cache"
    if not os.path.isdir(dir):
        os.makedirs(dir)


def get_rec(rec_ID):
    """
    Get record for given genome id.
    :param rec_ID: genome id
    :return: record
    """
    try:
        rec = pickle.load(open(dir + "/%s.pkl.gz" % rec_ID, "rb"))
    except IOError:  # , FileNotFoundError:
        print("downloading sequence id:", rec_ID)
        handle = Entrez.efetch(db="nucleotide", rettype="gb", id=rec_ID)
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
    """
    Get record and return gene sequence.
    :param rec: record
    :return: gene sequence
    """
    sequence = ""
    for f in rec.features:
        if f.type == "gene":
            start = f.location.nofuzzy_start
            end = f.location.nofuzzy_end
            if f.location.strand == 1:
                sequence += rec.seq[start:end]
            else:
                # ??
                sequence += rec.seq[start:end].complement()

    return str(sequence)


def update_taxonomy(taxonomy, tax_path, seq_record):
    """
    Create dictionary with taxonomy name and IDs of sequences which belongs to specific taxonomy.
    :param taxonomy: current taxonomy
    :param tax_path: taxonomy path
    :param seq_record: record
    :return: updated taxonomy
    """
    if len(tax_path) == 0:
        return taxonomy

    tax = tax_path[0].lower()
    if tax in taxonomy:  # check if tax in taxonomy and update
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


def check_taxonomy_filter(rec, to_filter):
    """
    Check if record is in filter list.
    :param rec: record
    :param to_filter: filter list
    :return: bool
    """
    in_to_filter = False
    for temp_tax in rec.annotations["taxonomy"]:
        temp_tax = temp_tax.lower().split()
        for temp_tax_el in temp_tax:
            if temp_tax_el in to_filter:
                in_to_filter = True
                print "filtered ", rec.annotations["taxonomy"]
    return in_to_filter


def print_nice(taxonomy, level=0):
    """
    Print taxonomy with tabs.
    :param taxonomy: taxonomy
    :param level: current level
    :return:
    """
    for i in sorted(taxonomy.keys()):
        if i == "data":
            if len(taxonomy) == 1:
                return
            else:
                continue
        else:
            print level * "\t", i.replace("->", "", 1), len(taxonomy[i]["data"])
            print_nice(taxonomy[i], level + 1)


def remove_lists(taxonomy):
    """
    Remove all list nodes from taxonomy.
    :param taxonomy: taxonomy
    :return: taxonomy
    """

    # check for recurse exit
    if type(taxonomy) is defaultdict or type(taxonomy) is dict:
        for i in [x for x in taxonomy.keys() if x != "data"]:
            if set(taxonomy[i]) == set(list({"data"})):
                # if parent has only one list node, remove it
                #if len([x for x in taxonomy.keys() if x != "data"]) == 1:
                taxonomy.pop(i)
                continue
            else:
                remove_lists(taxonomy[i])
    else:
        return taxonomy


def count_list_nodes(taxonomy):
    """
    Count list nodes and return sum.
    :param taxonomy: taxonomy
    :return: int
    """
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
    """
    Get taxonomy, count examples in every node and return sum.
    :param taxonomy: taxonomy
    :return: sum of examples
    """
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


def get_list_nodes_unique(taxonomy, parent=""):
    """
    Get taxonomy and return unique list nodes.
    :param taxonomy: taxonomy
    :param parent: parent of current node
    :return: unique list nodes
    """
    # preverjeno na roke in dela
    list_nodes = list()
    keys = [x for x in taxonomy.keys() if x != "data"]
    for i in keys:
        if set(taxonomy[i]) == set(list({"data"})):
            list_nodes.append(i)
        else:
            list_nodes += get_list_nodes_unique(taxonomy[i], parent + "->" + i)
    return list_nodes


def get_all_nodes(taxonomy, parent=""):
    """
    Get taxonomy and return all nodes (including list nodes).
    :param taxonomy: taxonomy
    :return: all nodes
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


def get_list_nodes_ids_labels(d, parent=""):
    """
    Get taxonomy and return tuples of all list nodes.
    :param d: taxonomy
    :param parent: parent
    :return: list of tuples (id, class)
    """
    if len(d.keys()) > 1 or d.keys() == ["viruses"]:
        temp = []
        for k in [x for x in d.keys() if x != "data"]:
            temp += get_list_nodes_ids_labels(d[k], k)
        return temp
    else:
        return [(x, parent) for x in d["data"]]


def get_taxonomy():
    # call: python get_viral_sequence.py>log.out 2>log.err

    # all virus sequences
    # term = "Viruses[Organism] NOT srcdb_refseq[PROP] NOT cellular organisms[ORGN] AND nuccore genome samespecies[Filter] NOT nuccore genome[filter] NOT gbdiv syn[prop]"
    # only reference (refSEQ) virues sequences
    # see distinction between the two, here:
    # http://www.ncbi.nlm.nih.gov/genomes/GenomesHome.cgi?taxid=10239&hopt=faq

    """
    Build taxonomy from Entrez search.
    :return: taxonomy
    """
    term = "Viruses[Organism] AND srcdb_refseq[PROP] AND complete_genome"
    handle = Entrez.esearch(db="nucleotide", term=term, retmax=100000)
    record = Entrez.read(handle)
    id_list = sorted(set(record["IdList"]))
    print(record["Count"], len(record["IdList"]), len(id_list))

    taxonomy = rec_dd()
    count = 0
    for genome_id in id_list:
        try:
            rec = get_rec(genome_id)
            in_filter = check_taxonomy_filter(rec, list({"bacteria", "unclassified", "unassigned"}))
            if not in_filter:
                update_taxonomy(taxonomy, rec.annotations["taxonomy"], rec)

            # if count == 200:
            #     break
            # count += 1
        except Exception as e:
            print("problems...")
            print e

    return taxonomy


def run():
    """
    Build taxonomy and get list ids and labels.
    :return: data, label
    """
    taxonomy = get_taxonomy()
    remove_lists(taxonomy)
    list_nodes = get_list_nodes_ids_labels(taxonomy)
    data, labels = zip(*list_nodes)
    for label in labels:
        print label
    label_number = -1
    temp_l = []
    label_n = []
    for l in labels:
        if l not in temp_l:
            temp_l.append(l)
            label_number += 1
        label_n.append(label_number)

    return data, label_n


def main_run():
    taxonomy = get_taxonomy()
    print "no of examples after taxonomy was built: %d" % count_examples(taxonomy)
    print "no of list nodes after taxonomy was built: %d" % count_list_nodes(taxonomy)
    print_nice(taxonomy)
    remove_lists(taxonomy)
    print_nice(taxonomy)
    run()

    # save data to file
    # dir = "media"
    # if not os.path.isdir(dir):
    #     os.makedirs(dir)
    # save_obj(dir, data, "data_raw")
    #
    # label_n = []
    # temp_l = []
    # label_number = 0
    # for l in label:
    #     if l not in temp_l:
    #         temp_l.append(l)
    #         label_number += 1
    #     label_n.append(label_number)
    # save_obj(dir, label_n, "labels_raw")

    # dir = "media"
    # if not os.path.isdir(dir):
    #     os.makedirs(dir)
    #
    # with open('data_raw.txt', 'w') as outfile:
    #     ujson.dump(data, outfile)
    #
    # label_n = []
    # temp_l = []
    # label_number = 0
    # for l in label:
    #     if l not in temp_l:
    #         temp_l.append(l)
    #         label_number += 1
    #     label_n.append(label_number)
    #
    # with open('labels_raw.txt', 'w') as outfile:
    #     ujson.dump(label_n, outfile)

    # with gzip.open('media/data3-100.gz', 'wb') as file:
    #     file.writelines('\t'.join(str(j) for j in i) + '\n' for i in train_data)

    # with gzip.open('media/labels3-100.gz', 'wb') as file:
    #     file.writelines(i + '\n' for i in label)


    # import subprocess
    #
    # p = subprocess.Popen("gzip -c > media/data3_1-100.gz", shell=True, stdin=subprocess.PIPE)
    # for i in train_data:
    #     p.stdin.writelines('\t'.join(str(j) for j in i) + '\n')
    # p.communicate()  # Finish writing data and wait for subprocess to finish

    # p = subprocess.Popen("gzip -c > media/labels3_1-100.gz", shell=True, stdin=subprocess.PIPE)
    # for i in label:
    #     p.stdin.writelines(i + '\n')
    # p.communicate()  # Finish writing data and wait for subprocess to finish


# main_run()