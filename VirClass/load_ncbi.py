"""
Module for preparing data from NCBI. Most low layer module for manipulating data.
"""

import os
import pickle
from collections import defaultdict

from Bio import Entrez
from Bio import SeqIO

# move to init
cache_dir = "../Diploma/cache"
if not os.path.isdir(cache_dir):
    cache_dir = "cache"
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)


# ************  NCBI RECORD OPERATIONS  ************ #

def get_gids(term="Viruses[Organism] AND srcdb_refseq[PROP] AND complete_genome"):
    """
    Get genome IDs for given search term.
    :param term: search term for NCBI query
    :return: list of genome IDs for given term
    """
    # term = "Viruses[Organism] AND srcdb_refseq[PROP] AND complete_genome"
    handle = Entrez.esearch(db="nucleotide", term=term, retmax=100000)
    record = Entrez.read(handle)
    id_list = sorted(set(record["IdList"]))
    print((record["Count"], len(record["IdList"]), len(id_list)))
    return id_list


def get_rec(rec_id):
    """
    Get record for given genome id.
    :param rec_id: genome id
    :return: record
    """
    try:
        rec = pickle.load(open(os.path.join(cache_dir, "/%s.pkl.gz" % rec_id), "rb"))
    except IOError:  # , FileNotFoundError:
        print(("downloading sequence id:", rec_id))
        handle = Entrez.efetch(db="nucleotide", rettype="gb", id=rec_id)
        rec = SeqIO.read(handle, "gb")
        handle.close()
        pickle.dump(rec, open(os.path.join(cache_dir, "/%s.pkl.gz" % rec_id), "wb"), -1)
        print(("genome size:", len(rec.seq), rec.seq[:20] + "..."))
        print(("Taxonomy:", rec.annotations['taxonomy']))
        for a, t in list(rec.annotations.items()):
            print(("  %s: %s" % (a, str(t)[:15])))
        print()
    return rec


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


def load_oid_seq_classification(ids):
    """
    Build dictionary of sequences and taxonomies for every genome ID.
    :param ids: genome IDs
    :return: sequences and taxonomy annotations dictionaries for every genome ID
    """
    seq = defaultdict()
    tax = {}
    for oid in ids:
        rec = get_rec(oid)
        seq[oid] = str(rec.seq)
        tax[oid] = ';'.join(rec.annotations["taxonomy"])

    return seq, tax


# ************  TAXONOMY OPERATIONS  ************ #

def rec_dd():
    """
    Create dictionary of dictionaries to 'simulate' tree.
    :return: dictionary of dictionaries
    """
    return defaultdict(rec_dd)


def update_taxonomy(temp_taxonomy, tax_path, genome_id):
    """
    Create dictionary with taxonomy name and IDs of sequences which belongs to specific taxonomy.
    :param temp_taxonomy: current taxonomy
    :param tax_path: taxonomy path
    :param genome_id: genome_id
    :return: updated taxonomy
    """
    if len(tax_path) == 0:
        return temp_taxonomy

    tax = tax_path[0].lower()
    if tax in temp_taxonomy:  # check if tax in taxonomy and update
        # temp_taxonomy[tax]["data"].append(seq_record.annotations["gi"])
        temp_taxonomy[tax]["data"].append(genome_id)
        # taxonomy[tax]["data"].append(get_gene(rec))
        update_taxonomy(temp_taxonomy[tax], tax_path[1:], genome_id)
    else:
        # temp_taxonomy[tax] = {"data": list({seq_record.annotations["gi"]})}
        temp_taxonomy[tax] = {"data": list({genome_id})}
        # taxonomy[tax] = dict({"data": list({get_gene(rec)})})
        temp = update_taxonomy(temp_taxonomy[tax], tax_path[1:], genome_id)
        if len(temp) > 1:  # 1 = data, 2 = data + key
            temp_taxonomy = temp
    return temp_taxonomy


def filter_classification(rec, to_filter):
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
                print("filtered ", rec.annotations["taxonomy"])
    return in_to_filter


def print_nice(temp_taxonomy, level=0):
    """
    Print taxonomy with tabs.
    :param temp_taxonomy: taxonomy
    :param level: current level
    :return:
    """
    for i in sorted(temp_taxonomy.keys()):
        if i == "data":
            if len(temp_taxonomy) == 1:
                return
            else:
                continue
        else:
            print(level * "\t", i.replace("->", "", 1), len(temp_taxonomy[i]["data"]))
            print_nice(temp_taxonomy[i], level + 1)


def load_whole_taxonomy():
    """
    Build taxonomy and get list ids and labels.
    :return: data, label
    """
    temp_taxonomy = get_taxonomy(get_gids())
    list_nodes = get_list_nodes_ids_labels(temp_taxonomy)
    data, labels = list(zip(*list_nodes))
    for label in labels:
        print(label)
    label_number = -1
    temp_l = []
    label_n = []
    for l in labels:
        if l not in temp_l:
            temp_l.append(l)
            label_number += 1
        label_n.append(label_number)

    return data, label_n


def get_taxonomy(id_list, count=-1):
    # call: python get_viral_sequence.py>log.out 2>log.err

    # all virus sequences
    # term = "Viruses[Organism] NOT srcdb_refseq[PROP] NOT cellular organisms[ORGN] AND
    #           nuccore genome samespecies[Filter] NOT nuccore genome[filter] NOT gbdiv syn[prop]"
    # only reference (refSEQ) virues sequences
    # see distinction between the two, here:
    # http://www.ncbi.nlm.nih.gov/genomes/GenomesHome.cgi?taxid=10239&hopt=faq

    """
    Build taxonomy from Entrez search.
    :param id_list: list of genome ids we want to build taxonomy tree from
    :param count: how many elements we want in taxonomy; -1 means whole taxonomy
    :return: taxonomy
    """

    temp_taxonomy = rec_dd()
    temp_count = 1
    for genome_id in id_list:
        try:
            rec = get_rec(genome_id)
            in_filter = filter_classification(rec, list({"bacteria", "unclassified", "unassigned"}))
            if not in_filter:
                update_taxonomy(temp_taxonomy, rec.annotations["taxonomy"], genome_id)

                if count != -1:
                    if temp_count == count:
                        break
                    temp_count += 1
        except IOError as e:
            # efetch - Raises an IOError exception if there's a network error.
            # http://biopython.org/DIST/docs/api/Bio.Entrez-module.html
            print("problems in network connection...")
            print(e)
        except ValueError as v:
            # http: // biopython.org / DIST / docs / api / Bio.SeqIO - module.html  # read
            print("problems with handling SeqIO...")
            print(v)
        except pickle.PicklingError as p:
            # https://docs.python.org/2/library/pickle.html#pickle.PicklingError
            print("problems with pickling object...")
            print(p)

    return temp_taxonomy


# ************  LIST OPERATIONS  ************ #

def remove_lists(temp_taxonomy):
    """
    Remove all list nodes from taxonomy.
    :param temp_taxonomy: taxonomy
    :return: taxonomy
    """

    # check for recurse exit
    if isinstance(temp_taxonomy, defaultdict) or isinstance(temp_taxonomy, dict):
        for i in [x for x in list(temp_taxonomy.keys()) if x != "data"]:
            if set(temp_taxonomy[i]) == set(list({"data"})):
                # if parent has only one list node, remove it
                # if len([x for x in taxonomy.keys() if x != "data"]) == 1:
                temp_taxonomy.pop(i)
                continue
            else:
                remove_lists(temp_taxonomy[i])
    else:
        return temp_taxonomy


def get_list_nodes_unique(temp_taxonomy, parent=""):
    """
    Get taxonomy and return unique list nodes.
    :param temp_taxonomy: taxonomy
    :param parent: parent of current node
    :return: unique list nodes
    """
    # checked by hand and it works as expected
    list_nodes = list()
    keys = [x for x in list(temp_taxonomy.keys()) if x != "data"]
    for i in keys:
        if set(temp_taxonomy[i]) == set(list({"data"})):
            list_nodes.append(i)
        else:
            list_nodes += get_list_nodes_unique(temp_taxonomy[i], parent + "->" + i)
    return list_nodes


def count_list_nodes(temp_taxonomy):
    """
    Count list nodes and return sum.
    :param temp_taxonomy: taxonomy
    :return: int
    """
    count = 0
    keys = [x for x in list(temp_taxonomy.keys()) if x != "data"]
    for i in keys:
        if set(temp_taxonomy[i]) == set(list({"data"})):
            if i == keys[-1]:
                count += 1
                return count
            else:
                count += 1
        else:
            count += count_list_nodes(temp_taxonomy[i])
    return count


def get_list_nodes_ids_labels(d, parent=""):
    """
    Get taxonomy and return tuples of all list nodes.
    :param d: taxonomy
    :param parent: parent
    :return: list of tuples (id, class)
    """
    if len(list(d.keys())) > 1 or list(d.keys()) == ["viruses"]:
        temp = []
        for k in [x for x in list(d.keys()) if x != "data"]:
            temp += get_list_nodes_ids_labels(d[k], k)
        return temp
    else:
        return [(x, parent) for x in d["data"]]


# ************  ALL NODES OPERATIONS  ************ #

def count_examples(temp_taxonomy):
    """
    Get taxonomy, count examples in every node and return sum.
    :param temp_taxonomy: taxonomy
    :return: sum of examples
    """
    count = 0
    keys = [x for x in list(temp_taxonomy.keys()) if x != "data"]
    for i in keys:
        if set(temp_taxonomy[i]) == set(list({"data"})):
            if i == keys[-1]:
                count += len(temp_taxonomy[i]["data"])
                return count
            else:
                count += len(temp_taxonomy[i]["data"])
        else:
            count += count_examples(temp_taxonomy[i])
    return count


def get_all_nodes(temp_taxonomy, parent=""):
    """
    Get taxonomy and return all nodes (including list nodes).
    :param parent: parent of current node - default ""
    :param temp_taxonomy: taxonomy
    :return: all nodes
    """
    all_nodes = list()
    keys = [x for x in list(temp_taxonomy.keys()) if x != "data"]
    for i in keys:
        # if we want all non-list nodes, than this stays, otherwise comment this
        # if len([x for x in taxonomy[i].keys() if x != "data"]) == 0:
        # continue
        if i == "rest":
            all_nodes.append(parent + "->" + i)
        else:
            all_nodes.append(i)
        all_nodes += get_all_nodes(temp_taxonomy[i], i)
    return all_nodes


# ************  OTHER  ************ #

def load_seqs_from_ncbi(seq_len=100, skip_read=0, overlap=50, taxonomy_el_count=-1):
    """
    Prepare sequences, sliced to seq_len length. Skip every skip_read and overlap two reads with overlap nucleotides.
    Overlap 50 means that half of the read is going to be overlapped with next read.
    If seq_len is -1, load whole sequences (do not strip them) - usually using with fasta format as we slice sequences
    later.
    :param seq_len: read length
    :param skip_read: number of skipped reads
    :param overlap: overlapping nucleotides count
    :param taxonomy_el_count: how many elements we want in taxonomy; -1 means whole taxonomy
    :return:    dictionary reads - each genome ID key contains list of reads for specific genome,
                dictionary taxonomy - each genome ID key contains taxonomy for specific genome
    """
    data, _ = run(taxonomy_el_count)
    print("getting sequences...")
    seqs, tax = load_oid_seq_classification(data)

    reads = defaultdict(list)

    if seq_len != -1:
        for oid, seq in seqs.items():
            while seq:
                if len(seq) < seq_len:
                    # we don't want shorter sequences than seq_len (shorter than 100)
                    break
                reads[oid].append(seq[:seq_len])
                seq = seq[seq_len - overlap + ((seq_len - overlap) * skip_read):]
    else:
        reads = seqs

    return reads, tax


def run(taxonomy_el_count=-1):
    """
    Build taxonomy and get list ids and labels.
    :param taxonomy_el_count: how many elements we want in taxonomy; -1 means whole taxonomy
    :return: data, label
    """
    temp_taxonomy = get_taxonomy(get_gids(), count=taxonomy_el_count)
    # remove_lists(taxonomy)
    list_nodes = get_list_nodes_ids_labels(temp_taxonomy)
    data, labels = list(zip(*list_nodes))
    # for label in labels:
    #     print label
    label_number = -1
    temp_l = []
    label_n = []
    for l in labels:
        if l not in temp_l:
            temp_l.append(l)
            label_number += 1
        label_n.append(label_number)

    return data, label_n


if __name__ == "__main__":
    taxonomy = get_taxonomy(get_gids())
    print("no of examples after taxonomy was built: %d" % count_examples(taxonomy))
    print("no of list nodes after taxonomy was built: %d" % count_list_nodes(taxonomy))
    print_nice(taxonomy)
    remove_lists(taxonomy)
    print_nice(taxonomy)
    run()
