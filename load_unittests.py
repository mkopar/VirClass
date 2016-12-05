import unittest
from load import *
import load_ncbi
import numpy as np


class LoadUnitTests(unittest.TestCase):

    def test_one_hot(self):
        # tests: 1x list, 1x np.array, n < number_of_classes, n = number_of_classes, n > number_of_classes
        x = [0, 1, 3, 2, 0]
        x_1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]]).astype(float)
        x_2 = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0], [1, 0, 0, 0, 0]]).astype(float)
        number_of_classes = max(x) + 1
        self.assertRaisesRegexp(AssertionError, "USER ERROR - cannot create numpy array; invalid number of classes", one_hot, x, number_of_classes - 1)
        np.testing.assert_array_equal(one_hot(x, number_of_classes), x_1)
        np.testing.assert_array_equal(one_hot(x, number_of_classes + 1), x_2)
        np.testing.assert_array_equal(one_hot(np.array(x), number_of_classes), x_1)

    def test_seq_to_bits(self):
        # vse moznosti
        vec = "ATCGYM"
        test_atcgym = [1, 0, 0, 0, 0, 0,
                       0, 1, 0, 0, 0, 0,
                       0, 0, 1, 0, 0, 0,
                       0, 0, 0, 1, 0, 0,
                       0, 0, 0, 0, 1, 0,
                       0, 0, 0, 0, 0, 1]
        test_atcg = [1, 0, 0, 0,
                     0, 1, 0, 0,
                     0, 0, 1, 0,
                     0, 0, 0, 1,
                     1, 1, 1, 1,
                     1, 1, 1, 1]
        dict_1 = {"A": [1, 1, 0], "G": [1, 0, 0], "T": [1, 1, 1]}
        test_dict_1 = [1, 1, 0,
                       1, 1, 1,
                       1, 1, 1,
                       1, 0, 0,
                       1, 1, 1,
                       1, 1, 1]
        dict_2 = {"T": [1, 0], "C": [0, 1]}
        test_dict_2 = [1, 1,
                       1, 0,
                       0, 1,
                       1, 1,
                       1, 1,
                       1, 1]
        self.assertRaisesRegexp(AssertionError, "USER ERROR - number of unique nucleotides and transmission dictionary not present.", seq_to_bits, vec, None, None)
        res = seq_to_bits(vec, "ATCGYM", None)
        self.assertEqual(res, test_atcgym)
        self.assertEqual(len(res) % 6, 0)  # we have 6 unique nucleotides - len % 6 must be 0

        res = seq_to_bits(vec, "ATCG", None)
        self.assertEqual(res, test_atcg)
        self.assertEqual(len(res) % 4, 0)

        res = seq_to_bits(vec, None, dict_1)
        self.assertEqual(res, test_dict_1)
        self.assertEqual(len(res) % 3, 0)

        res = seq_to_bits(vec, "AT", dict_1)
        self.assertEqual(res, test_dict_1)
        self.assertEqual(len(res) % 3, 0)

        res = seq_to_bits(vec, None, dict_2)
        self.assertEqual(res, test_dict_2)
        self.assertEqual(len(res) % 2, 0)

        res = seq_to_bits(vec, "CTGM", dict_2)
        self.assertEqual(res, test_dict_2)
        self.assertEqual(len(res) % 2, 0)

    def test_load_from_file_fasta(self):
        # poklici prvic, da se nardi en fasta file, potem pa poklici se enkrat in se mora nalozit
        # preveri vse parametre

        # 1. poklici z enim imenom fajla - naredu se bo fasta, vrnu bo sekvence pa razrede
        # 2. preveri da fajl res obstaja
        # 3. poklici se enkrat z istim imenom - nalozit se mora fasta (kako bom to vedu) in vrnt sekvence in razrede - razredi morajo bit isti kot prej
        # 4. pobrisi fajl
        pass

    def test_dataset_from_id(self):
        # za podane ID-je zelimo zgraditi teX in teY
        pass

    def test_load_data(self):
        # good luck :)
        pass


if __name__ == '__main__':
    unittest.main()
