# pylint: disable=missing-docstring, protected-access
# pydocstyle: disable=missing-docstring
from collections import defaultdict
from io import StringIO
from unittest.mock import patch, mock_open, MagicMock, file_spec

import unittest
import numpy as np
import VirClass.VirClass.load as load


class LoadUnitTests(unittest.TestCase):
    def test_one_hot(self):
        # tests: 1x list, 1x np.array, n < number_of_classes, n = number_of_classes, n > number_of_classes
        x = [0, 1, 3, 2, 0]
        x_1 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [1, 0, 0, 0]])
        x_2 = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0], [0, 0, 0, 1, 0], [0, 0, 1, 0, 0],
                        [1, 0, 0, 0, 0]])
        number_of_classes = max(x) + 1
        self.assertRaisesRegex(AssertionError, "USER ERROR - cannot create numpy array; number of classes must be "
                                               "bigger than max number of list", load.one_hot, x, number_of_classes - 1)
        np.testing.assert_array_equal(load.one_hot(x, number_of_classes), x_1)
        np.testing.assert_array_equal(load.one_hot(x, number_of_classes + 1), x_2)
        np.testing.assert_array_equal(load.one_hot(np.array(x), number_of_classes), x_1)

    def test_seq_to_bits(self):
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
        self.assertRaisesRegex(AssertionError, "USER ERROR - number of unique nucleotides and transmission dictionary "
                                               "not present.", load.seq_to_bits, vec, None, None)
        res = load.seq_to_bits(vec, "ATCGYM", None)
        self.assertEqual(res, test_atcgym)
        self.assertEqual(len(res) % 6, 0)  # we have 6 unique nucleotides - len % 6 must be 0

        res = load.seq_to_bits(vec, "ATCG", None)
        self.assertEqual(res, test_atcg)
        self.assertEqual(len(res) % 4, 0)

        res = load.seq_to_bits(vec, None, dict_1)
        self.assertEqual(res, test_dict_1)
        self.assertEqual(len(res) % 3, 0)

        res = load.seq_to_bits(vec, "AT", dict_1)
        self.assertEqual(res, test_dict_1)
        self.assertEqual(len(res) % 3, 0)

        res = load.seq_to_bits(vec, None, dict_2)
        self.assertEqual(res, test_dict_2)
        self.assertEqual(len(res) % 2, 0)

        res = load.seq_to_bits(vec, "CTGM", dict_2)
        self.assertEqual(res, test_dict_2)
        self.assertEqual(len(res) % 2, 0)

    @patch('VirClass.VirClass.load.os.path.isfile')
    @patch('VirClass.VirClass.load.load_seqs_from_ncbi')
    def test_load_from_file_fasta(self, arg1, arg2):
        # pylint: disable=unused-argument
        load.os.path.isfile.return_value = True

        temp = defaultdict(list)
        temp['1004345262'] = \
            'TGTTGCGTTAACAACAAACCAACCTCCGACCCAAAACAAAGATGAAAATAAAAGATGCCACCCAAACGCCGACTAGTGGACAGCCCAGAAGATATGGAAGAAA' \
            'GATGCCACCCAAACGCCGACTAGTGGACAGCCCAGAAGATATGGAAGACGAGGGACCCTCTGACCGACCAACTCACCTACCCAAACTCCCAGGAACC'

        res_tuple = (
            temp,
            {
                '1004345262':
                    'Viruses;ssRNA viruses;ssRNA negative-strand viruses;Mononegavirales;Bornaviridae;Bornavirus'
            }
        )

        read_data = \
            '>1004345262 Viruses;ssRNA viruses;ssRNA negative-strand viruses;Mononegavirales;Bornaviridae;Bornavirus' \
            '\nTGTTGCGTTAACAACAAACCAACCTCCGACCCAAAACAAAGATGAAAATAAAAGATGCCACCCAAACGCCGACTAGTGGACAGCCCAGAAGATATGGAAG\n' \
            'AAAGATGCCACCCAAACGCCGACTAGTGGACAGCCCAGAAGATATGGAAGACGAGGGACCCTCTGACCGACCAACTCACCTACCCAAACTCCCAGGAACC\n'

        # https://www.biostars.org/p/190067/
        with patch('VirClass.VirClass.load.gzip.open') as mocked_open:
            handle = MagicMock(spec=file_spec)
            handle.__enter__.return_value = StringIO(read_data)
            mocked_open.return_value = handle
            res = load.load_from_file_fasta('bla.bla')
            mocked_open.assert_called_once_with('bla.bla', 'rt')
            self.assertEqual(res, res_tuple)

        load.os.path.isfile.return_value = False
        load.load_seqs_from_ncbi.return_value = res_tuple
        with patch('VirClass.VirClass.load.gzip.open', mock_open(), create=True) as mocked_open:
            res = load.load_from_file_fasta('bla.bla')
            mocked_open.assert_called_once_with('bla.bla', 'wt')
            self.assertEqual(res, res_tuple)

    # @patch('VirClass.VirClass.load.seq_to_bits')
    def test_dataset_from_id(self):
        # data
        dict_1 = {"A": [1, 0, 0, 0], "T": [0, 1, 0, 0], "C": [0, 0, 1, 0], "G": [0, 0, 0, 1]}
        temp_data = defaultdict(list)
        temp_data['1004345262'] = \
            'TGTTGCGTTAACAACAAACCAACCTCCGACCCAAAACAAAGATGAAAATAAAAGATGCCACCCAAACGCCGACTAGTGGACAGCCCAGAAGATATGGAAGAAA' \
            'GATGCCACCCAAACGCCGACTAGTGGACAGCCCAGAAGATATGGAAGACGAGGGACCCTCTGACCGACCAACTCACCTACCCAAACTCCCAGGAACC'
        temp_data['10043452'] = \
            'GATGCCACCCAAACGCCGACTAGTGGACAGCCCAGAAGATATGGAAGACGAGGGACCCTCTGACCGACCAACTCACCTACCCAAACTCCCAGGAACC' \
            'TGTTGCGTTAACAACAAACCAACCTCCGACCCAAAACAAAGATGAAAATAAAAGATGCCACCCAAACGCCGACTAGTGGACAGCCCAGAAGATATGGA'
        temp_tax = {'10043452': 0, '1004345262': 1}
        ids = ['1004345262', '10043452']

        # test1
        expected_x = [
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
             0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
             0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0,
             1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 0, 1],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0,
             1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1,
             0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
             1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
             0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0,
             0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
             1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0,
             0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
             0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0,
             0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
             0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
             0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
             1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
             0, 1, 0, 0]]
        expected_y = [1, 1, 0]
        res = load.dataset_from_id(temp_data, temp_tax, ids, 100, 1.0, dict_1)
        self.assertTrue(res, (expected_x, expected_y))

        # test2
        res = load.dataset_from_id(defaultdict(list), {}, [], 100, 0.5, dict_1)
        self.assertTrue(res, ([], []))

        # test3
        expected_x2 = [
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
             0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
             0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0,
             1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 0, 1],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
             0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0,
             0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0,
             0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
             0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
             1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1,
             0, 1, 0, 0]]
        expected_y2 = [1, 0]
        res = load.dataset_from_id(temp_data, temp_tax, ids, 100, 0.2, dict_1)
        self.assertTrue(res, (expected_x2, expected_y2))

        # test4
        expected_x20 = [
            [0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0,
             0, 0, 1, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0,
             1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0,
             0, 0, 1, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
             0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0,
             0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
             0, 1, 0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0,
             0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0,
             0, 0, 0, 1, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0,
             1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1,
             1, 0, 0, 0, 0, 1, 0, 0],
            [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,
             0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0,
             1, 0, 0, 0, 0, 0, 0, 1]]
        expected_y20 = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
        res = load.dataset_from_id(temp_data, temp_tax, ids, 20, 0.5, dict_1)
        self.assertTrue(res, (expected_x20, expected_y20))

        self.assertRaisesRegex(AssertionError, "", load.dataset_from_id, temp_data, temp_tax, ids, 20, 20, dict_1)

    @patch('VirClass.VirClass.load.pickle.load')
    def test_load_dataset(self, mock_pickle_load):
        m_file = mock_open()
        with patch('VirClass.VirClass.load.gzip.open', m_file):
            load.load_dataset('bla.bla')
            mock_pickle_load.assert_called_once()
            self.assertTrue(m_file.called)
            m_file.assert_called_once_with('bla.bla', 'rt')

    @patch('VirClass.VirClass.load.pickle.dump')
    def test_save_dataset(self, mock_pickle_dump):
        m_file = mock_open()
        with patch('VirClass.VirClass.load.gzip.open', m_file):
            load.save_dataset('bla.bla', {'test_key': 'test_val'})
            mock_pickle_dump.assert_call_once_with({'test_key': 'test_val'})
            self.assertTrue(m_file.called)
            m_file.assert_called_once_with('bla.bla', 'wt')

    def test_build_dataset_ids(self):
        oids = ['1006610892', '1021076629', '1023464444', '1028356461', '1028356384', '1006160387', '10086561',
                '1016776533', '1005739119', '10140926', '10313991', '1007626122', '1021076583', '10257473',
                '1021076642', '1004345262', '1002160105', '1023176908', '1007626112', '1024325226']
        res = load.build_dataset_ids(oids=oids, test=0.2, seed=0)
        self.assertTrue(isinstance(res, dict))
        self.assertEqual(len(res.keys()), 4)
        self.assertCountEqual(res.keys(), ['tr_ids', 'te_ids', 'trtr_ids', 'trte_ids'])
        self.assertEqual(len(res['tr_ids'] + res['te_ids']), len(oids))
        self.assertCountEqual(res['tr_ids'] + res['te_ids'], oids)
        self.assertTrue(set(res['tr_ids']).isdisjoint(res['te_ids']))
        self.assertEqual(len(res['trtr_ids'] + res['trte_ids']), len(res['tr_ids']))
        self.assertCountEqual(res['trtr_ids'] + res['trte_ids'], res['tr_ids'])
        self.assertTrue(set(res['trtr_ids']).isdisjoint(res['trte_ids']))

        self.assertRaisesRegex(ValueError, "test_size=1.000000 should be smaller than 1.0 or be an integer",
                               load.build_dataset_ids, oids, 1.0, 0)

        datasets2 = {'tr_ids': [], 'te_ids': [], 'trte_ids': [], 'trtr_ids': []}
        res = load.build_dataset_ids([], test=0.2, seed=0)
        self.assertTrue(isinstance(res, dict))
        self.assertDictEqual(res, datasets2)

        res = load.build_dataset_ids(oids, test=0.0, seed=0)
        self.assertTrue(isinstance(res, dict))
        self.assertEqual(len(res.keys()), 4)
        self.assertCountEqual(res.keys(), ['tr_ids', 'te_ids', 'trtr_ids', 'trte_ids'])
        self.assertEqual(len(res['tr_ids'] + res['te_ids']), len(oids))
        self.assertCountEqual(res['tr_ids'] + res['te_ids'], oids)
        self.assertTrue(set(res['tr_ids']).isdisjoint(res['te_ids']))
        self.assertEqual(len(res['te_ids']), 0)
        self.assertEqual(len(res['tr_ids']), len(oids))
        self.assertEqual(len(res['trtr_ids'] + res['trte_ids']), len(res['tr_ids']))
        self.assertCountEqual(res['trtr_ids'] + res['trte_ids'], res['tr_ids'])
        self.assertTrue(set(res['trtr_ids']).isdisjoint(res['trte_ids']))

        self.assertRaisesRegex(ValueError, "test_size=1.000000 should be smaller than 1.0 or be an integer",
                               load.build_dataset_ids, oids, 1.0, 0)

    def test_classes_to_numerical(self):
        temp = defaultdict(list)
        temp['1004345262'] = \
            'TGTTGCGTTAACAACAAACCAACCTCCGACCCAAAACAAAGATGAAAATAAAAGATGCCACCCAAACGCCGACTAGTGGACAGCCCAGAAGATATGGAAGAAA' \
            'GATGCCACCCAAACGCCGACTAGTGGACAGCCCAGAAGATATGGAAGACGAGGGACCCTCTGACCGACCAACTCACCTACCCAAACTCCCAGGAACC'
        temp['10043452'] = \
            'GATGCCACCCAAACGCCGACTAGTGGACAGCCCAGAAGATATGGAAGACGAGGGACCCTCTGACCGACCAACTCACCTACCCAAACTCCCAGGAACC' \
            'TGTTGCGTTAACAACAAACCAACCTCCGACCCAAAACAAAGATGAAAATAAAAGATGCCACCCAAACGCCGACTAGTGGACAGCCCAGAAGATATGGA'

        labels = {'1004345262':
                  'Viruses;ssRNA viruses;ssRNA negative-strand viruses;Mononegavirales;Bornaviridae;Bornavirus',
                  '10043452': 'Viruses;ssRNA viruses;ssRNA positive-strand viruses;ViruslesA;ViridaeB;VirusC'}

        res_temp = defaultdict(int)
        res_temp[0] = 195.0
        res_temp[1] = 200.0
        res_expect = ({'10043452': 0, '1004345262': 1}, res_temp)

        res = load.classes_to_numerical(temp, labels)
        self.assertTrue(res, res_expect)

        # try with empty
        res = load.classes_to_numerical(defaultdict(list), {})
        self.assertTrue(res, ({}, defaultdict(int)))

    def test_load_data(self):
        # good luck :)
        pass


if __name__ == '__main__':
    unittest.main()
