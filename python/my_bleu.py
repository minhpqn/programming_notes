"""
Implementation of BLEU Score
References:
    [1] https://en.wikipedia.org/wiki/BLEU
    [2] http://www.aclweb.org/anthology/P02-1040.pdf
"""
import math
import unittest
from collections import defaultdict


def makefilter(hstr):
    set = '0123456789-'
    return ''.join([c for c in hstr if c in set])


def makelist_str(hstr):
    text = makefilter(hstr)
    return text.split('-')


def makelist_char(hstr):
    text = makefilter(hstr)
    temp=[]
    for i in range(0, len(text)):
        temp.append(text[i])
    return temp


def _get_ngrams(n, tokens):
    ngram_counts = defaultdict(int)
    for i in range(0, len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngram_counts[ngram] += 1
    return ngram_counts


def compute_precision(n, output_tokens, reference_tokens):
    ref_ngram_counts = _get_ngrams(n, reference_tokens)
    output_ngram_counts = _get_ngrams(n, output_tokens)
    total_output_ngrams = 0
    matched_ngrams = 0
    for ngram in output_ngram_counts:
        total_output_ngrams += output_ngram_counts[ngram]
        if ngram in ref_ngram_counts:
            matched_ngrams += min(ref_ngram_counts[ngram], output_ngram_counts[ngram])
    if total_output_ngrams > 0:
        if matched_ngrams > 0:
            return matched_ngrams / total_output_ngrams
        else:
            return 0.0
    else:
        return 0.0


def compute_bleu(output_tokens, reference_tokens, max_order=4, use_bp=True):
    precisions = [0] * max_order
    for i in range(0, max_order):
        precisions[i] = compute_precision(i+1, output_tokens, reference_tokens)

    if max(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions if p)
        geo_mean = math.exp(p_log_sum / max_order)

    reference_length = len(reference_tokens)
    translation_length = len(output_tokens)
    if use_bp:
        ratio = translation_length / reference_length
        bp = min(1.0, ratio)
    bleu = geo_mean * bp
    return bleu


def str_score(output, expected):
    list_output = makelist_str(output)
    list_expected = makelist_str(expected)
    # print(list_output, list_expected)
    return compute_bleu(list_output, list_expected)


def char_score(output, expected):
    list_output = makelist_char(output)
    list_expected = makelist_char(expected)
    # print(list_output, list_expected)
    return compute_bleu(list_output, list_expected)


class TestBLEU(unittest.TestCase):

    def test_bleu(self):
        reference = ['2007', '07', '11']
        output1 = ['2007', '07', '11']
        output2 = ['2007', '07', '12']
        self.assertAlmostEqual(1.0, compute_bleu(output1, reference))
        self.assertAlmostEqual(0.7598356856515925, compute_bleu(output2, reference))

        # Test modified BLEU score by using the example from Wikipedia article
        # https://en.wikipedia.org/wiki/BLEU
        reference1 = ["the", "cat", "is", "on", "the", "mat"]
        reference2 = ["there", "is", "a", "cat", "on", "the", "mat"]
        candidate1 = ["the", "the", "the", "the", "the", "the", "the"]
        candidate2 = ["the", "cat"]
        self.assertAlmostEqual(2.0/7, compute_precision(1, candidate1, reference1))
        self.assertAlmostEqual(1.0 / 7, compute_precision(1, candidate1, reference2))
        self.assertEqual(1.0, compute_precision(1, candidate2, reference1))
        self.assertEqual(1.0, compute_precision(2, candidate2, reference1))

    def test_char_score(self):
        output1 = '2007-07-11<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
        output2 = '2007-07-12<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'  # it's 12 instead of 011
        expected = '2007-07-11'

        self.assertAlmostEqual(1.0, char_score(output1, expected))
        self.assertAlmostEqual(0.8801117367933934, char_score(output2, expected))

    def test_str_score(self):
        output1 = '2007-07-11<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'
        output2 = '2007-07-12<pad><pad><pad><pad><pad><pad><pad><pad><pad><pad>'  # it's 12 instead of 011
        expected = '2007-07-11'

        self.assertAlmostEqual(1.0, str_score(output1, expected))
        self.assertAlmostEqual(0.7598356856515925, str_score(output2, expected))