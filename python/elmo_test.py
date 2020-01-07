import numpy as np
np.random.seed(42)
from elmoformanylangs import Embedder

e = Embedder('/Users/minhpqn/nlp/data/elmo/ja')


# inp: list of tokenized sentences
# E.g: sents = [
#     ['今'],
#     ['潮水', '退']
# ]
# option = {0,1,2,-1} the target layer to output
# 0: word encoder
# 1: first lstm hidden layer
# 2: second lstm hidden layer
# -1: average of three layers (default)
#
# return: list of numpy.ndarray size Nx1024 (N = number of tokens in a sentence)
#
def get_elmo_vec(inp, option=-1):
        return e.sents2elmo(inp, output_layer=option)


def get_elmo_sen_vec(sen, option=-1):
    inp = [sen]
    vec = get_elmo_vec(inp, option)
    return vec[0]


def padding_elmo_vec(elmo_vec, padding_len):
    """Do padding for an ELMo vector

    Args:
        elmo_vec: numpy array of shape (sequence_length, elmo_dim)
        padding_len:

    Returns:
        numpy array of shape (padding_len, elmo_dim)
    """
    seq_len, elmo_dim = elmo_vec.shape
    pad = np.zeros((1, elmo_dim), dtype="float32")
    if seq_len >= padding_len:
        ret = elmo_vec[:padding_len]
    else:
        ret = elmo_vec
        for i in range(padding_len - seq_len):
            ret = np.vstack([ret, pad])
    return ret


if __name__ == '__main__':
    sents = [
        ['今'],
        ['潮水', '退']
    ]
    print(get_elmo_vec(sents, option=1))
    print()
    print(get_elmo_vec(sents, option=1))
    print()
    print(get_elmo_vec(sents, option=1))
    print()
    sen_vec = get_elmo_sen_vec(['今', '潮水', '退'])
    print(sen_vec.shape)
    print(sen_vec)

    print()
    new_sen_vec = padding_elmo_vec(sen_vec, 4)
    print(new_sen_vec.shape)
    print(new_sen_vec)

    print()
    new_sen_vec = padding_elmo_vec(sen_vec, 2)
    print(new_sen_vec.shape)
    print(new_sen_vec)