# adapted from https://github.com/taolei87/rcnn/tree/master/code/rationale

import gzip
import random
import json

# import theano
import numpy as np
import torch

# from nn import EmbeddingLayer
# from utils import say, load_embedding_iterator


# this method from https://github.com/taolei87/rcnn/tree/master/code/utils/__init__.py:
def load_embedding_iterator(path):
    file_open = gzip.open if path.endswith(".gz") else open
    with file_open(path) as fin:
        for line in fin:
            line = line.strip()
            if line:
                parts = line.split()
                word = parts[0]
                # vals = np.array([float(x) for x in parts[1:]])
                vals = torch.FloatTensor([float(x) for x in parts[1:]])
                yield word, vals


# this class from https://github.com/taolei87/rcnn/tree/master/code/nn/basic.py:
class EmbeddingLayer(object):
    '''
        Embedding layer that
                (1) maps string tokens into integer IDs
                (2) maps integer IDs into embedding vectors (as matrix)

        Inputs
        ------

        n_d             : dimension of word embeddings; may be over-written if embs
                            is specified
        vocab           : an iterator of string tokens; the layer will allocate an ID
                            and a vector for each token in it
        oov             : out-of-vocabulary token
        embs            : an iterator of (word, vector) pairs; these will be added to
                            the layer
        fix_init_embs   : whether to fix the initial word vectors loaded from embs

    '''
    def __init__(self, n_d, vocab, oov="<unk>", embs=None, fix_init_embs=True):

        if embs is not None:
            lst_words = []
            vocab_map = {}
            emb_vals = []
            for word, vector in embs:
                assert word not in vocab_map, "Duplicate words in initial embeddings"
                vocab_map[word] = len(vocab_map)
                emb_vals.append(vector)
                lst_words.append(word)

            self.init_end = len(emb_vals) if fix_init_embs else -1
            if n_d != len(emb_vals[0]):
                say("WARNING: n_d ({}) != init word vector size ({}). Use {} instead.\n".format(
                        n_d, len(emb_vals[0]), len(emb_vals[0])
                    ))
                n_d = len(emb_vals[0])

            say("{} pre-trained embeddings loaded.\n".format(len(emb_vals)))

            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    emb_vals.append(random_init((n_d,))*(0.001 if word != oov else 0.0))
                    lst_words.append(word)

            emb_vals = np.vstack(emb_vals).astype(theano.config.floatX)
            self.vocab_map = vocab_map
            self.lst_words = lst_words
        else:
            lst_words = []
            vocab_map = {}
            for word in vocab:
                if word not in vocab_map:
                    vocab_map[word] = len(vocab_map)
                    lst_words.append(word)

            self.lst_words = lst_words
            self.vocab_map = vocab_map
            emb_vals = random_init((len(self.vocab_map), n_d))
            self.init_end = -1

        if oov is not None and oov is not False:
            assert oov in self.vocab_map, "oov {} not in vocab".format(oov)
            self.oov_tok = oov
            self.oov_id = self.vocab_map[oov]
        else:
            self.oov_tok = None
            self.oov_id = -1

        self.embeddings = create_shared(emb_vals)
        if self.init_end > -1:
            self.embeddings_trainable = self.embeddings[self.init_end:]
        else:
            self.embeddings_trainable = self.embeddings

        self.n_V = len(self.vocab_map)
        self.n_d = n_d

    def map_to_words(self, ids):
        n_V, lst_words = self.n_V, self.lst_words
        return [lst_words[i] if i < n_V else "<err>" for i in ids]

    def map_to_ids(self, words, filter_oov=False):
        '''
            map the list of string tokens into a numpy array of integer IDs

            Inputs
            ------

            words           : the list of string tokens
            filter_oov      : whether to remove oov tokens in the returned array


            Outputs
            -------

            return the numpy array of word IDs

        '''
        vocab_map = self.vocab_map
        oov_id = self.oov_id
        if filter_oov:
            def not_oov(x):
                return x != oov_id
            return np.array(
                    filter(not_oov, [vocab_map.get(x, oov_id) for x in words]),
                    dtype="int32"
                )
        else:
            return np.array(
                    [vocab_map.get(x, oov_id) for x in words],
                    dtype="int32"
                )

    def forward(self, x):
        '''
            Fetch and return the word embeddings given word IDs x

            Inputs
            ------

            x           : a theano array of integer IDs


            Outputs
            -------

            a theano matrix of word embeddings
        '''
        return self.embeddings[x]

    @property
    def params(self):
        return [self.embeddings_trainable]

    @params.setter
    def params(self, param_list):
        self.embeddings.set_value(param_list[0].get_value())


def read_rationales(path):
    data = []
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path, 'rt') as fin:
        for line in fin:
            item = json.loads(line)
            data.append(item)
    return data


def read_annotations(path):
    data_x, data_y = [], []
    fopen = gzip.open if path.endswith(".gz") else open
    with fopen(path, 'rt') as fin:
        for line in fin:
            y, sep, x = line.partition("\t")
            x, y = x.split(), y.split()
            if len(x) == 0:
                continue
            y = np.asarray([float(v) for v in y], dtype=np.float32)
            data_x.append(x)
            data_y.append(y)
    print("{} examples loaded from {}\n".format(
            len(data_x), path
        ))
    print("max text length: {}\n".format(
        max(len(x) for x in data_x)
    ))
    return data_x, data_y

# def create_embedding_layer(path):
#     embedding_layer = EmbeddingLayer(
#             n_d = 200,
#             vocab = [ "<unk>", "<padding>" ],
#             embs = load_embedding_iterator(path),
#             oov = "<unk>",
#             #fix_init_embs = True
#             fix_init_embs = False
#         )
#     return embedding_layer


def create_batches(x, y, batch_size, padding_id, sort=True):
    batches_x, batches_y = [], []
    N = len(x)
    M = (N - 1) // batch_size + 1
    # M = (N + batch_size - 1) // batch_size
    if sort:
        perm = range(N)
        perm = sorted(perm, key=lambda i: len(x[i]))
        x = [x[i] for i in perm]
        y = [y[i] for i in perm]
    for i in range(M):
        bx, by = create_one_batch(
                    # batch_size=batch_size,
                    lstx=x[i*batch_size:(i+1)*batch_size],
                    lsty=y[i*batch_size:(i+1)*batch_size],
                    padding_id=padding_id
                )
        batches_x.append(bx)
        batches_y.append(by)
    if sort:
        random.seed(5817)
        perm2 = list(range(M))
        random.shuffle(perm2)
        batches_x = [batches_x[i] for i in perm2]
        batches_y = [batches_y[i] for i in perm2]
    return batches_x, batches_y


def create_one_batch(lstx, lsty, padding_id):
    """
    lstx is a list of 1-d LongTensors
    """
    batch_size = len(lstx)
    print('lengths', [x.shape[0] for x in lstx])
    max_len = max(x.shape[0] for x in lstx)
    print('max_len', max_len)
    assert min(x.shape[0] for x in lstx) > 0
    # bx = torch.LongTensor(batch_size, max_len)
    bx = torch.LongTensor(max_len, batch_size)
    print('bx.shape', bx.shape)
    bx.fill_(padding_id)
    # print('lstx', lstx)
    for n in range(batch_size):
        this_len = lstx[n].shape[0]
        bx[:this_len, n] = lstx[n]
    # bx = np.column_stack([np.pad(x, (max_len-len(x), 0), "constant",
    #                      constant_values=padding_id) for x in lstx])
    # by = np.vstack(lsty).astype(np.float32)
    by = torch.Tensor(lsty)
    return bx, by
