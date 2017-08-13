# adapted from https://github.com/taolei87/rcnn/tree/master/code/rationale

import gzip
import random
import json

import numpy as np
import torch


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


def create_batches(x, y, batch_size, padding_id, sort=True):
    batches_x, batches_y = [], []
    N = len(x)
    M = (N - 1) // batch_size + 1
    if sort:
        perm = range(N)
        perm = sorted(perm, key=lambda i: len(x[i]))
        x = [x[i] for i in perm]
        y = [y[i] for i in perm]
    for i in range(M):
        bx, by = create_one_batch(
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
    max_len = max(x.shape[0] for x in lstx)
    assert min(x.shape[0] for x in lstx) > 0
    bx = torch.LongTensor(max_len, batch_size)
    bx.fill_(padding_id)
    for n in range(batch_size):
        this_len = lstx[n].shape[0]
        bx[:this_len, n] = lstx[n]
    by = torch.Tensor(lsty)
    return bx, by
