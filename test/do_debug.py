"""
Various debug/diagnostics things. Sort of like unit tests, but, well, more like pre-cursors
to unit tests
"""
from __future__ import print_function, division
import argparse
import myio
import pickle
import os
import numpy as np
import rationale_helper
import embeddings_helper


def print_samples(emb_file, num_samples, seed):
    r = np.random.RandomState(seed)

    with open(emb_file, 'rb') as f:
        d = pickle.load(f)
    x_idxes = d['x_idxes']
    words = d['words']
    N = len(x_idxes)
    print('N', N)
    sample_idxes = r.choice(N, num_samples, replace=False)
    print('sample_idxes', sample_idxes)
    for i in range(num_samples):
        idx = sample_idxes[i]
        x_idx = x_idxes[idx]
        print(rationale_helper.rationale_to_string(words=words, rationale=x_idx))


def print_combined_samples(emb_files, num_samples, seed):
    r = np.random.RandomState(seed)

    x_idxes_list = []
    print('loading %s' % emb_files[0], end='', flush=True)
    with open(emb_files[0], 'rb') as f:
        d = pickle.load(f)
        x_idxes_list.append(d['x_idxes'])
    print(' ... loaded')
    words = d['words']
    idx_by_word = d['idx_by_word']
    embedding = d['embedding']
    for emb_file in emb_files[1:]:
        print('loading %s' % emb_file, end='', flush=True)
        with open(emb_file, 'rb') as f:
            d_next = pickle.load(f)
        print(' ... loaded')
        x_idxes_list.append(d_next['x_idxes'])
        c = embeddings_helper.combine_embeddings(
            [embedding, d_next['embedding']],
            [idx_by_word, d_next['idx_by_word']])
        idx_by_word = c['idx_by_word']
        words = c['words']
    for j, x_idxes in enumerate(x_idxes_list):
        print('list %s' % j)
        N = len(x_idxes)
        print('N', N)
        sample_idxes = r.choice(N, num_samples, replace=False)
        print('sample_idxes', sample_idxes)
        for i in range(num_samples):
            idx = sample_idxes[i]
            x_idx = x_idxes[idx]
            print(rationale_helper.rationale_to_string(words=words, rationale=x_idx))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_ = subparsers.add_parser('print-samples')
    parser_.add_argument('--emb-file', type=str, default='data/reviews.aspect1.train.emb')
    parser_.add_argument('--num-samples', type=int, default=10)
    parser_.add_argument('--seed', type=int, default=123)
    parser_.set_defaults(func=print_samples)

    parser_ = subparsers.add_parser('print-combined-samples')
    parser_.add_argument(
        '--emb-files', type=str,
        default='data/reviews.aspect1.train.emb,data/reviews.aspect1.heldout.emb', help='comma-separated')
    parser_.add_argument('--num-samples', type=int, default=10)
    parser_.add_argument('--seed', type=int, default=123)
    parser_.set_defaults(func=print_combined_samples)

    args = parser.parse_args()
    func = args.func
    args_dict = args.__dict__
    del args_dict['func']
    if func == print_combined_samples:
        args_dict['emb_files'] = args_dict['emb_files'].split(',')
    func(**args_dict)
