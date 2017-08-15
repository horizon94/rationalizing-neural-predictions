from __future__ import print_function, division
import torch
import pickle


def combine_embeddings(embeddings_list, idx_by_word_list):
    """
    any duplicates are assumed to be identical, ie if a word
    exists in both embeddings, the embedding should already be identical,
    otherwise behavior is undefined
    """
    num_lists = len(embeddings_list)
    idx_by_word_new = {}
    vals = []
    words = []
    for i in range(num_lists):
        idx_by_word = idx_by_word_list[i]
        embedding = embeddings_list[i]
        for word, idx in idx_by_word.items():
            if word not in idx_by_word_new:
                new_idx = len(idx_by_word_new)
                idx_by_word_new[word] = new_idx
                vals.append(embedding[idx])
                words.append(word)
    new_V = len(idx_by_word_new)
    num_hidden = vals[0].shape[-1]
    print('num_hidden', num_hidden)
    new_embedding = torch.zeros(new_V, num_hidden)
    for v in range(new_V):
        new_embedding[v] = vals[v]
    return {
        'embedding': new_embedding,
        'idx_by_word': idx_by_word_new,
        'num_hidden': num_hidden,
        'words': words
    }


def load_embedded_data(in_filename, max_examples, aspect_idx):
    print('loading %s ...' % in_filename)
    with open(in_filename, 'rb') as f:
        d = pickle.load(f)
    print(' ... loaded')
    if max_examples > 0:
        for k in ['x_idxes', 'x', 'y']:
            d[k] = d[k][:max_examples]
    d['N'] = len(d['x'])
    N = d['N']
    d['y_aspect'] = torch.zeros(N)
    for n, yv in enumerate(d['y']):
        d['y_aspect'][n] = yv[aspect_idx].item()
    d['num_hidden'] = d['embedding'].shape[1]
    return d
