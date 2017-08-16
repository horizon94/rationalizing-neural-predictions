from __future__ import print_function, division
import torch
import pickle


def combine_embeddings(embedding_list, words_lists, idx_by_word_list, x_idxes_list):
    """
    any duplicates are assumed to be identical, ie if a word
    exists in both embeddings, the embedding should already be identical,
    otherwise behavior is undefined
    """
    num_lists = len(embedding_list)
    idx_by_word_new = {}
    vals = []
    words = []
    x_idxes_list_new = []
    for i in range(num_lists):
        idx_by_word = idx_by_word_list[i]
        embedding = embedding_list[i]
        for word, idx in idx_by_word.items():
            if word not in idx_by_word_new:
                new_idx = len(idx_by_word_new)
                idx_by_word_new[word] = new_idx
                vals.append(embedding[idx])
                words.append(word)
    new_V = len(idx_by_word_new)
    num_hidden = vals[0].shape[-1]
    # print('num_hidden', num_hidden)
    new_embedding = torch.zeros(new_V, num_hidden)
    for v in range(new_V):
        new_embedding[v] = vals[v]
    for i, x_idxes in enumerate(x_idxes_list):
        # print('i', i)
        x_idxes_new = []
        for j, ex in enumerate(x_idxes):
            # print('ex[:6]', ex[:6])
            # ex_new = ex.clone()
            ex_new = torch.LongTensor(*ex.shape)
            # print('ex_new[:6]', ex_new[:6])
            # print('ex.shape', ex.shape)
            for idx in range(ex.shape[0]):
                word_idx = ex[idx]
                word = words_lists[i][word_idx]
                new_idx = idx_by_word_new[word]
                ex_new[idx] = new_idx
                # if idx < 6:
                #     print('idx', idx, 'word', word, 'new_idx', new_idx)
            # print('ex_new[:6]', ex_new[:6])
            x_idxes_new.append(ex_new)
            # if j > 3:
            #     asdfasdf
        x_idxes_list_new.append(x_idxes_new)
    return {
        'embedding': new_embedding,
        'idx_by_word': idx_by_word_new,
        'num_hidden': num_hidden,
        'words': words,
        'x_idxes_list': x_idxes_list_new
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
