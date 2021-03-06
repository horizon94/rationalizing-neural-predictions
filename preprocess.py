"""
Take a dataset, convert to word vectors, store in pickle


notes on format:
in the reviewsx.txt file, eg:
[ 0.80000001  1.          0.80000001  0.80000001  0.89999998]
canning craft brew - i 'm a fan . accessability

corresponding review:
https://github.com/jeanlauliac/beerd/blob/master/mockdata/beer-list.txt#L9631
review/appearance: 4
review/aroma: 5
review/palate: 4
review/taste: 4
review/overall: 4.5

=>
0 apperance
1 aroma
2 taste
3 palate
4 overall

Or:

in the text:
[ 0.5         0.2         0.40000001  0.5         0.40000001]
out of most malt liqours , this is probably one of the most tolerable

which is:
https://www.beeradvocate.com/beer/profile/478/6158/?view=beer&sort=&start=75
look: 2.5 | smell: 1 | taste: 2.5 | feel: 2 | overall: 2

so:
0 look
1 smell
2 feel
3 taste
4 overall

[ 0.60000002  0.89999998  0.69999999  0.80000001  0.5       ]
appearance : deep brown color with a thin tan head that quickly dissipated

https://www.beeradvocate.com/beer/profile/144/30806/?ba=Will_Turner
look: 3 | smell: 4.5 | taste: 4 | feel: 3.5 | overall: 2.5

0 => look
1 => smell
2 => feel
3 => taste
4 => overall

"""
import pickle
import argparse
import time
import myio
import torch


def run(in_train_file, out_train_file_embedded, word_vectors, max_len):
    x, y = myio.read_annotations(in_train_file)
    print('len(x)', len(x))
    idx_by_word = {}
    words = []

    # add <unk> and <pad>
    words.append('<pad>')
    words.append('<unk>')
    idx_by_word['<pad>'] = 0
    idx_by_word['<unk>'] = 1

    for n, ex in enumerate(x):
        for word in ex[:max_len]:
            if word not in idx_by_word:
                idx_by_word[word] = len(idx_by_word)
                words.append(word)

    V = len(words)
    it = myio.load_embedding_iterator(word_vectors)
    embedding_vals = [None for i in range(V)]
    for word, vals in it:
        if word in idx_by_word:
            idx = idx_by_word[word]
            nd = len(vals)
            embedding_vals[idx] = vals
    embedding = torch.zeros(V, nd)
    # add unk and pad
    # well, pad is easy, so add unk
    # well... lets leave it for the trainer to do this
    for i, vals in enumerate(embedding_vals):
        if vals is not None:
            embedding[i] = vals
    x_idxes = []
    unk_idx = idx_by_word['<unk>']
    for n, ex in enumerate(x):
        num_words = len(ex[:max_len])
        idxes = torch.LongTensor(num_words)
        idxes.fill_(0)
        for i, word in enumerate(ex[:max_len]):
            if word in idx_by_word:
                idx = idx_by_word[word]
            else:
                idx = unk_idx
            idxes[i] = idx
        # print('idxes.shape', idxes.shape)
        x_idxes.append(idxes)

    d = {
        'embedding': embedding,
        'idx_by_word': idx_by_word,
        'words': words,
        'x': x,
        'y': y,
        'x_idxes': x_idxes
    }
    with open(out_train_file_embedded, 'wb') as f:
        pickle.dump(d, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-train-file', type=str, default='data/reviews.aspect1.train.txt.gz')
    parser.add_argument('--out-train-file-embedded', type=str, default='data/reviews.aspect1.train.emb')
    parser.add_argument('--word_vectors', type=str, default='data/glove.6B.200d.txt')
    parser.add_argument('--max_len', type=int, default=256)
    args = parser.parse_args()
    run(**args.__dict__)
