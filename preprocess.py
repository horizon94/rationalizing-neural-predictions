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
    # words = set()
    x, y = myio.read_annotations(in_train_file)
    print('len(x)', len(x))
    idx_by_word = {}
    words = []
    # vals = []

    for n, ex in enumerate(x):
        for word in ex[:max_len]:
            if word not in idx_by_word:
                idx_by_word[word] = len(idx_by_word)
                words.append(word)
                # vals.append()
            # words.add(word)
        # break

    V = len(words)
    it = myio.load_embedding_iterator(word_vectors)
    embedding_vals = [None for i in range(V)]
    for word, vals in it:
        if word in idx_by_word:
            idx = idx_by_word[word]
            nd = len(vals)
            embedding_vals[idx] = vals
            # break
    # nd = len(embedding_vals[0])
    embedding = torch.zeros(V + 1, nd)
    for i, vals in enumerate(embedding_vals):
        if vals is not None:
            # print('i', i)
            embedding[i] = vals
            # print('embedding[i]', embedding[i])
    # print('embedding', embedding)
    x_idxes = []
    for n, ex in enumerate(x):
        # idxes = []
        num_words = len(ex)
        idxes = torch.LongTensor(num_words)
        idxes.fill_(0)
        for i, word in enumerate(ex[:max_len]):
            if word in idx_by_word:
                idx = idx_by_word[word]
            else:
                idx = V
            idxes[i] = idx
        x_idxes.append(idxes)
    # x = x_new

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


        # print('ex', ex)
        # if n >= 4:
        # break/
    # for i in range(10):
        # print(' '.join(y[i]))
        # print(y[i])
        # print(' '.join(x[i]))
    # print('x[:5]', x[:5])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-train-file', type=str, default='data/reviews.aspect1.train.txt.gz')
    parser.add_argument('--out-train-file-embedded', type=str, default='data/reviews.aspect1.train.emb')
    parser.add_argument('--word_vectors', type=str, default='data/glove.6B.200d.txt')
    parser.add_argument('--max_len', type=int, default=256)
    args = parser.parse_args()
    run(**args.__dict__)
