"""
check we are able to read the data ok

designed to be run using py.test
"""
import myio
import time
import pickle


def test_basic():
    # code adapted from Tao's `rationale.py`:
    train = 'data/reviews.aspect1.train.txt.gz'
    train_x, train_y = myio.read_annotations(train)
    # train_x = [embedding_layer.map_to_ids(x)[:max_len] for x in train_x]

    dev = 'data/reviews.aspect1.heldout.txt.gz'
    dev_x, dev_y = myio.read_annotations(dev)
    # dev_x = [embedding_layer.map_to_ids(x)[:max_len] for x in dev_x]

    load_rationale = 'data/annotations.json'
    rationale_data = myio.read_rationales(load_rationale)
    # for x in rationale_data:
    #     x["xids"] = embedding_layer.map_to_ids(x["x"])


def test_embedding_iter1():
    # embedding = 'data/GoogleNews-vectors-negative300.bin.gz'
    # embedding = 'data/glove.6B.zip'
    embedding = 'data/glove.6B.200d.txt'
    it = myio.load_embedding_iterator(embedding)
    i = 0
    # vals_by_word = {}
    # last = time.time()
    for word, vals in it:
        # vals_by_word[word] = vals
        # if time.time() - last >= 1.0:
        #     print(len(vals_by_word))
        #     last = time.time()
        print('[%s]' % word, 'len(vals)', len(vals), 'vals[:6]', vals[:6])
        i += 1
        if i > 10:
            break


def test_embedding_iter2():
    # embedding = 'data/GoogleNews-vectors-negative300.bin.gz'
    # embedding = 'data/glove.6B.zip'
    embedding = 'data/glove.6B.200d.txt'
    it = myio.load_embedding_iterator(embedding)
    i = 0
    # vals_by_word = {}
    # last = time.time()
    for word, vals in it:
        # vals_by_word[word] = vals
        # if time.time() - last >= 1.0:
        #     print(len(vals_by_word))
        #     last = time.time()
        # print('[%s]' % word, 'len(vals)', len(vals), 'vals[:6]', vals[:6])
        i += 1
        # if i > 10:
            # break
    print('i', i)


def test_load_embeddings():
    embedding = 'data/glove.6B.200d.txt'
    it = myio.load_embedding_iterator(embedding)
    i = 0
    vals_by_word = {}
    start = time.time()
    last = time.time()
    word_list = []
    idx_by_word = {}
    for word, vals in it:
        vals_by_word[word] = vals
        word_list.append(word)
        idx_by_word[word] = len(idx_by_word)
        if time.time() - last >= 1.0:
            print(len(vals_by_word))
            last = time.time()
            # if time.time() - start >= 100.0:
                # break
        # print('[%s]' % word, 'len(vals)', len(vals), 'vals[:6]', vals[:6])
        # i += 1
        # if i > 10:
            # break
    pickle_path = embedding + '.pickle'
    d = {
        'vals_by_word': vals_by_word,
        'word_list': word_list,
        'idx_by_word': idx_by_word
    }
    start = time.time()
    with open(pickle_path, 'wb') as f:
        # pickle.dump(vals_by_word, f)
        pickle.dump(d, f)
    print('time to write pickle', time.time() - start)

    start = time.time()
    with open(pickle_path, 'rb') as f:
        pickle.load(f)
    print('time to load pickle', time.time() - start)
