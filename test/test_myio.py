"""
check we are able to read the data ok

designed to be run using py.test
"""
import myio


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


def test_embedding():
    # embedding = 'data/GoogleNews-vectors-negative300.bin.gz'
    # embedding = 'data/glove.6B.zip'
    embedding = 'data/glove.6B.200d.txt'
    it = myio.load_embedding_iterator(embedding)
    i = 0
    for word, vals in it:
        print('[%s]' % word, 'len(vals)', len(vals), 'vals[:6]', vals[:6])
        i += 1
        if i > 10:
            break
