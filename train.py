"""
Let's try with getting an rnn encoder to predict the outputs, and then check
that against the test data first
"""

import torch
import numpy as np
import argparse
import pickle
import myio
from torch import nn, optim, autograd


class Encoder(nn.Module):
    def __init__(self, embeddings, num_layers):
        """
        embeddings should be a torch tensor, of dimension
        max_idx - 1 x num_hidden

        we'll derive num_hidden from the second dimension of embeddings
        """
        super().__init__()
        self.num_hidden = embeddings.shape[1]
        self.num_layers = num_layers
        self.embedding = nn.Embedding(
            embeddings.shape[0],
            self.num_hidden
        )
        self.embedding.weight.data = embeddings
        self.embedding.weight.requires_grad = False
        self.lstm = nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.num_hidden,
            num_layers=num_layers)
        self.initial_state = None
        self.initial_cell = None
        self.linear = nn.Linear(self.num_hidden, 1)  # modeling as a regression

    def forward(self, x):
        """
        x should be [seq_len][batch_size]
        """
        batch_size = x.size()[1]
        # we reuse initial_state and initial_cell, if they havent changed
        # since last time.
        if self.initial_state is None or self.initial_state.size()[1] != batch_size:
            self.initial_state = autograd.Variable(torch.zeros(
                self.num_layers,
                batch_size,
                self.num_hidden
            ))
            self.initial_cell = autograd.Variable(torch.zeros(
                self.num_layers,
                batch_size,
                self.num_hidden
            ))
        x = self.embedding(x)
        x, _ = self.lstm(x, (self.initial_state, self.initial_cell))
        x = x[:, -1, :]
        x = self.linear(x)

        return x


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


def rand_uniform(shape, min_value, max_value):
    return torch.rand(shape) * (max_value - min_value) + min_value


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


def run(
        in_train_file_embedded, aspect_idx, max_train_examples, batch_size, learning_rate,
        in_validate_file_embedded, max_validate_examples, validate_every):
    train_d = load_embedded_data(
        in_filename=in_train_file_embedded,
        max_examples=max_train_examples,
        aspect_idx=aspect_idx)
    validate_d = load_embedded_data(
        in_filename=in_validate_file_embedded,
        max_examples=max_validate_examples,
        aspect_idx=aspect_idx)
    combined = combine_embeddings(
        embeddings_list=[train_d['embedding'], validate_d['embedding']],
        idx_by_word_list=[train_d['idx_by_word'], validate_d['idx_by_word']])
    embedding = combined['embedding']
    num_hidden = combined['num_hidden']
    idx_by_word = combined['idx_by_word']
    words = combined['words']

    # these numbers, ie -0.05 to 0.05 come from
    # https://github.com/taolei87/rcnn/blob/master/code/nn/initialization.py#L79
    unk_idx = idx_by_word['<unk>']
    pad_idx = idx_by_word['<pad>']
    embedding[unk_idx] = rand_uniform((num_hidden,), -0.05, 0.05)
    model = Encoder(embeddings=embedding, num_layers=2)
    params = filter(lambda p: p.requires_grad, model.parameters())
    opt = optim.Adam(params=params, lr=learning_rate)
    epoch = 0
    while True:
        batches_x, batches_y = myio.create_batches(
            x=train_d['x_idxes'], y=train_d['y_aspect'], batch_size=batch_size, padding_id=pad_idx)
        num_batches = len(batches_x)
        epoch_loss = 0
        for b in range(num_batches):
            print('b %s' % b)
            model.zero_grad()
            bx = autograd.Variable(batches_x[b])
            by = autograd.Variable(batches_y[b])
            out = model.forward(bx)
            loss = ((by - out) * (by - out)).sum().sqrt()
            epoch_loss += loss.data[0]
            loss.backward()
            opt.step()
        print('epoch %s loss %.3f' % (epoch, epoch_loss / num_batches))

        def run_validation():
            batches_x, batches_y = myio.create_batches(
                x=validate_d['x_idxes'], y=validate_d['y_aspect'], batch_size=batch_size, padding_id=pad_idx)
            num_batches = len(batches_x)
            epoch_loss = 0
            for b in range(num_batches):
                print('b %s' % b)
                bx = autograd.Variable(batches_x[b])
                by = autograd.Variable(batches_y[b])
                out = model.forward(bx)
                loss = ((by - out) * (by - out)).sum().sqrt()
                epoch_loss += loss.data[0]
            print('validate epoch %s loss %.3f' % (epoch, epoch_loss / num_batches))

        if (epoch + 1) % validate_every == 0:
            run_validation()
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-train-file-embedded', type=str, default='data/reviews.aspect1.train.emb')
    parser.add_argument('--in-validate-file-embedded', type=str, default='data/reviews.aspect1.heldout.emb')
    parser.add_argument('--aspect-idx', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--max-train-examples', type=int, default=0, help='0 means all')
    parser.add_argument('--max-validate-examples', type=int, default=0, help='0 means all')
    parser.add_argument('--validate-every', type=int, default=1, help='after how many epochs run validation')
    parser.add_argument
    args = parser.parse_args()
    run(**args.__dict__)
