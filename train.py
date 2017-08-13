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


def rand_uniform(shape, min_value, max_value):
    return torch.rand(shape) * (max_value - min_value) + min_value


def run(in_train_file_embedded, aspect_idx, max_train_examples, batch_size, learning_rate):
    print('loading training data...')
    with open(in_train_file_embedded, 'rb') as f:
        d = pickle.load(f)
    print(' ... loaded')
    # reminder, d is: embedding, idx_by_word, words, x, y, x_idxes
    embedding = d['embedding']
    y = d['y']
    # print('y.shape', y.shape)
    # print('y[0]', y[0])
    x_idxes = d['x_idxes']
    x = d['x']
    idx_by_word = d['idx_by_word']
    if max_train_examples > 0:
        x_idxes = x_idxes[:max_train_examples]
        y = y[:max_train_examples]
        x = x[:max_train_examples]
    N = len(x_idxes)
    y_aspect = torch.zeros(N)
    for n, yv in enumerate(y):
        y_aspect[n] = yv[aspect_idx].item()
    print('y_aspect.shape', y_aspect.shape)
    print('y_aspect[:5]', y_aspect[:5])
    print('num training examples', len(x_idxes))
    # handle unk
    unk_idx = idx_by_word['<unk>']
    num_hidden = embedding.shape[1]

    # these numbers, ie -0.05 to 0.05 come from
    # https://github.com/taolei87/rcnn/blob/master/code/nn/initialization.py#L79
    embedding[unk_idx] = rand_uniform((num_hidden,), -0.05, 0.05)
    model = Encoder(embeddings=embedding, num_layers=2)
    pad_idx = idx_by_word['<pad>']
    # print('len(batches_x)', len(batches_x))
    # print('len(batches_y)', len(batches_y))
    # print('batches_x[0][:3]', batches_x[0][:3])
    # print('batches_y[0][:3]', batches_y[0][:3])
    params = filter(lambda p: p.requires_grad, model.parameters())
    opt = optim.Adam(params=params, lr=learning_rate)
    epoch = 0
    while True:
        batches_x, batches_y = myio.create_batches(x=x_idxes, y=y_aspect, batch_size=batch_size, padding_id=pad_idx)
        num_batches = len(batches_x)
        epoch_loss = 0
        for b in range(num_batches):
            bx = autograd.Variable(batches_x[b])
            by = autograd.Variable(batches_y[b])
            out = model.forward(bx)
            loss = ((by - out) * (by - out)).sum().sqrt()
            epoch_loss += loss
            loss.backward()
            opt.step()
            # print('loss %.3f' % loss)
        print('epoch %s loss %.3f' % (epoch, epoch_loss))
        epoch += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-train-file-embedded', type=str, default='data/reviews.aspect1.train.emb')
    parser.add_argument('--aspect-idx', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--max-train-examples', type=int, default=0, help='0 means all')
    parser.add_argument
    args = parser.parse_args()
    run(**args.__dict__)
