"""
Let's try with getting an rnn encoder to predict the outputs, and then check
that against the test data first
"""

import torch
import numpy as np
import argparse
import pickle
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
        self.lstm = nn.LSTM(
            input_size=self.num_hidden,
            hidden_size=self.num_hidden,
            num_layers=num_layers)
        self.initial_state = None
        self.initial_cell = None

    def forward(self, x):
        """
        x should be [seq_len][batch_size]
        """
        batch_size = x.shape[0]
        # we reuse initial_state and initial_cell, if they havent changed
        # since last time.
        if self.initial_state is None or self.initial_state.shape[1] != batch_size:
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
        x = self.lstm(x, (self.initial_state, self.initial_cell))
        return x


def run(in_train_file_embedded, aspect_idx, max_train_examples):
    print('loading training data...')
    with open(in_train_file_embedded, 'rb') as f:
        d = pickle.load(f)
    print(' ... loaded')
    # reminder, d is: embedding, idx_by_word, words, x, y, x_idxes
    embedding = d['embedding']
    y = d['y']
    x_idxes = d['x_idxes']
    x = d['x']
    if max_train_examples > 0:
        x_idxes = x_idxes[:max_train_examples]
        yu = y[:max_train_examples]
        x = x[:max_train_examples]
    print('num training examples', len(x_idxes))
    model = Encoder(embeddings=embedding, num_layers=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--in-train-file-embedded', type=str, default='data/reviews.aspect1.train.emb')
    parser.add_argument('--aspect-idx', type=int, default=0)
    parser.add_argument('--max-train-examples', type=int, default=0, help='0 means all')
    parser.add_argument
    args = parser.parse_args()
    run(**args.__dict__)
