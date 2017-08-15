"""
Let's try with getting an rnn encoder to predict the outputs, and then check
that against the test data first
"""

import torch
import argparse
from torch import nn, optim, autograd
import torch.nn.functional as F
import myio
import embeddings_helper


def rand_uniform(shape, min_value, max_value):
    return torch.rand(shape) * (max_value - min_value) + min_value


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
            if x.is_cuda:
                self.initial_state = self.initial_state.cuda()
                self.initial_cell = self.initial_cell.cuda()
        x = self.embedding(x)
        x, _ = self.lstm(x, (self.initial_state, self.initial_cell))
        x = x[:, -1, :]
        x = self.linear(x)

        return x


class Generator(nn.Module):
    """
    This will be the 'independent' form for now

    It's basically almost identical to the Encoder, except:
    - bidirectional LSTM, not unidirectional
    - the linear runs at each word position, so there are:
    - ... outputs given for each position
    - sigmoid squashed
    - we then sample the words based on the linear output, and return
      the indexes of the selected words

    (ok, finally, quite a few differences :) )

    Some things taht are the same then:
    - embeddings
    - number of layers
    - fact that it contains an rnn, and the type of rnn (RCNN in the paper,
      LSTM here)
    - the fact taht we have in sequence:
        - embedding, then
        - lstm/rnn, then
        - linear
    """
    def __init__(self, embeddings, num_layers, pad_id):
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
            num_layers=num_layers,
            bidirectional=True)
        self.initial_state = None
        self.initial_cell = None
        # self.rationales = None
        self.linear = nn.Linear(self.num_hidden * 2, 1)
        self.pad_id = pad_id

    def forward(self, input):
        """
        x should be [seq_len][batch_size]
        """
        seq_len = input.size()[0]
        batch_size = input.size()[1]
        # we reuse initial_state and initial_cell, if they havent changed
        # since last time.
        if self.initial_state is None or self.initial_state.size()[1] != batch_size:
            self.initial_state = autograd.Variable(torch.zeros(
                self.num_layers * 2,
                batch_size,
                self.num_hidden
            ))
            self.initial_cell = autograd.Variable(torch.zeros(
                self.num_layers * 2,
                batch_size,
                self.num_hidden
            ))
            if input.is_cuda:
                self.initial_state = self.initial_state.cuda()
                self.initial_cell = self.initial_cell.cuda()
        x = self.embedding(input)
        x, _ = self.lstm(x, (self.initial_state, self.initial_cell))
        x = self.linear(x)
        x = F.sigmoid(x)
        rationale_selected_node = torch.bernoulli(x)
        rationale_selected = rationale_selected_node.view(seq_len, batch_size)
        rationale_lengths = rationale_selected.sum(dim=0).int()
        max_rationale_length = rationale_lengths.max()
        # if self.rationales is None or self.rationales.shape[1] != batch_size:
        rationales = torch.LongTensor(max_rationale_length.data[0], batch_size)
        if input.is_cuda:
            rationales = rationales.cuda()
        rationales.fill_(self.pad_id)
        for n in range(batch_size):
            this_len = rationale_lengths[n].data[0]
            rationales[:this_len, n] = torch.masked_select(
                input[:, n].data, rationale_selected[:, n].data.byte()
            )
        return rationale_selected_node, rationale_selected, rationales, rationale_lengths


def run(
        in_train_file_embedded, aspect_idx, max_train_examples, batch_size, learning_rate,
        in_validate_file_embedded, max_validate_examples, validate_every,
        sparsity, coherence, use_cuda):
    train_d = embeddings_helper.load_embedded_data(
        in_filename=in_train_file_embedded,
        max_examples=max_train_examples,
        aspect_idx=aspect_idx)
    validate_d = embeddings_helper.load_embedded_data(
        in_filename=in_validate_file_embedded,
        max_examples=max_validate_examples,
        aspect_idx=aspect_idx)
    combined = embeddings_helper.combine_embeddings(
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
    torch.manual_seed(123)
    embedding[unk_idx] = rand_uniform((num_hidden,), -0.05, 0.05)

    enc = Encoder(embeddings=embedding, num_layers=2)
    gen = Generator(embeddings=embedding, num_layers=2, pad_id=pad_idx)
    if use_cuda:
        enc.cuda()
        gen.cuda()
        embedding = embedding.cuda()

    params = filter(lambda p: p.requires_grad, set(enc.parameters()) | set(gen.parameters()))
    opt = optim.Adam(params=params, lr=learning_rate)
    epoch = 0
    while True:
        batches_x, batches_y = myio.create_batches(
            x=train_d['x_idxes'], y=train_d['y_aspect'], batch_size=batch_size, padding_id=pad_idx)
        num_batches = len(batches_x)
        epoch_loss = 0
        print('    t', end='', flush=True)
        for b in range(num_batches):
            # print('b %s' % b)
            print('.', end='', flush=True)
            if (b + 1) % 70 == 0:
                print(b)
                print('    t', end='', flush=True)
            gen.zero_grad()
            enc.zero_grad()
            bx = autograd.Variable(batches_x[b])
            by = autograd.Variable(batches_y[b])
            if use_cuda:
                bx = bx.cuda()
                by = by.cuda()
            seq_len = bx.size()[0]
            batch_size = bx.size()[1]

            # print('bx.shape', bx.data.shape)
            rationale_selected_node, rationale_selected, rationales, rationale_lengths = gen.forward(bx)
            # print('rationales.shape', rationales.shape)
            out = enc.forward(rationales)
            loss_mse = ((by - out) * (by - out)).sum().sqrt()
            loss_z1 = rationale_lengths.sum().float()
            loss_transitions = (rationale_selected[1:] - rationale_selected[:-1]).abs().sum().float()
            loss = loss_mse + sparsity * loss_z1 + coherence * loss_transitions
            rationale_selected_node.reinforce(-loss.data[0])
            loss.backward(rationale_selected_node)
            opt.step()
            # epoch_loss += loss.data[0]
            epoch_loss += loss_mse.data[0]
        print(num_batches)

        def run_validation():
            batches_x, batches_y = myio.create_batches(
                x=validate_d['x_idxes'], y=validate_d['y_aspect'], batch_size=batch_size, padding_id=pad_idx)
            num_batches = len(batches_x)
            epoch_loss = 0
            print('    v', end='', flush=True)
            for b in range(num_batches):
                # print('b %s' % b)
                print('.', end='', flush=True)
                if (b + 1) % 70 == 0:
                    print(b)
                    print('    v', end='', flush=True)
                bx = autograd.Variable(batches_x[b])
                by = autograd.Variable(batches_y[b])
                if use_cuda:
                    bx = bx.cuda()
                    by = by.cuda()
                rationale_selected_node, rationale_selected, rationales, rationale_lengths = gen.forward(bx)
                out = enc.forward(rationales)
                loss = ((by - out) * (by - out)).sum().sqrt()
                epoch_loss += loss.data[0]
            print(num_batches)
            return epoch_loss / num_batches

        if (epoch + 1) % validate_every == 0:
            validation_loss = run_validation()
            print('epoch %s train loss %.3f validate loss %.3f' % (epoch, epoch_loss / num_batches, validation_loss))
            # print('    validate loss %.3f' % (epoch_loss / num_batches))
        else:
            print('epoch %s train loss %.3f' % (epoch, epoch_loss / num_batches))
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
    parser.add_argument('--sparsity', type=float, default=0.0003)
    parser.add_argument('--coherence', type=float, default=2.0)
    parser.add_argument('--use-cuda', action='store_true')
    parser.add_argument
    args = parser.parse_args()
    run(**args.__dict__)
