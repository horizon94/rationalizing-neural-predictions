# rationalizing-neural-predictions

pytorch implementation of Tao Lei's "Rationalizing Neural Predictions", https://arxiv.org/abs/1606.04155

Note: Tao's original implementation for Theano is at: https://github.com/taolei87/rcnn/tree/master/code/rationale

## word vectors

I am trying using glove.6B.zip, from https://nlp.stanford.edu/projects/glove/, which I unzipped, and then used the 200-dimensional version inside, ie glove.6B.200d.txt

## Differences from Tao's paper

- he uses RCNN, I'm just using LSTM
- probably a bunch of other stuff :P
