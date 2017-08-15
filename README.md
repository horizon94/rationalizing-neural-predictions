# rationalizing-neural-predictions

pytorch implementation of Tao Lei's "Rationalizing Neural Predictions", https://arxiv.org/abs/1606.04155

Note: Tao's original implementation for Theano is at: https://github.com/taolei87/rcnn/tree/master/code/rationale

## word vectors

I am trying using glove.6B.zip, from https://nlp.stanford.edu/projects/glove/, which I unzipped, and then used the 200-dimensional version inside, ie glove.6B.200d.txt

## Differences from Tao's paper

- he uses RCNN, I'm just using LSTM
- RNP includes a 'dependent' implementation of the generator: this repo provides only the 'independent' version
- probably a bunch of other stuff :P

## To use

- install pytorch 0.2
- run `bin/getdata.sh` to download the beer reviews data, and glove vectors
- run `preprocess.py` to combine the beer review data and the glove vectors:
```
export PYTHONPATH=.
python preprocess.py
python preprocess.py --in-train-file data/reviews.aspect1.heldout.txt.gz --out-train-file-embedded data/reviews.aspect1.heldout.emb
```
- run `train.py` (in progress...)
```
python train.py --use-cuda --max-train-examples 1024 --max-validate-examples 256
```
(you can vary the number of trainin/test examples, depending on how confident it's working. for now I'm still fixing stuff :P)

## Environment

Tested/developed(ing) using:
- ~~Mac OS X Sierra~~ aws ec2 g3.xlarge instance (NVIDIA M60 gpu)
- Python 3.6.2
- pytorch 0.2
