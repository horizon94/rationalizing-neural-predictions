# rationalizing-neural-predictions

pytorch implementation of Tao Lei's "Rationalizing Neural Predictions", https://arxiv.org/abs/1606.04155

Note: Tao's original implementation for Theano is at: https://github.com/taolei87/rcnn/tree/master/code/rationale

## word vectors

I am trying using glove.6B.zip, from https://nlp.stanford.edu/projects/glove/, which I unzipped, and then used the 200-dimensional version inside, ie glove.6B.200d.txt

## Differences from Tao's paper

- he uses RCNN, I'm just using LSTM
- RNP includes a 'dependent' implementation of the generator: this repo provides only the 'independent' version
- no dropout (yet)
- I'm training using `batch-size` 128 (though you can try with batch size 256, if your GPU has enough memory, of course)
- probably a bunch of other stuff :P

## To use

- install pytorch 0.2, and activate it, eg, on Ubuntu:
```
wget https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh -O Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/conda
source ~/conda/bin/activate
conda create -n pytorch
source activate pytorch
conda install pytorch cuda80 -c soumith
```
- run `bin/getdata.sh` to download the beer reviews data, and glove vectors
- run `preprocess.py` to combine the beer review data and the glove vectors:
```
export PYTHONPATH=.
python preprocess.py
python preprocess.py --in-train-file data/reviews.aspect1.heldout.txt.gz --out-train-file-embedded data/reviews.aspect1.heldout.emb
```
- run `train.py` (in progress...)
```
python train.py --use-cuda --max-validate-examples 128 --batch-size 128
```
(you can vary the number of trainin/test examples, depending on how confident it's working. for now I'm still fixing stuff :P)

## Environment

Tested/developed(ing) using:
- ~~Mac OS X Sierra~~ Ubuntu 16.04
- NVIDIA M60 gpu, (using aws ec2 g3.xlarge instance)
- Python 3.6.2
- pytorch 0.2

## Example results

Using commandline above, ie full training set, and batch size 128:
```
    v.    [poured a cloudy yellow , orange with a medium white head and no lacing . huge fruity nose with lots of pineapple . tasted pretty flat which makes me think i may have gotten a bad bottle . had more of a wheat beer feel to it than and definitely not an american pale ale . pretty sour which also makes me think it may have been a bad bottle . hugely disappointing considering how much i had heard about hair of the dog .]
    [it 's been a while since i 've reviewed , and i wanted to get the ball rolling with the right equipment . the ten dollar , 12oz bottle at the beerstore was an obvious choice . probably the most expensive beer i have ever bought at retail by volume . but hey , still cheaper than moderately priced wine ! used a tulip glass for this special occasion beer . pours a dark ruby-mulch color . exceptional head retention , leaves a paper-thin sheet of head on the side of the glass . smell is very focused and potent , sweet , toasty , and chocolaty . the smell is strong enough smell to sample without bringing to my nose . definitely a sipping beer . obviously , alcohol taste is dominant , as is the malt profile . tingly and warming , this beer is not unlike brandy . though a very occasional smoker , this is probably the number one beer i would pick to pair with a cigar . i could n't drink this everyday , as it would kick my ass , but a definite must-sample for any beer lover .]
    [pours an orangey golden colour . head is huge and frothy . sudsy lacing . boisterous bubbling . very vibrant and alive , which is what you want in a saison . looks pretty interesting . nose is big and funky , with huge citrus characters , and a really nice almost smoky malt character on the back . something solid and very grounded connect with the light and airy saison characteristics . bit of pepper coming through as well . it has some really pleasant saison characters . perhaps not as well integrated as some of the best examples , but the malt character grounds it nicely . taste is unfortunately quite thin , and rather insipid . some spice on the front , with a lingering acerbic bitterness and a touch of acidity throughout . but there 's no depth to it , no connection with the malt as in the nose - indeed , there 's little body or malt on it whatsoever . just a woody acidity , some funk and a green organic character on the back . and that bitterness , which does n't really match with anything else . no , it does n't work for me . oh , no . the bruery has done it again for me . every time i 'm expecting something big , exciting and robust , and there 's always an insipidness to their beers . it 's happened every time so far , and i 'm wondering how long i can]
    [a-full bronze . big light tan head . brilliantly clear . s-some toasty malt . light citrus hop aroma . not a lot going on . t-carmelly malty . firm bitterness but more balanced to the malt than their pale ale . not much in the way of hop flavor . lots of bitterness but not much flavor . m-medium bodied . medium carbonation . no alcohol burn . d-an average tasting ipa . needs more hop flavor and aroma . bitterness and malt are spot on . still not a bad beer .]
1/1
epoch 30 train loss 19.541 traintime 105 validate loss 27.446
```
(it crashed at this point. a bug I need to fix :P )
