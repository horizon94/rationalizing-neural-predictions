#!/bin/bash

function maybe_wget {
    if [[ ! -f $2 ]]; then {
        wget $1
    } fi
}

mkdir data
cd data
maybe_wget http://people.csail.mit.edu/taolei/beer/annotations.json annotations.json
maybe_wget http://people.csail.mit.edu/taolei/beer/review+wiki.filtered.200.txt.gz review+wiki.filtered.200.txt.gz

for n in 1 2 3; do {
    maybe_wget http://people.csail.mit.edu/taolei/beer/reviews.aspect${n}.heldout.txt.gz reviews.aspect${n}.heldout.txt.gz
    maybe_wget http://people.csail.mit.edu/taolei/beer/reviews.aspect${n}.train.txt.gz reviews.aspect${n}.train.txt.gz
} done

maybe_wget http://people.csail.mit.edu/taolei/beer/select.py select.py

maybe_wget http://nlp.stanford.edu/data/glove.6B.zip glove.6B.zip

unzip glove.6B.zip

for filename in $(ls *.gz); do {
    gunzip -k ${filename}
} done

echo done
