#!/bin/bash

# Semcor + tagged glosses + examples
python3 train.py --name bert-large --language_model bert-large-cased \
    --train_path data/preprocessed/glosses/semcor.glosses.examples.json \
    --include_similar \
    --include_related \
    --include_verb_groups \
    --include_also_see \
    --include_hypernyms \
    --include_hyponyms