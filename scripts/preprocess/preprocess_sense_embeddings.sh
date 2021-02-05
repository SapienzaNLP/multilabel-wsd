#!/bin/bash

python scripts/preprocess/preprocess_sense_embeddings.py \
    --lmms ../../Data/embeddings/lmms/lmms-2048.txt \
    --sensembert ../../Data/embeddings/sensembert/sensembert_EN_supervised.txt \
    --output embeddings/synset_embeddings.txt \
    --output_size 512 \
    --log DEBUG