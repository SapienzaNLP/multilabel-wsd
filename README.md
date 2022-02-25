<div align="center">    
 
# Framing Word Sense Disambiguation as a Multi-Label Problem for Model-Agnostic Knowledge Integration

[![Paper](http://img.shields.io/badge/paper-ACL--anthology-B31B1B.svg)](https://www.researchgate.net/publication/349058027_Framing_Word_Sense_Disambiguation_as_a_Multi-Label_Problem_for_Model-Agnostic_Knowledge_Integration)
[![Conference](http://img.shields.io/badge/conference-EACL--2021-4b44ce.svg)](https://2021.eacl.org/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)

</div>

## Description
This is the repository for the paper [*Framing Word Sense Disambiguation as a Multi-Label Problem for Model-Agnostic Knowledge Integration*](https://www.researchgate.net/publication/349058027_Framing_Word_Sense_Disambiguation_as_a_Multi-Label_Problem_for_Model-Agnostic_Knowledge_Integration),
to be presented at EACL 2021 by [Simone Conia](https://c-simone.github.io) and [Roberto Navigli](http://wwwusers.di.uniroma1.it/~navigli/).


## Abstract
> Recent studies treat Word Sense Disambiguation (WSD) as a single-label classification problem in which one is asked to
  choose only the best-fitting sense for a target word, given its context. However, gold data labelled by expert annotators
  suggest that maximizing the probability of a single sense may not be the most suitable training objective for WSD, especially
  if the sense inventory of choice is fine-grained. In this paper, we approach WSD as a multi-label classification problem
  in which multiple senses can be assigned to each target word. Not only does our simple method bear a closer resemblance
  to how human annotators disambiguate text, but it can also be extended seamlessly to exploit structured knowledge
  from semantic networks to achieve state-of-the-art results in English all-words WSD.


## Download
You can download a copy of all the files in this repository by cloning the
[git](https://git-scm.com/) repository:

    git clone https://github.com/SapienzaNLP/multilabel-wsd.git

or [download a zip archive](https://github.com/SapienzaNLP/multilabel-wsd/archive/master.zip).

### Model Checkpoint
* [Best Model (Google Drive)](https://drive.google.com/file/d/1Unfrd4432o6Xy89UD2W5unx4BYIpKyqE/view?usp=sharing)
This is the link to download the checkpoint of the best model (1.3GB).
You can unzip this file in `checkpoints/` as follows:
```
multilabel-wsd/
    checkpoints/
        bert-large/
            checkpoint.ckpt
            processor_config.json
            ...
```

## How to run
You'll need a working Python environment to run the code.
The recommended way to set up your environment is through the
[Anaconda Python distribution](https://www.anaconda.com/download/) which
provides the `conda` package manager.
Anaconda can be installed in your user directory and does not interfere with
the system Python installation.

We use `conda` virtual environments to manage the project dependencies in
isolation.
Thus, you can install our dependencies without causing conflicts with your
setup (even with different Python versions).

Run the following command to create a separate environment:

    conda create --name multilabel-wsd python=3.7

And install all required dependencies in it:

    conda activate multilabel-wsd
    conda install pytorch==1.5.0 cudatoolkit=10.1 -c pytorch

    cd multilabel-wsd

    pip install -r requirements.txt
    pip install torch-scatter==2.0.5 -f https://pytorch-geometric.com/whl/torch-1.5.0+${CUDA}.html
    pip install torch-sparse==0.6.5 -f https://pytorch-geometric.com/whl/torch-1.5.0+${CUDA}.html

where `${CUDA}` should be replaced by either `cpu`, `cu92`, `cu101`, `cu102`, or `cu110` depending on your PyTorch installation.

### Getting the data
We use two main sources of data: the Unified Evaluation Framework for WSD and the Princeton WordNet Gloss Corpus (WNGC).
* The Unified Evaluation Framework for WSD is required to train and evaluate this model. It contains SemCor, the standard
training corpus for WordNet-based WSD. It also contains several evaluation datasets from previous SemEval (and Senseval) tasks.
You can download the data [here](http://lcl.uniroma1.it/wsdeval/).
* WNGC is often used as an additional source of training data. The official website is [here](). We use the preprocessed data 
available [here](https://github.com/SapienzaNLP/ewiser/tree/master/res/corpora/training/preprocessed).

Once you have downloaded the data, place it in `data/original` and run the scripts:
```
bash scripts/preprocess/preprocess_raganato.sh
bash scripts/preprocess/preprocess_glosses.sh
```
**Note:** Make sure that the datasets are renamed as specified in `preprocess_raganato.sh` and `preprocess_glosses.sh`.

### Train a model
You can train a model from scratch using the following command:

    python3 train.py \
        --name bert-large \
        --language_model bert-large-cased
  
where `--name` indicates the name of the experiment and `--language_model` indicates the name of the underlying language model
to use. The model supports most of the BERT-based models from the Huggingface's Transformers library.

If you want to train the model to include relational knowledge from WordNet, you can use the following flags:

    python3 train.py --name bert-large --language_model bert-large-cased \
        --include_similar \
        --include_related \
        --include_verb_groups \
        --include_also_see \
        --include_hypernyms \
        --include_hyponyms \
        --include_instance_hypernyms \
        --include_instance_hyponyms

If you want to train the model with a different training dataset (or development dataset):

    python3 train.py \
        --name bert-large \
        --language_model bert-large-cased \
        --train_path path_to_training_set \
        --dev_path path_to_development_set

By default the training script assumes that the training dataset is located at `data/preprocessed/semcor/semcor.json`
and the development dataset is located at `data/preprocessed/semeval2007/semeval2007.json`.

### Evaluate a model
You can evaluate the model on a dataset using the following command:

    python3 evaluate.py \
        --model checkpoint.ckpt \
        --processor config.json \
        --model_input preprocessed_dataset.json \
        --model_output predictions.txt \
        --evaluation_input gold_keys.txt

The command loads `checkpoint.ckpt` (and its configuration `config.json`), runs the model to obtain the predictions on the instances contained
in `processed_dataset.json`, writes the predictions in `predictions.txt` and computes the overall scores compared to `gold_keys.txt`.

If you have downloaded the checkpoint above, you should be able to reproduce the results of the best model in the paper.
```
# Output of the evaluation script
Accuracy    = 80.201% (5817/7253)
NOUNs       = 82.884% (3564/4300)
VERBs       = 70.278% (1161/1652)
ADJs        = 83.351% (796/955)
ADVs        = 85.549% (296/346)
```

## Cite this work
If you use any part of this work, please consider citing the paper as follows:

```
@inproceedings{conia-navigli-2021-multilabel-wsd,
    title      = "Framing {W}ord {S}ense {D}isambiguation as a Multi-Label Problem for Model-Agnostic Knowledge Integration",
    author     = "Conia, Simone and Navigli, Roberto",
    booktitle  = "Proceedings of the 16th Conference of the European Chapter of the Association for Computational Linguistics: Main Volume",
    month      = apr,
    year       = "2021",
    address    = "Online",
    publisher  = "Association for Computational Linguistics",
    url        = "https://aclanthology.org/2021.eacl-main.286",
    doi        = "10.18653/v1/2021.eacl-main.286",
    pages      = "3269--3275",
}
```