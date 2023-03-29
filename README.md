# Context Retrieval on SQuAD Dataset

* [Getting started](#getting-started)
    * [Description](#project-description)
    * [Data](#data)
* [Reproduce results](#reproduce-results)
    * [Requirements](#Requirements)
    * [Repo Architecture](#repo-architecture)
    * [Instructions to run](#instructions-to-run)
* [Results](#results)

# Getting started

## Description
The motivation of context retrieval for question answering is to efficiently identify relevant passages that contain the answer to a given question. In this repository, retriever based on TF-IDF, OKAPI BM25 and BERT models are implemented and tested on SQuAD dataset.

## Data
Context retrieval is performed on SQuaD dataset. The Stanford Question Answering Dataset (SQuAD) is a collection of over 100,000 question-answer pairs based on articles from Wikipedia. The version 1.1 and 2.0 of SQuAD are already available in the `dataset` folder.

# Reproduce results
## Requirements
- python==3.7.7
- numpy==1.21.6
- nltk==3.7
- scikit-learn==1.0.2
- torch==1.13.1
- transformers==4.27.3

## Repo Architecture
<pre>  
├─── dataset
    ├─── dev-v1.1.json
    ├─── dev-v2.0.json
    ├─── train-v1.1.json
    ├─── train-v2.0.json
├─── initialize.py: Python script to initialize TF-IDF, BM25 and BERT based retrievers. Retrievers are saved as pickle files in `retrievers` folder.
├─── pretrained
    ├─── bi_encoder.pth: Pretrained BiEncoder BERT based model.
├─── README.md: README
├─── references
    ├─── BERT.pdf: Original paper of BERT model.
    ├─── DPR_for_QA.pdf: Original paper of FaceBook DPR based on BERT.
├─── requirements.txt: requirements
├─── retrieve.py: Pyhton script to retrieve a context from a given question.
├─── retrievers
    ├─── bert.pkl: Initialized BERT based retriever.
    ├─── bm25.pkl: Initialized BM25 based retriever.
    ├─── tf_idf.pkl: Initialized TF-IDF based retriever.
├─── src
    ├─── __init__.py: File to define src directory as a python package.
    ├─── bert.py: Implementation of Bi-Encoder BERT based retriever.
    ├─── bm25.py: Implementation of OKAPI BM25 based retriever.
    ├─── data.py: Implementation of data loading and Dataset class for SQuAD.
    ├─── retriever.py: Implementation of parent class of each retriever.
    ├─── tf_idf.py: Implementation of TF-IDF based retriever.
├─── train.py: Python script to train the BiEncoder BERT based model. Produce log every epoch in `pretrained` folder.
</pre>

## Instructions to run 
First make sure to have all the requirements. To start retrieving without training and initializing the retrievers, please download the `retrievers` folder [here]() and place it in the root of the repository (see repo architecture).

The following commands give more details about the positional arguments and a description of the process done while running:

```
python train.py -h
python initialize.py -h
python retrieve.py -h
```
Please run them before running the following. The commands showed bellow have to be executed in the same order to keep consistency. 

To retrieve a context from a given question run the following:
```
python retrieve.py model_type question
```

`YOU DO NOT NEED` to execute the following commands if you already have downloaded and copied the `retrievers` folder in the root of the repo.

To train from scratch the BiEncoder BERT based model run:
```
python train.py data_path nb_epochs batch_size  
```
Training BERT model is a long process and should be done on GPUs. 

To initialize and get the accuracy of TF-IDF, BM25 and BERT based model run:
```
python initialize.py model_type data_path
```
Beware that initializing the model can take few minutes and will produce pickle files in the `retrievers` folder. 

# Results
The accuracies of the models on SQuAD v-1.1 Validation Set (dev-v1.1.json) are reported bellow. The accuracy is simply computed as the percentage of context correctly (exact match with target) retrieved over the whole dataset. A retrieved context that contains the answer of the question but that was originally not the context associated to the question in the dataset are counted as non-correct. Calculating the accuracy this way make the task harder. 

| Model Type | Top-1 Accuracy |     
|-------------------|------------|
| TF-IDF | 59.25% |       
| BM25 | 77.60% |       
| BERT | ? |       

