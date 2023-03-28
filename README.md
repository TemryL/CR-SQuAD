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
- python==3.10.10
- numpy==1.22.3
- nltk==3.7
- scikit-learn==1.0.2
- pytorch==2.0.0
- transformers==4.27.3

## Repo Architecture
<pre>  
├─── dataset
    ├─── dev-v1.1.json
    ├─── dev-v2.0.json
    ├─── train-v1.1.json
    ├─── train-v2.0.json
├─── initialize.py: Python script to initialize TF-IDF and BM25 based models. Models are saved as pickle files in `pretrained` folder.
├─── pretrained
    ├─── bm25.pkl: Initialized BM25 based model.
    ├─── tf_idf.pkl: Initialized TF-IDF based model.
├─── README.md: README
├─── references
    ├─── BERT.pdf: Original paper of BERT model.
    ├─── DPR_for_QA.pdf: Original paper of FaceBook DPR based on BERT.
├─── requirements.txt: requirements
├─── retrieve.py: Pyhton script to retrieve a context from a given question.
├─── src
    ├─── __init__.py: File to define src directory as a python package.
    ├─── bert.py: Implementation of Bi-Encoder BERT based retriever.
    ├─── bm25.py: Implementation of OKAPI BM25 based retriever.
    ├─── data.py: Implemetation of data loading and Dataset class for SQuAD.
    ├─── retriever.py: Implementation of parent class of each retriever.
    ├─── tf_idf.py: Implementation of TF-IDF based retriever.
</pre>

## Instructions to run 
First make sure to have all the requirements.

The following commands give more details about the positional arguments and a description of the process done while running:

```
python initialize.py -h
python retrieve.py -h
```
Please run them before running the following. The commands showed bellow have to be executed in the same order to keep consistency.

To initialize and get the accuracy of TF-IDF and BM25 based model run:
```
python initialize.py model_type dataset
```
Beware that initializing the model can take few minutes and will produce pickle files in the `pretrained` folder.

To retrieve a context from a given question run the following:
```
python retrieve.py model_type question
```

# Results
The accuracies of the models on SQuAD v-1.1 Validation Set (dev-v1.1.json) are reported bellow. The accuracy is simply computed as the percentage of context correctly (exact match with target) retrieved over the whole dataset. A retrieved context that contains the answer of the question but that was originally not the context associated to the question in the dataset are counted as non-correct. Calculating the accuracy this way make the task harder. 

| Model Type | Top-1 Accuracy |     
|-------------------|------------|
| TF-IDF | 59.25% |       
| BM25 | 77.60% |       
| BERT | ? |       

