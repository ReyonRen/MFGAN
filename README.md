# MFGAN
Code for our SIGIR 2020 paper:

>Sequential Recommendation with Self-Attentive Multi-Adversarial Network.   
>[Paper](https://dl.acm.org/doi/10.1145/3397271.3401111)

<img src="https://github.com/RUCAIBox/MFGAN/blob/master/model.png" width="750" alt="model"/>


## Introduction
In this paper, we have proposed a Multi-Factor Generative Adversarial Network (MFGAN) for sequential recommendation. In our
framework, the generator taking user behavior sequences as input is
used to generate possible next items, and multiple factor-specific discriminators are used to evaluate the generated sub-sequence from
the perspectives of different factors.

## Setup
This code is based on [SASREC](https://github.com/kang205/SASRec).

### Requirements
* Tensorflow 1.2.0
* Python 3.6

### Data
<img src="https://github.com/RUCAIBox/MFGAN/blob/master/dataset.png" width="450" alt="dataset"/>

For the training data of the generator, each line contains an user id and item id meaning an interaction. 

As for the training data of the discriminators, it contains n files if the dataset have n attributes. In an attribute file, each line contains an attribute of a item, which depends on how the attribute information defined.

## Model training
`python main.py --dataset="generator training set name" --train_dir=default`

## Reference

```
@inproceedings{DBLP:conf/sigir/RenLLZWDW20,
  author    = {Ruiyang Ren,
               Zhaoyang Liu,
               Yaliang Li,
               Wayne Xin Zhao,
               Hui Wang,
               Bolin Ding,
               Ji{-}Rong Wen},                 
  title     = {Sequential Recommendation with Self-Attentive Multi-Adversarial Network},  
  booktitle = {Proceedings of the 43rd International {ACM} {SIGIR} conference on
               research and development in Information Retrieval, {SIGIR} 2020, Virtual
               Event, China, July 25-30, 2020},                 
  year      = {2020},    
}
```
