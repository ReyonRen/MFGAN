# MFGAN
Code for our SIGIR 2020 paper "Sequential Recommendation with Self-Attentive Multi-Adversarial Network". 
See the paper: https://dl.acm.org/doi/10.1145/3397271.3401111

## Setup
This code is based on [SASREC](https://github.com/kang205/SASRec).

### Requirements
We use the Tensorflow version '1.2.0', and Python version '3.6'.

### Data
First you need the training data of the generator, where each line contains an user id and item id meaning an interaction. 

Then you need the training data of the discriminators, which contain n files if the dataset have n attributes. In an attribute file, each line contain an attribute of a item, which depends on how  define the attribute information.

## Model training
`python main.py --dataset="generator training set name" --train_dir=default`

## Reference
`@inproceedings{DBLP:conf/sigir/RenLLZWDW20,  
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
}`
