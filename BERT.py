#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:33:25 2021

@author: vasu
"""

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel
import numpy as np


class BERT:
    def __init__(self,layer=11):       
        self.bt = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.eval()
        self.layer = layer

        
    def getMean(self,sent):
        tt = self.bt.tokenize("[CLS] "+sent+" [SEP]")
        it = self.bt.convert_tokens_to_ids(tt)

        with torch.no_grad():   encoded_layers, _ = self.bert(torch.tensor([it]), torch.tensor([[1]*len(tt)]))
    
        return torch.mean(encoded_layers[self.layer], 1)[0].numpy()
    
class Glove:
    def __init__(self):
        
        #print("Loading Glove Model")
        f = open('../Data/glove.840B.300d.txt','r')
        self.gloveModel = {}
        for line in f:
            splitLines = line.split()
            if len(splitLines)!=301:    continue
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            self.gloveModel[word] = wordEmbedding
        print(len(self.gloveModel)," words loaded!")
        #return gloveModel
        
    def getMean(self,sent):        
        embs = [self.gloveModel[w] for w in sent.split() if w in self.gloveModel]
        emb = np.mean(embs,axis=0)
        assert len(emb)==300
        return emb
    
    