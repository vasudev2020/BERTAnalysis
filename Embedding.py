#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:02:37 2021

@author: vasu
"""
import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel

class Embedding:
    def __init__(self,emb,layer=11):
        self.emb = emb
        if emb=='BERT':
            self.bt = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.bert.eval()
            self.layer = layer
        if emb=='Glove':
            f = open('../Data/glove.840B.300d.txt','r')
            self.gloveModel = {}
            for line in f:
                splitLines = line.split()
                if len(splitLines)!=301:    continue
                word = splitLines[0]
                wordEmbedding = np.array([float(value) for value in splitLines[1:]])
                self.gloveModel[word] = wordEmbedding
            print(len(self.gloveModel)," words loaded!")
            
    def getMean(self,sent):
        if self.emb=='Glove':
            embs = [self.gloveModel[w] for w in sent.split() if w in self.gloveModel]
            emb = np.mean(embs,axis=0)
            assert len(emb)==300
            return emb
        if self.emb=='BERT':
            tt = self.bt.tokenize("[CLS] "+sent+" [SEP]")
            it = self.bt.convert_tokens_to_ids(tt)

            with torch.no_grad():   encoded_layers, _ = self.bert(torch.tensor([it]), torch.tensor([[1]*len(tt)]))
    
            return torch.mean(encoded_layers[self.layer], 1)[0].numpy()
        if self.emb=='Rand':
            return np.random.randn(768)