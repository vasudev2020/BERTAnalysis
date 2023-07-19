#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 11 09:02:37 2021

@author: vasu
"""
import torch
import numpy as np
from pytorch_pretrained_bert import BertTokenizer, BertModel
from transformers import RobertaModel, AutoTokenizer


class Embedding:
    def __init__(self,allembs=False,roberta=False):
        self.allembs = allembs
        
        self.roberta = roberta
        
        if self.roberta:
            self.rt = AutoTokenizer.from_pretrained("roberta-base")
            self.roberta = RobertaModel.from_pretrained("roberta-base")
        else:
            self.bt = BertTokenizer.from_pretrained('bert-base-uncased')
            self.bert = BertModel.from_pretrained('bert-base-uncased')
            self.bert.eval()
            
        f = open('./data/glove.840B.300d.txt','r')
        self.gloveModel = {}
        for line in f:
            splitLines = line.split()
            if len(splitLines)!=301:    continue
            word = splitLines[0]
            wordEmbedding = np.array([float(value) for value in splitLines[1:]])
            self.gloveModel[word] = wordEmbedding
        print('Loaded glove embeddings with',len(self.gloveModel),"words !")
            
    
    def getMean(self,sent):
        emb = {}
        embs = [self.gloveModel[w] for w in sent.split() if w in self.gloveModel]
        emb['Glove'] = np.mean(embs,axis=0)
        if self.roberta:
            inputs = self.rt(sent, return_tensors="pt")
            if self.allembs:
                encoded_layers = self.roberta(**inputs,output_hidden_states=True).hidden_states[1]
                for layer in range(12):
                    emb['RoBERTa'+str(layer)] = torch.mean(encoded_layers[layer], 1)[0].detach().numpy()
            else:   
                last_hidden_states = self.roberta(**inputs).last_hidden_state
                emb['RoBERTa11'] = torch.mean(last_hidden_states, 1)[0].detach().numpy()
        else:
            tt = self.bt.tokenize("[CLS] "+sent+" [SEP]")
            if len(tt)>512: tt=tt[:512]
            it = self.bt.convert_tokens_to_ids(tt)
    
            with torch.no_grad():   encoded_layers, _ = self.bert(torch.tensor([it]), torch.tensor([[1]*len(tt)]))
    
            if self.allembs:
                for layer in range(12):
                    emb['BERT'+str(layer)] = torch.mean(encoded_layers[layer], 1)[0].numpy()
            else:   emb['BERT11'] = torch.mean(encoded_layers[11], 1)[0].numpy()
            

        emb['Rand']=np.random.randn(768)
        

        
        return emb