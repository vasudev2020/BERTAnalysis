#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 13:33:25 2021

@author: vasu
"""

import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel


class BERT:
    def __init__(self):       
        self.bt = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.eval()

        
    def getMean(self,sent):
        tt = self.bt.tokenize("[CLS] "+sent+" [SEP]")
        it = self.bt.convert_tokens_to_ids(tt)

        with torch.no_grad():   encoded_layers, _ = self.bert(torch.tensor([it]), torch.tensor([[1]*len(tt)]))
    
        return torch.mean(encoded_layers[11], 1)[0].numpy()