#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  8 17:09:53 2021

@author: vasu
"""
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from scipy import stats
import warnings
from warnings import filterwarnings
from sklearn.exceptions import ConvergenceWarning

import random

from Embedding import Embedding

class Analyser:   
    def __init__(self,num_folds,multiclass=False,allembs=False):
        self.num_folds = num_folds
        self.emb = Embedding(allembs)        
        self.Embs = {}
        self.multiclass = multiclass
         
    def CreateFolds(self, Data):
        skf = StratifiedKFold(n_splits=self.num_folds)
        
        self.Folds = []
        assert sorted(Data['Dominant_Topic'].unique())==list(range(len(Data['Dominant_Topic'].unique())))
        for t in Data['Dominant_Topic'].unique():   self.Folds.append([])
        
        for t in Data['Dominant_Topic'].unique():
            seg = Data.loc[Data['Dominant_Topic'] == t]
            
            for train_index, test_index in skf.split(seg, seg['Labels']):
                self.Folds[int(t)].append(seg.iloc[test_index])
                        
    def getEmb(self,sent,embtype):
        if sent not in self.Embs:   self.Embs[sent] = self.emb.getMean(sent)
        return self.Embs[sent][embtype]
                
    def perTopicAnalysis(self,Data,embtype):
        model = MLPClassifier()
        self.CreateFolds(Data)
                
        k = len(Data['Dominant_Topic'].unique())*5
        skf = StratifiedKFold(n_splits=k)
        mbs = 0.0
        fold_index_full = [test_index for _, test_index in skf.split(Data, Data['Labels'])]
        for _ in range(10):
            fold_index = random.sample(fold_index_full,self.num_folds)
            train_index = [x for f in fold_index[:self.num_folds-1] for x in f]
            test_index = fold_index[self.num_folds-1]
            base_trainX = [self.getEmb(sent,embtype) for sent in Data.iloc[train_index]['Sent']]
            base_trainY = [label for label in Data.iloc[train_index]['Labels']]
            base_testX = [self.getEmb(sent,embtype) for sent in Data.iloc[test_index]['Sent']]
            base_testY = [label for label in Data.iloc[test_index]['Labels']]
    
            with warnings.catch_warnings():
                filterwarnings("ignore", category=ConvergenceWarning)
                model.fit(base_trainX,base_trainY)
                if self.multiclass:
                    pred_proba = [proba for proba in model.predict_proba(base_testX)]
                    base_Auc = roc_auc_score(base_testY,pred_proba,multi_class='ovr')
                else:
                    pred_proba = [proba[1] for proba in model.predict_proba(base_testX)]
                    base_Auc = roc_auc_score(base_testY,pred_proba)
                
            mbs += base_Auc
        mbs = mbs / 10

        Auc = [[0 for ts in range(len(self.Folds))] for tr in range(len(self.Folds))]
    
        for tr_topic in range(len(self.Folds)):
            ts_topic = tr_topic
    
            for test_fold in range(self.num_folds):
                trainX = [self.getEmb(sent,embtype) for fold in range(self.num_folds) for sent in self.Folds[tr_topic][fold]['Sent'] if fold!=test_fold]
                trainY = [label for fold in range(self.num_folds) for label in self.Folds[tr_topic][fold]['Labels'] if fold!=test_fold]
                with warnings.catch_warnings():
                    filterwarnings("ignore", category=ConvergenceWarning)
                    model.fit(trainX,trainY)
    
                for ts_topic in range(len(self.Folds)):
                    testX = [self.getEmb(sent,embtype) for sent in self.Folds[ts_topic][test_fold]['Sent']]
                    testY = [label for label in self.Folds[ts_topic][test_fold]['Labels']]
                    
                    if sum(testY)==0 or sum(testY)==len(testY): 
                        print(ts_topic,test_fold, sum(testY))
                        print(Data.groupby(['Dominant_Topic','Labels']).size()[ts_topic])
                    
                    if self.multiclass:
                        Auc[tr_topic][ts_topic] += roc_auc_score(testY,model.predict_proba(testX),multi_class='ovr')
                    else:
                        pred_proba = [proba[1] for proba in model.predict_proba(testX)]
                        Auc[tr_topic][ts_topic] += roc_auc_score(testY,pred_proba)
                             
        seen_score = [Auc[t][t]/self.num_folds for t in range(len(self.Folds))]
        unseen_score = [(sum(Auc[t]) - Auc[t][t]) / (self.num_folds * (len(self.Folds)-1)) for t in range(len(self.Folds))]
        
        ttest = list(stats.ttest_rel(seen_score,unseen_score))
    
        scores = [[str(len(self.Folds))+'_'+str(i),seen,unseen,seen-unseen] for i, (seen,unseen) in enumerate(zip(seen_score,unseen_score))]
        return scores,mbs,ttest
    
    
        
        